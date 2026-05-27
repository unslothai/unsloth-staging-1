# SPDX-License-Identifier: AGPL-3.0-only
"""Regression check for unsloth-zoo PR #684 against unsloth_zoo/compiler.py.

PR #684 (branch mmathew23/explore/mlx) revives the double-matmul bug
that unsloth-zoo PR #666 (commit f45c31e5) was merged to fix. The bug
lives in the regex template `cross_entropy_replacement_2`: the
dedicated `UNSLOTH_RETURN_LOGITS=1` elif branch that reuses the
already-materialised logits was deleted. Under `UNSLOTH_RETURN_LOGITS=1`
with labels present, control now falls through to the unconditional
`else:` branch which calls `self.lm_head(hidden_states)` a second time
on top of the prepended matmul at compiler.py:2079/2087.

The check below reads `unsloth_zoo/compiler.py` as raw text (no
import) and looks for the dedicated elif branch verbatim. This
sidesteps unsloth_zoo's import-time GPU + triton requirements, so
the test runs on a stock CPU runner with no spoofing.

Expected outcome:
- On unsloth-zoo main:                  PASS (elif present)
- On unsloth-zoo PR #684 (explore/mlx): FAIL (elif removed)
"""
from __future__ import annotations

import importlib.util
import os
import re


def _locate_compiler_py() -> str:
    """Return the absolute path to the installed
    ``unsloth_zoo/compiler.py`` without importing the package."""
    spec = importlib.util.find_spec("unsloth_zoo")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("unsloth_zoo not installed")
    pkg_path = spec.submodule_search_locations[0]
    path = os.path.join(pkg_path, "compiler.py")
    if not os.path.isfile(path):
        raise RuntimeError(f"unsloth_zoo/compiler.py not found at {path}")
    return path


def test_compiler_template_keeps_dedicated_return_logits_elif():
    """The compiler.py template path must keep the dedicated
    UNSLOTH_RETURN_LOGITS=1 elif branch added by zoo PR #666
    (commit f45c31e5). Without it, the template falls through to a
    double-`self.lm_head` matmul `else:` when UNSLOTH_RETURN_LOGITS=1.
    """
    path = _locate_compiler_py()
    src = open(path, encoding="utf-8").read()

    # The template constant we are inspecting.
    template_name = "cross_entropy_replacement_2"
    assert template_name in src, (
        f"{template_name} string constant missing from compiler.py at {path}; "
        "the file shape has changed and the regression check needs an update."
    )

    # Extract the literal text of cross_entropy_replacement_2 (it's a
    # `name = """...""".replace(...)` triple-quoted string).
    body_match = re.search(
        r'cross_entropy_replacement_2\s*=\s*"""(.+?)"""',
        src,
        flags=re.DOTALL,
    )
    assert body_match, "could not locate cross_entropy_replacement_2 body in compiler.py"
    template_body = body_match.group(1)

    # The dedicated UNSLOTH_RETURN_LOGITS=1 elif branch from #666 looks
    # like (verbatim from f45c31e5):
    #
    #   elif self.loss_function.__name__.endswith("ForCausalLMLoss") \
    #        and labels is not None:
    #       # UNSLOTH_RETURN_LOGITS=1 path. Prepended `logits = self.lm_head(...)`
    #       # already materialised the full lm_head matmul; apply the captured logit
    #       # scale/softcap transforms and route loss through self.loss_function on
    #       # those logits instead of letting unsloth_fused_ce_loss redo the matmul.
    #       if (\2) != ():
    #           logits = logits * (\2)
    #       ...
    #       loss = self.loss_function(logits, labels.to(self.lm_head.weight.device), vocab_size=\8, **\9)
    elif_present = re.search(
        r'elif self\.loss_function\.__name__\.endswith\("ForCausalLMLoss"\)\s+'
        r'and labels is not None:\s*\n'
        r'(?:[^\n]*\n){0,3}'
        r'\s*#[^\n]*UNSLOTH_RETURN_LOGITS=1 path',
        template_body,
    )

    if not elif_present:
        # Show a useful failure: print the tail of the template so the
        # reviewer can see the current else: branch the template falls
        # through to.
        else_match = re.search(
            r'(else:\s*\n\s*logits = self\.lm_head\(hidden_states\\1\)[\s\S]*?)$',
            template_body,
            flags=re.MULTILINE,
        )
        else_excerpt = else_match.group(1) if else_match else "<else: branch not found>"
        raise AssertionError(
            "compiler.py template regression detected:\n"
            f"  installed compiler.py: {path}\n"
            "  dedicated UNSLOTH_RETURN_LOGITS=1 elif branch is MISSING from "
            "cross_entropy_replacement_2.\n"
            "  PR #684 (commit 5895b20c) removed the elif; ca086522 only "
            "restored the NOT_RETURN_LOGITS guard on the fused_ce_loss branch.\n"
            "  Net effect: under UNSLOTH_RETURN_LOGITS=1 with labels, the "
            "template now falls through to:\n"
            f"---\n{else_excerpt[:600]}\n---\n"
            "...which calls self.lm_head a second time on top of the prepended "
            "matmul at compiler.py:2079/2087. This is the bug PR #666 fixed."
        )
