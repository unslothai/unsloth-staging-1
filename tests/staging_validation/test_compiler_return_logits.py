# SPDX-License-Identifier: AGPL-3.0-only
"""Regression check for unsloth-zoo PR #684 against unsloth_zoo/compiler.py.

PR #684 (branch mmathew23/explore/mlx) revives the double-matmul bug
that unsloth-zoo PR #666 (commit f45c31e5) was merged to fix. The bug
lives in the regex template `cross_entropy_replacement_2`: the
dedicated `UNSLOTH_RETURN_LOGITS=1` elif branch that reuses the
already-materialised logits was deleted. Under `UNSLOTH_RETURN_LOGITS=1`
with labels present, control now falls through to the unconditional
`else:` branch which calls `self.lm_head(hidden_states)` a SECOND time
on top of the prepended matmul at compiler.py:2079/2087.

The default path (`UNSLOTH_FUSED_FORWARD=1`, AST rewriter) is
unaffected because it produces a single-matmul forward for the
canonical HF triplet. The regression bites only the fallback path:
`UNSLOTH_FUSED_FORWARD=0`, or any model forward whose shape doesn't
match the AST rewriter's triplet.

This test runs `apply_fused_lm_head` on a small set of HF model
families under `UNSLOTH_FUSED_FORWARD=0` and asserts that the
dedicated elif branch is present. Expected outcome:
- On unsloth-zoo main:                  PASS
- On unsloth-zoo PR #684 (explore/mlx): FAIL

The test uses CPU only and a CUDA spoof so it runs on any Linux runner.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import types


def _ensure_cuda_spoof() -> None:
    """Force unsloth_zoo.device_type to capture 'cuda' on CPU runners."""
    try:
        import torch  # noqa: F401
    except Exception:
        return

    # Stub mem_get_info / capability so unsloth_zoo's import-time probes
    # don't crash on a CPU-only runner.
    try:
        import torch.cuda.memory as _cuda_memory  # type: ignore
        _cuda_memory.mem_get_info = lambda *a, **k: (0, 80 * 1024**3)
    except Exception:
        pass
    try:
        import torch
        torch.cuda.get_device_capability = lambda *a, **k: (8, 0)
        torch.cuda.is_bf16_supported = lambda *a, **k: True
    except Exception:
        pass

    package = "unsloth_zoo"
    target = f"{package}.device_type"
    if target in sys.modules:
        return
    pkg_spec = importlib.util.find_spec(package)
    if pkg_spec is None or not pkg_spec.submodule_search_locations:
        return
    pkg_path = pkg_spec.submodule_search_locations[0]

    skeleton_already = package in sys.modules
    if not skeleton_already:
        skel = types.ModuleType(package)
        skel.__path__ = [pkg_path]
        skel.__spec__ = pkg_spec
        skel.__package__ = package
        sys.modules[package] = skel

    try:
        utils_full = f"{package}.utils"
        if utils_full not in sys.modules:
            utils_path = os.path.join(pkg_path, "utils.py")
            utils_spec = importlib.util.spec_from_file_location(utils_full, utils_path)
            utils_mod = importlib.util.module_from_spec(utils_spec)
            sys.modules[utils_full] = utils_mod
            utils_spec.loader.exec_module(utils_mod)

        dt_path = os.path.join(pkg_path, "device_type.py")
        dt_spec = importlib.util.spec_from_file_location(target, dt_path)
        dt_mod = importlib.util.module_from_spec(dt_spec)
        sys.modules[target] = dt_mod

        import torch
        _orig = torch.cuda.is_available
        torch.cuda.is_available = lambda: True  # type: ignore[assignment]
        try:
            dt_spec.loader.exec_module(dt_mod)
        finally:
            torch.cuda.is_available = _orig
    finally:
        if not skeleton_already:
            sys.modules.pop(package, None)


MODELS = [
    "transformers.models.llama.modeling_llama:LlamaForCausalLM",
    "transformers.models.mistral.modeling_mistral:MistralForCausalLM",
    "transformers.models.qwen3.modeling_qwen3:Qwen3ForCausalLM",
    "transformers.models.gemma2.modeling_gemma2:Gemma2ForCausalLM",
]


def _load_attr(dotted: str):
    mod_path, _, attr = dotted.partition(":")
    import importlib
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def test_compiler_template_keeps_dedicated_return_logits_elif():
    """The compiler.py template path must keep the dedicated
    UNSLOTH_RETURN_LOGITS=1 elif branch (the one that reuses the
    prepended logits via self.loss_function), otherwise the template
    falls through to a double-matmul else.
    """
    # Force fallback path so we measure the regex template, not the AST
    # rewriter that runs at unsloth_zoo import time.
    os.environ["UNSLOTH_FUSED_FORWARD"] = "0"

    _ensure_cuda_spoof()

    import inspect

    import unsloth_zoo.compiler as c

    failures = []

    for dotted in MODELS:
        cls = _load_attr(dotted)
        src = inspect.getsource(cls.forward)
        out = c.fixup_fused_lm_head(src)
        out, _ = c.apply_fused_lm_head(out, cls.__name__)

        if "NOT_RETURN_LOGITS" not in out:
            failures.append(
                f"{cls.__name__}: template path did not patch the forward "
                f"(missing NOT_RETURN_LOGITS sentinel). Cannot evaluate."
            )
            continue

        # The dedicated elif from #666 looks like:
        #   elif self.loss_function.__name__.endswith("ForCausalLMLoss") \
        #        and labels is not None:
        #       # UNSLOTH_RETURN_LOGITS=1 path. Prepended `logits = self.lm_head(...)`
        #       ...
        #       loss = self.loss_function(logits, labels.to(...), vocab_size=...)
        dedicated = re.search(
            r"elif self\.loss_function\.__name__\.endswith\(.ForCausalLMLoss.\) "
            r"and labels is not None:\s*\n[^\n]*UNSLOTH_RETURN_LOGITS",
            out,
        )

        if not dedicated:
            failures.append(
                f"{cls.__name__}: dedicated UNSLOTH_RETURN_LOGITS=1 elif branch is "
                f"MISSING from compiler.py template output. PR #684 reintroduces the "
                f"double-matmul bug #666 fixed (commit f45c31e5)."
            )

    if failures:
        msg = "\n".join(["compiler.py template regression detected:"] + failures)
        raise AssertionError(msg)
