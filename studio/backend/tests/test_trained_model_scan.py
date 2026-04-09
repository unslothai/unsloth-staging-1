# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Tests for Studio trained-model discovery used by Chat."""

import json
import logging
from pathlib import Path
import sys
import types as _types
import importlib


_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = logging.getLogger
sys.modules.setdefault("loggers", _loggers_stub)

from unittest.mock import patch

from utils.models.model_config import (
    ModelConfig,
    get_base_model_from_checkpoint,
    get_base_model_from_lora,
    scan_trained_loras,
    scan_trained_models,
)


def test_scan_trained_models_includes_lora_and_full_finetune_outputs(tmp_path: Path):
    lora_dir = tmp_path / "unsloth_SmolLM-135M_1775412608"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (lora_dir / "adapter_model.safetensors").write_bytes(b"")

    full_dir = tmp_path / "unsloth_SmolLM-135M_full_1775412609"
    full_dir.mkdir()
    (full_dir / "config.json").write_text(
        json.dumps({"_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (full_dir / "model.safetensors").write_bytes(b"")

    found = {
        name: (path, model_type)
        for name, path, model_type in scan_trained_models(str(tmp_path))
    }

    assert found[lora_dir.name] == (str(lora_dir), "lora")
    assert found[full_dir.name] == (str(full_dir), "merged")


def test_get_base_model_from_checkpoint_falls_back_to_full_finetune_config(
    tmp_path: Path,
):
    (tmp_path / "config.json").write_text(
        json.dumps({"_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert get_base_model_from_checkpoint(str(tmp_path)) == "HuggingFaceTB/SmolLM-135M"


def test_get_base_model_from_lora_rejects_full_finetune_dirs(tmp_path: Path):
    (tmp_path / "config.json").write_text(
        json.dumps({"_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (tmp_path / "model.safetensors").write_bytes(b"")

    assert get_base_model_from_lora(str(tmp_path)) is None


@patch("utils.models.model_config.is_audio_input_type", return_value = False)
@patch("utils.models.model_config.detect_audio_type", return_value = None)
@patch("utils.models.model_config.is_vision_model", return_value = False)
def test_model_config_full_finetune_local_path_is_not_lora(
    _mock_vision,
    _mock_audio_type,
    _mock_audio_input,
    tmp_path: Path,
):
    (tmp_path / "config.json").write_text(
        json.dumps({"_name_or_path": "unsloth/Qwen3-4B"})
    )
    (tmp_path / "model.safetensors").write_bytes(b"")

    config = ModelConfig.from_identifier(str(tmp_path))

    assert config is not None
    assert config.is_lora is False
    assert config.base_model is None


def test_scan_trained_loras_backward_compatible_2_tuples(tmp_path: Path):
    """scan_trained_loras must keep its legacy 2-tuple contract so existing
    callers that do ``for name, path in scan_trained_loras(...)`` keep
    working, and it must only surface LoRA adapters (not merged runs)."""
    lora_dir = tmp_path / "unsloth_SmolLM-135M_1775412608"
    lora_dir.mkdir()
    (lora_dir / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (lora_dir / "adapter_model.safetensors").write_bytes(b"")

    merged_dir = tmp_path / "unsloth_SmolLM-135M_full_1775412609"
    merged_dir.mkdir()
    (merged_dir / "config.json").write_text(
        json.dumps({"_name_or_path": "HuggingFaceTB/SmolLM-135M"})
    )
    (merged_dir / "model.safetensors").write_bytes(b"")

    legacy = scan_trained_loras(str(tmp_path))

    # Every entry must be a 2-tuple (name, path), not a 3-tuple.
    assert all(len(t) == 2 for t in legacy)
    names = {name for name, _ in legacy}
    assert lora_dir.name in names
    # The merged full finetune must NOT be surfaced through the
    # LoRA-only backward-compat wrapper.
    assert merged_dir.name not in names

    # And the 2-tuple unpacking pattern used by existing callers must
    # succeed without a ValueError.
    for display_name, adapter_path in scan_trained_loras(str(tmp_path)):
        assert display_name == lora_dir.name
        assert adapter_path == str(lora_dir)


def test_scan_trained_loras_is_not_the_same_object_as_scan_trained_models():
    """Regression guard: the backward-compat wrapper must be a distinct
    callable so its return-type contract cannot drift back to 3-tuples."""
    utils_models = importlib.import_module("utils.models")
    core_module = importlib.import_module("core")

    assert utils_models.scan_trained_loras is not utils_models.scan_trained_models
    assert core_module.scan_trained_loras is not core_module.scan_trained_models


def test_has_model_weight_files_skips_mmproj_safetensors(tmp_path: Path):
    """A directory containing only mmproj vision projection weights must
    not be classified as a full finetune / merged model."""
    from utils.models.model_config import _has_model_weight_files

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (tmp_path / "mmproj-BF16.safetensors").write_bytes(b"")

    assert _has_model_weight_files(tmp_path) is False


def test_detect_training_output_type_handles_sharded_lora(tmp_path: Path):
    """Sharded adapters (adapter_model-00001-of-00002.safetensors) must be
    classified as LoRA rather than merged."""
    from utils.models.model_config import _detect_training_output_type

    (tmp_path / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Qwen3-4B"})
    )
    (tmp_path / "adapter_model-00001-of-00002.safetensors").write_bytes(b"")
    (tmp_path / "adapter_model-00002-of-00002.safetensors").write_bytes(b"")
    # Also drop a config.json that would otherwise push it into "merged".
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))

    assert _detect_training_output_type(tmp_path) == "lora"


def test_get_base_model_from_checkpoint_ignores_training_args_bin_pickle(
    tmp_path: Path,
):
    """training_args.bin is a pickle; deserializing untrusted pickles is
    RCE. get_base_model_from_checkpoint must never call torch.load on it,
    even when adapter_config.json is missing."""
    # Bogus payload so any accidental torch.load would raise.
    (tmp_path / "training_args.bin").write_bytes(b"INVALID_PICKLE_PAYLOAD")

    with patch("torch.load") as torch_load:
        # Should not crash and should not touch torch.load.
        _ = get_base_model_from_checkpoint(str(tmp_path))
        assert not torch_load.called


def test_get_base_model_from_checkpoint_dir_name_fallback_rejects_empty(
    tmp_path: Path,
):
    """For directories named ``unsloth_<timestamp>`` or ``unsloth__<ts>``
    (no model segment) the directory-name heuristic must return None
    rather than fabricating ``"unsloth/"``."""
    bad = tmp_path / "unsloth_1775545843"
    bad.mkdir()
    (bad / "config.json").write_text(json.dumps({"model_type": "llama"}))

    assert get_base_model_from_checkpoint(str(bad)) is None

    empty = tmp_path / "unsloth__1775545843"
    empty.mkdir()
    (empty / "config.json").write_text(json.dumps({"model_type": "llama"}))

    assert get_base_model_from_checkpoint(str(empty)) is None


def test_scan_trained_models_survives_entry_level_errors(tmp_path: Path):
    """A permission error on a single subdirectory must not wipe out the
    rest of the scan results."""
    good = tmp_path / "unsloth_Qwen3-4B_1775000001"
    good.mkdir()
    (good / "adapter_config.json").write_text(
        json.dumps({"base_model_name_or_path": "unsloth/Qwen3-4B"})
    )
    (good / "adapter_model.safetensors").write_bytes(b"")

    bad = tmp_path / "unsloth_broken_1775000002"
    bad.mkdir()

    orig_is_dir = Path.is_dir

    def flaky_is_dir(self):
        if self == bad:
            raise PermissionError("denied")
        return orig_is_dir(self)

    with patch.object(Path, "is_dir", flaky_is_dir):
        results = scan_trained_models(str(tmp_path))

    names = {name for name, _, _ in results}
    assert good.name in names
