# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_ascend.patch.worker import patch_uva


def _set_uva_env(monkeypatch, alloc_conf: str | None, triton_ok: bool = True):
    if alloc_conf is None:
        monkeypatch.delenv("PYTORCH_NPU_ALLOC_CONF", raising=False)
    else:
        monkeypatch.setenv("PYTORCH_NPU_ALLOC_CONF", alloc_conf)
    monkeypatch.setattr(patch_uva, "check_triton_ascend_version_valid", lambda: triton_ok)


def test_is_uva_available_true_case_insensitive(monkeypatch):
    # The config is written as 'pinned_mem_register:True' (L162); a user typing
    # lowercase must still enable UVA.
    for conf in (
        "pinned_mem_register:True",
        "pinned_mem_register:true",
        "expandable_segments:False,pinned_mem_register:True",
    ):
        _set_uva_env(monkeypatch, conf)
        assert patch_uva.is_uva_available() is True, conf


def test_is_uva_available_false_when_absent(monkeypatch):
    # Default must be a string, not a dict: the old `in {}` only happened to
    # return False because nothing is a key of an empty dict.
    _set_uva_env(monkeypatch, None)
    assert patch_uva.is_uva_available() is False


def test_is_uva_available_false_when_unset_flag(monkeypatch):
    _set_uva_env(monkeypatch, "expandable_segments:False")
    assert patch_uva.is_uva_available() is False


def test_is_uva_available_false_when_triton_unsupported(monkeypatch):
    _set_uva_env(monkeypatch, "pinned_mem_register:True", triton_ok=False)
    assert patch_uva.is_uva_available() is False
