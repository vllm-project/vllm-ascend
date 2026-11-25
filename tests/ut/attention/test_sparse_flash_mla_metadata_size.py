# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The NoPE sparse-MLA plan buffer follows the installed operator contract.

The core-split plan that ``SparseFlashMla`` writes is a fixed-size int32
buffer whose size belongs to the operator package: CANN 9.2 produced 1024
words, CANN 9.3 produces 2048. vllm-ascend keeps that plan in a persistent
buffer (ACL Graph replay reads the address captured on the first run), so the
buffer has to be sized from the operator that fills it.
"""

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.attention.sfa_v1 as sfa_v1
import vllm_ascend.attention.sparse_flash_mla as sparse_flash_mla
from vllm_ascend.attention.sparse_flash_mla import get_smla_metadata_size
from vllm_ascend.device.hardware_profile import DeviceAdaptorFamily


@pytest.fixture(autouse=True)
def _clear_resolved_size():
    """The resolved size is cached per process; keep the tests independent."""
    sparse_flash_mla.get_smla_metadata_size.cache_clear()
    yield
    sparse_flash_mla.get_smla_metadata_size.cache_clear()


def _fake_package(monkeypatch, value, module_name: str | None = None):
    """Make the operator package declare ``value`` as its plan size."""
    declared = SimpleNamespace(SMLA_METADATA_SIZE=value)

    def fake_import(name):
        if module_name is not None and name != module_name:
            raise ImportError(f"no module {name}")
        return declared

    monkeypatch.setattr(sparse_flash_mla, "import_module", fake_import)


def test_metadata_size_follows_the_installed_operator_package(monkeypatch):
    """CANN 9.3 declares a 2048-word plan; the buffer has to follow it."""
    _fake_package(monkeypatch, 2048)

    assert get_smla_metadata_size() == 2048


def test_metadata_size_uses_the_metadata_operator_module(monkeypatch):
    """The size may live in the metadata operator module instead."""
    _fake_package(
        monkeypatch,
        2048,
        module_name="cann_ops_transformer.ops.attention.sparse_flash_mla.sparse_flash_mla_metadata",
    )

    assert get_smla_metadata_size() == 2048


def test_metadata_size_falls_back_without_the_operator_package(monkeypatch):
    """A toolkit without the declaration keeps the historical plan size."""
    _fake_package(monkeypatch, None)
    monkeypatch.setattr(
        sparse_flash_mla,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(f"no module {name}")),
    )

    assert get_smla_metadata_size() == sparse_flash_mla._SMLA_METADATA_SIZE_FALLBACK


@pytest.mark.parametrize("value", [0, -1, "2048", None, True])
def test_metadata_size_ignores_an_invalid_declaration(monkeypatch, value):
    """Only a positive integer is a usable plan size."""
    _fake_package(monkeypatch, value)

    assert get_smla_metadata_size() == sparse_flash_mla._SMLA_METADATA_SIZE_FALLBACK


def test_metadata_size_is_resolved_once(monkeypatch):
    """The operator package is consulted once per process."""
    calls = []

    def fake_import(name):
        calls.append(name)
        return SimpleNamespace(SMLA_METADATA_SIZE=2048)

    monkeypatch.setattr(sparse_flash_mla, "import_module", fake_import)

    assert get_smla_metadata_size() == 2048
    assert get_smla_metadata_size() == 2048
    assert len(calls) == 1


def _metadata_state(monkeypatch, plan_size: int):
    """A NoPE metadata state whose plan buffer is sized from the resolver."""
    config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=128),
        model_config=SimpleNamespace(
            max_model_len=4096,
            hf_text_config=SimpleNamespace(num_attention_heads=4),
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            prefill_context_parallel_size=1,
        ),
        compilation_config=SimpleNamespace(cudagraph_capture_sizes=[]),
        scheduler_config=SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=32),
    )
    monkeypatch.setattr(
        sfa_v1,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED),
    )
    monkeypatch.setattr(
        sfa_v1,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_sparse_sfa_c8=False, enable_sparse_sfa_turboquant=False),
    )
    monkeypatch.setattr(sfa_v1, "get_smla_metadata_size", lambda: plan_size)
    return sfa_v1.SparseMLAMetadataState(
        SimpleNamespace(block_size=128),
        config,
        torch.device("cpu"),
        kernel_block_size=128,
    )


def test_state_allocates_the_resolved_plan_buffer(monkeypatch):
    """The persistent buffer holds exactly the operator's plan."""
    state = _metadata_state(monkeypatch, 2048)

    assert state.metadata_buffer.numel() == 2048
    assert state.metadata_buffer.dtype == torch.int32


def test_plan_size_mismatch_reports_both_sizes(monkeypatch):
    """A plan that does not match the buffer names both sizes."""
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.tensor([4], dtype=torch.int32),
        max_query_len=1,
        max_seq_len=4,
        smla_topk_length=torch.ones(1, 1, dtype=torch.int32),
    )
    monkeypatch.setattr(sfa_v1, "sparse_flash_mla_metadata", lambda **_: torch.zeros(2048, dtype=torch.int32))
    buffer = torch.empty(1024, dtype=torch.int32)

    with pytest.raises(ValueError, match=r"1024 int32 values, got 2048"):
        sfa_v1.build_smla_metadata(
            metadata,
            buffer,
            2,
            128,
            7,
            metadata.query_start_loc,
            metadata.smla_topk_length,
        )
