# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Rejection-adjusted host lengths for DFlash2 MLA draft metadata.

After a rejection the draft FIA sliding window must right-align to the
accepted context, not the optimistic CPU upper bound. The speculator
mirrors the per-step device seq_lens to a zero-padded host tensor (the
DSpark pattern) and builds MLA draft metadata through
``_build_attn_metadata`` (the upstream hook both eager ``propose()`` and
FULL-graph replay reach) with ``attn_state=ChunkedPrefill``, so both
paths emit the exact integer list and take the sliding-window FIA
branch.
"""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_speculator_module
from vllm_ascend.attention.attention_v1 import AscendAttentionBackend
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator
from vllm_ascend.worker.v2.spec_decode.host_lengths import prepare_draft_host_lengths, proposal_host_lengths


def _spec(lengths, *, mla_draft=True):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.attn_architecture = "MLA" if mla_draft else "GQA"
    spec.use_flashmla = False
    spec.attn_backends = {"draft.0": AscendAttentionBackend}
    spec.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        positions=torch.zeros(64),
    )
    spec.input_batch = SimpleNamespace(num_reqs=2)
    spec.num_query_per_req = 8
    spec.arange_np = np.arange(65, dtype=np.int32)
    spec._group_causal = True
    spec.vllm_config = SimpleNamespace(parallel_config=object())
    return spec


def _capture_factory(monkeypatch, supplied):
    @contextmanager
    def factory(*args, **kwargs):
        supplied.append(kwargs)
        yield

    monkeypatch.setattr(dflash_speculator_module, "build_attn_metadata_factory", factory)
    monkeypatch.setattr(dflash_speculator_module, "build_attn_metadata_wrapper", nullcontext)


class TestRejectionAdjustedHostLengths:
    def test_zero_padded_host_copy(self):
        spec = _spec([1034, 1040, 7, 9])
        host = prepare_draft_host_lengths(spec, 2, 4, 8)
        assert host.dtype == torch.int32
        assert host.device.type == "cpu"
        assert host.tolist() == [1034, 1040, 0, 0]

    def test_host_lengths_refresh_after_rejection(self):
        spec = _spec([1034, 1040, 7, 9])
        first = prepare_draft_host_lengths(spec, 2, 4, 8)
        spec.input_buffers.seq_lens[:2].copy_(torch.tensor([1035, 1047], dtype=torch.int32))
        second = prepare_draft_host_lengths(spec, 2, 4, 8)
        assert first.tolist() == [1034, 1040, 0, 0]
        assert second.tolist() == [1035, 1047, 0, 0]

    def test_proposal_scope_caches_same_shape(self):
        spec = _spec([1034, 1040, 7, 9])
        with proposal_host_lengths(spec, 2):
            first = prepare_draft_host_lengths(spec, 2, 4, 8)
            second = prepare_draft_host_lengths(spec, 2, 4, 8)
            assert first is second
        third = prepare_draft_host_lengths(spec, 2, 4, 8)
        assert third is not first


class TestBuildDraftAttnMetadatasFactory:
    """Both the eager hook and the ACL replay entry must supply the exact
    rejection-adjusted host lengths and pin ChunkedPrefill, following the
    DSpark regression coverage in test_dspark_valid_kv_lengths."""

    @staticmethod
    def _run_entrypoint(monkeypatch, spec, entrypoint):
        supplied = []
        _capture_factory(monkeypatch, supplied)
        decode = SimpleNamespace(
            actual_seq_lengths_q=[8],
            seq_lens_list=[1034, 1040, 0, 0, 0],
            seq_lens=torch.tensor([1034, 1040, 0, 0, 0], dtype=torch.int32),
            seq_lens_cpu=torch.tensor([1034, 1040, 0, 0, 0], dtype=torch.int32),
            block_tables=None,
        )
        metadata = {"draft.0": SimpleNamespace(decode=decode)}
        monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", MagicMock(return_value=metadata))
        upper_bound = torch.tensor([1040], dtype=torch.int32)
        if entrypoint == "ACL_update":
            result = spec.build_draft_attn_metadatas(4, upper_bound)[0]
        else:
            result = spec._build_uniform_attn_metadata(
                num_reqs=2,
                batch_desc=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=32, num_reqs=4),
                num_query_per_req=8,
                seq_lens_cpu_upper_bound=upper_bound,
                step=8,
                causal=True,
            )
        return result, metadata, supplied, decode

    def test_mla_draft_eager_and_acl_get_exact_lengths(self, monkeypatch):
        monkeypatch.setattr(
            AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config)
        )
        for entrypoint in ("eager", "ACL_update"):
            spec = _spec([1034, 1040, 7, 9])
            result, metadata, supplied, decode = self._run_entrypoint(monkeypatch, spec, entrypoint)
            assert result is metadata
            assert len(supplied) == 1
            kwargs = supplied[0]
            assert kwargs["attn_state"] == AscendAttentionState.ChunkedPrefill
            assert kwargs["is_prefilling"].dtype == torch.bool
            assert kwargs["is_prefilling"].tolist() == [False] * 4
            assert kwargs["seq_lens_cpu"].tolist() == [1034, 1040, 0, 0]

    def test_full_graph_update_pads_query_lengths(self, monkeypatch):
        monkeypatch.setattr(
            AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config)
        )
        spec = _spec([1034, 1040, 7, 9])
        _, _, _, decode = self._run_entrypoint(monkeypatch, spec, "eager")
        assert decode.actual_seq_lengths_q == [8, 16, 24, 32]
        assert decode.seq_lens_list == [1034, 1040, 0, 0]
        assert decode.seq_lens_cpu.tolist() == [1034, 1040, 0, 0]

    def test_non_mla_draft_keeps_upstream_path(self, monkeypatch):
        def forbidden_factory(*args, **kwargs):
            raise AssertionError("build_attn_metadata_factory must not be used for non-MLA drafts")

        monkeypatch.setattr(dflash_speculator_module, "build_attn_metadata_factory", forbidden_factory)
        spec = _spec([1034, 1040, 7, 9], mla_draft=False)
        upstream = MagicMock(return_value={"draft.0": SimpleNamespace()})
        monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", upstream)
        spec._build_uniform_attn_metadata(
            num_reqs=2,
            batch_desc=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.FULL, num_tokens=16, num_reqs=2),
            num_query_per_req=8,
            seq_lens_cpu_upper_bound=torch.tensor([1040], dtype=torch.int32),
            step=8,
            causal=True,
        )
        assert upstream.call_count == 1
