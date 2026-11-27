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
from typing import Any

import numpy as np
import pytest
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
from vllm.v1.worker.gpu.spec_decode.dflash.speculator import DFlashSpeculator

import vllm_ascend.worker.v2.spec_decode.dflash.speculator as dflash_speculator_module
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.context_parallel.mla_cp import AscendMlaDCPImpl
from vllm_ascend.attention.mla_v1 import AscendMLABackend, AscendMLADecodeMetadata, AscendMLAImpl, AscendMLAMetadata
from vllm_ascend.worker.v2.spec_decode.dflash.speculator import AscendDFlashSpeculator, draft_metadata_request_count


@pytest.mark.parametrize(
    ("impl_cls", "causal", "band", "error"),
    [
        (AscendMLAImpl, True, False, None),
        (AscendMLAImpl, True, True, None),
        (AscendMLAImpl, False, False, "must be causal"),
        (AscendMlaDCPImpl, True, False, "decode context parallelism"),
    ],
)
def test_windowed_mla_draft_layers(monkeypatch, impl_cls, causal, band, error):
    window = impl_cls.__new__(impl_cls)
    window.sliding_window, window.use_flashmla = 1024, band
    layers = {"window": SimpleNamespace(impl=window, get_attn_backend=lambda: AscendMLABackend)}
    full = AscendMLAImpl.__new__(AscendMLAImpl)
    full.sliding_window = None
    layers["full"] = SimpleNamespace(impl=full, get_attn_backend=lambda: AscendMLABackend)

    spec = _spec([0])
    spec.device = torch.device("cpu")
    spec.max_num_tokens = 8
    spec.draft_kv_cache_group_ids = [0]
    spec.draft_attn_layer_names = set(layers)
    spec.model = SimpleNamespace(
        get_draft_kv_cache_layer_names=lambda: list(layers),
        get_draft_attn_causal=lambda: [causal, True],
    )
    states = {name: object() for name in layers}
    spec.attn_groups = [
        [
            SimpleNamespace(
                layer_names=[name], get_metadata_builder=lambda _, state=state: SimpleNamespace(flashmla_state=state)
            )
        ]
        for name, state in states.items()
    ]
    monkeypatch.setattr(dflash_speculator_module, "detect_draft_attn_architecture", lambda _: ("MLA", band, False))
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(layer_names=[*layers, "target"])])
    monkeypatch.setattr(DFlashSpeculator, "set_attn", lambda *args: None)
    monkeypatch.setattr(dflash_speculator_module, "get_layers_from_vllm_config", lambda *args: layers)

    if error:
        with pytest.raises(ValueError, match=error):
            spec.set_attn(None, cache, None, None, None)
    else:
        spec.set_attn(None, cache, None, None, None)
        assert spec._windowed_fia_layers == (() if band else ("window",))
        assert spec._windowed_flashmla_builders == ((states["window"],) if band else ())


def _spec(lengths):
    spec = AscendDFlashSpeculator.__new__(AscendDFlashSpeculator)
    spec.attn_architecture = "MLA"
    spec.use_flashmla = False
    spec.attn_backends = {"draft.0": AscendMLABackend}
    spec.input_buffers = SimpleNamespace(
        seq_lens=torch.tensor(lengths, dtype=torch.int32),
        positions=torch.zeros(64),
    )
    spec.input_batch = SimpleNamespace(num_reqs=2)
    spec.draft_is_prefilling = torch.zeros(8, dtype=torch.bool)
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


def test_mla_draft_eager_and_acl_get_exact_lengths(monkeypatch):
    monkeypatch.setattr(AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    spec = _spec([1034, 1040, 7, 9])
    supplied: list[dict[str, Any]] = []
    _capture_factory(monkeypatch, supplied)
    upper_bound = torch.tensor([1040, 1048], dtype=torch.int32)

    def upstream_build(*args, query_start_loc_np, **kwargs):
        decode = SimpleNamespace(actual_seq_lengths_q=query_start_loc_np[1:].tolist())
        return {"draft.0": SimpleNamespace(decode=decode)}

    monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", upstream_build)
    for expected in ([1034, 1040], [1035, 1047]):
        spec.input_buffers.seq_lens[:2] = torch.tensor(expected)
        with draft_metadata_request_count(spec, 2):
            eager = spec._build_uniform_attn_metadata(
                num_reqs=2,
                batch_desc=BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=16, num_reqs=2),
                num_query_per_req=8,
                seq_lens_cpu_upper_bound=upper_bound,
                step=8,
                causal=True,
            )
            replay = spec.build_draft_attn_metadatas(4, upper_bound)[0]
        assert eager["draft.0"].decode.actual_seq_lengths_q == [8, 16]
        assert replay["draft.0"].decode.actual_seq_lengths_q == [8, 16, 24, 32]
        for inputs, kv_lens in zip(supplied, (expected, [*expected, 0, 0]), strict=True):
            assert inputs["attn_state"] == AscendAttentionState.ChunkedPrefill
            flags = inputs["is_prefilling"]
            assert flags.tolist() == [False] * len(kv_lens)
            assert flags.data_ptr() == spec.draft_is_prefilling.data_ptr()
            lengths = inputs["seq_lens_cpu"]
            assert lengths.device.type == "cpu" and lengths.dtype == torch.int32
            assert lengths.tolist() == kv_lens
        supplied.clear()


@pytest.mark.parametrize("mode", [CUDAGraphMode.NONE, CUDAGraphMode.FULL])
def test_window_at_model_limit_preserves_valid_draft_rows(monkeypatch, mode):
    # Only four queries of request 0 fit. Request 1 has room for all eight.
    num_rows = 3 if mode == CUDAGraphMode.FULL else 2
    spec = _spec([4096, 2048, 0][:num_rows])
    spec._windowed_fia_layers = ("window",)
    spec.attn_backends = dict.fromkeys(("window", "full"), AscendMLABackend)
    monkeypatch.setattr(AscendDFlashSpeculator, "max_model_len", 4096, raising=False)
    spec.draft_max_seq_len = 4096
    spec.num_speculative_steps = 7
    spec.sample_pos = torch.tensor([*range(4093, 4100), *range(2041, 2048)])
    positions = torch.tensor([*range(4092, 4100), *range(2040, 2048)] + [0] * (8 * num_rows - 16)).clamp_max(4095)
    spec.input_buffers.positions = positions
    lengths = spec.input_buffers.seq_lens
    tables = torch.arange(1, num_rows * 32 + 1, dtype=torch.int32).reshape(num_rows, 32)
    decode = AscendMLADecodeMetadata(positions, tables, lengths, 4096, lengths.tolist(), [8, 16])
    metadata = AscendMLAMetadata(
        num_actual_tokens=16,
        slot_mapping=torch.zeros(8 * num_rows, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 8, 16]),
        seq_lens=lengths,
        seq_lens_cpu=lengths,
        block_tables=tables,
        num_decodes=2,
        num_decode_tokens=16,
        num_prefills=0,
        decode=decode,
    )
    monkeypatch.setattr(AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    _capture_factory(monkeypatch, [])
    monkeypatch.setattr(
        DFlashSpeculator, "_build_attn_metadata", lambda *args, **kwargs: dict.fromkeys(spec.attn_backends, metadata)
    )
    sample_pos_copies = 0
    copy_tensor = torch.Tensor.copy_

    def count_copy(dst, src, *args, **kwargs):
        nonlocal sample_pos_copies
        if dst.device.type == "cpu" and src.data_ptr() == spec.sample_pos.data_ptr():
            sample_pos_copies += 1
        return copy_tensor(dst, src, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "copy_", count_copy)
    with draft_metadata_request_count(spec, 2):
        built = spec._build_uniform_attn_metadata(
            num_reqs=2,
            batch_desc=BatchExecutionDescriptor(cg_mode=mode, num_tokens=8 * num_rows, num_reqs=num_rows),
            num_query_per_req=8,
            seq_lens_cpu_upper_bound=lengths,
            step=8,
            causal=True,
        )
        if mode == CUDAGraphMode.FULL:
            replay = spec.build_draft_attn_metadatas(num_rows, lengths)[0]["window"].decode
            assert replay.actual_seq_lengths_q == built["window"].decode.actual_seq_lengths_q
            assert replay.seq_lens_list == built["window"].decode.seq_lens_list
        assert sample_pos_copies == 1

    # Reconstruct the newest visible key for each physical query as FIA does.
    # The first sampled row must see key 4093 (window [3070, 4093]).
    window = built["window"].decode
    ends: list[tuple[int, int] | None] = []
    start = 0
    for end, kv_len, table in zip(window.actual_seq_lengths_q, window.seq_lens_list, window.block_table, strict=True):
        ends.extend((int(table[0]), pos) if kv_len else None for pos in range(kv_len - (end - start), kv_len))
        start = end
    assert ends == (
        [(1, pos) for pos in range(4092, 4096)]
        + [None] * 4
        + [(33, pos) for pos in range(2040, 2048)]
        + [None] * (8 * num_rows - 16)
    )
    assert built["full"].decode is decode
    assert decode.actual_seq_lengths_q == list(range(8, 8 * num_rows + 1, 8))
    assert decode.block_table is tables

    if mode == CUDAGraphMode.FULL:
        # A fresh proposal in the same graph bucket now has six valid queries.
        spec.sample_pos[:7].sub_(2)
        with draft_metadata_request_count(spec, 2):
            fresh = spec.build_draft_attn_metadatas(num_rows, lengths)[0]["window"].decode
        assert fresh.actual_seq_lengths_q[:2] == [6, 8]
        assert sample_pos_copies == 2


@pytest.mark.parametrize("a3_sink", [False, True], ids=["a5", "a3"])
def test_flashmla_window_supplies_unclipped_first_positions(monkeypatch, a3_sink):
    spec = _spec([4096, 4096, 2048, 0])
    spec.use_flashmla = not a3_sink
    spec.use_a3_mla_sink = a3_sink
    bounds = []

    @contextmanager
    def limit_query_positions(positions, limit):
        bounds.append((positions.tolist(), limit))
        yield

    spec._windowed_flashmla_builders = (SimpleNamespace(limit_query_positions=limit_query_positions),)
    spec.num_speculative_steps = 7
    spec.sample_pos = torch.zeros(28, dtype=torch.int64)
    spec.sample_pos[::7] = torch.tensor([4093, 4097, 2041, 0])
    monkeypatch.setattr(AscendDFlashSpeculator, "max_model_len", 4096, raising=False)
    monkeypatch.setattr(AscendDFlashSpeculator, "attn_vllm_config", property(lambda self: self.vllm_config))
    supplied: list[dict[str, Any]] = []
    _capture_factory(monkeypatch, supplied)
    monkeypatch.setattr(DFlashSpeculator, "_build_attn_metadata", lambda *args, **kwargs: {})
    monkeypatch.setattr(dflash_speculator_module, "prepare_draft_fia_lengths", lambda *_a, **_kw: pytest.fail("D2H"))

    for max_len, expected in [(4095, []), (4096, [([4092, 4096, 2040], 4096)]), (4096, [([4090, 4096, 2040], 4096)])]:
        bounds.clear()
        spec.draft_max_seq_len = max_len
        with draft_metadata_request_count(spec, 3):
            spec.build_draft_attn_metadatas(4, spec.input_buffers.seq_lens)
        inputs = supplied[-1]
        assert not inputs["is_prefilling"].any() and inputs["seq_lens_cpu"] is None
        assert bounds == expected
        if bounds:
            spec.sample_pos[0] -= 2
