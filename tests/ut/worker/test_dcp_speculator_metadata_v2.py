# SPDX-License-Identifier: Apache-2.0
"""Exercise the DSpark/MTP entry points, not just the length helper."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.config import AttentionConfig
from vllm.v1.worker.gpu.spec_decode import speculator as upstream_speculator
from vllm.v1.worker.gpu.spec_decode.autoregressive.speculator import AutoRegressiveSpeculator

from vllm_ascend.attention.context_parallel import sfa_cp
from vllm_ascend.attention.context_parallel.sfa_cp import AscendSFADCPMetadata, AscendSFADCPMetadataBuilder
from vllm_ascend.worker.dcp_utils import DCPManager
from vllm_ascend.worker.v2 import attn_utils
from vllm_ascend.worker.v2.spec_decode.dspark.speculator import AscendDSparkSpeculator
from vllm_ascend.worker.v2.spec_decode.mtp.speculator import AscendMTPSpeculator


def _local(lengths, rank=1, size=8, interleave=4):
    return torch.tensor(
        [sum((i // interleave) % size == rank for i in range(n)) for n in lengths],
        dtype=torch.int32,
    )


def _dcp_local_cpu(common):
    """CPU DCP-local lengths under either the Ascend or upstream field name.

    Ascend's ``AscendCommonAttentionMetadata`` keeps ``dcp_local_seq_lens_cpu``;
    vLLM main (#56157) renamed the upstream field to
    ``dcp_local_seq_lens_cpu_upper_bound``. The direct SFA pass-through receives
    the upstream metadata, so both spellings must be tolerated.
    """
    if hasattr(common, "dcp_local_seq_lens_cpu"):
        return common.dcp_local_seq_lens_cpu
    return getattr(common, "dcp_local_seq_lens_cpu_upper_bound", None)


class FakeDecodeMetadata:
    """Minimal stand-in mirroring ``AscendMLADCPDecodeMetadata``."""

    def __init__(self, actual_seq_lengths_q):
        self.actual_seq_lengths_q = actual_seq_lengths_q
        self.seq_lens_list = None

    def update_dcp_seq_lens_cpu(
        self,
        seq_lens_cpu,
        dcp_local_seq_lens_cpu,
        query_lens_cpu,
        *,
        dcp_size,
        dcp_rank,
        cp_kv_cache_interleave_size,
    ):
        self.cp_seq_len = dcp_local_seq_lens_cpu.tolist()


def _speculator(monkeypatch, kind, architecture, width, padded, step, use_dcp=True):
    config = SimpleNamespace(
        attention_config=AttentionConfig(),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=8 if use_dcp else 1,
            cp_kv_cache_interleave_size=4,
        ),
    )
    manager = object.__new__(DCPManager)
    manager.dcp_world_size = config.parallel_config.decode_context_parallel_size
    manager.dcp_world_rank = 1 if use_dcp else 0
    manager.vllm_config = config
    cls = AscendDSparkSpeculator if kind == "dspark" else AscendMTPSpeculator
    spec = object.__new__(cls)
    spec.attn_architecture = architecture
    spec.use_dcp = use_dcp
    spec.dcp_manager = manager
    spec.max_model_len = 128
    spec.draft_max_seq_len = 128
    spec.num_query_per_req = width
    spec.vllm_config = spec.draft_vllm_config = config
    spec.requires_non_causal = kind == "dspark"
    spec._group_causal = {0: False} if kind == "dspark" else True
    spec.draft_attn_layer_names = {"draft.layer"}
    target = torch.tensor([31, 127, 99, 99], dtype=torch.int32)
    spec.target_input_buffers = SimpleNamespace(seq_lens_cpu=target)
    spec._get_seq_lens_cpu = lambda n: target[:n]
    spec.input_batch = SimpleNamespace(
        num_reqs=2,
        is_prefilling_np=np.array([True, False]),
        seq_lens_cpu_upper_bound=target[:2],
    )
    # Deliberately different from the CPU view: the device path must not be
    # overwritten while rejection correction on CPU remains deferred.
    device_lengths = torch.tensor([30 + step, 126, 0, 0], dtype=torch.int32)
    spec.input_buffers = SimpleNamespace(
        seq_lens=device_lengths,
        positions=torch.arange(padded * width),
        query_start_loc=torch.tensor([0, width, 2 * width, 2 * width, 2 * width], dtype=torch.int32),
        dcp_local_seq_lens=torch.zeros(4, dtype=torch.int32),
    )
    spec.arange_np = np.arange(5, dtype=np.int32)
    # Upstream #56107 reads this in _build_attn_metadata; Ascend forces its own
    # is_prefilling through the draft metadata factory, so the contents are unused.
    spec.draft_is_prefilling = torch.zeros(4, dtype=torch.bool)
    spec.block_tables = SimpleNamespace(
        cp_size=8 if use_dcp else 1,
        cp_rank=1 if use_dcp else 0,
        cp_interleave=4,
        input_block_tables=[torch.zeros((4, 4), dtype=torch.int32)],
        slot_mappings=torch.zeros((1, padded * width), dtype=torch.int64),
    )
    spec.kv_cache_config = SimpleNamespace(kv_cache_groups=[None])

    class RecordingBuilder:
        def build(self, common_prefix_len, common_attn_metadata):
            common = common_attn_metadata
            decode = FakeDecodeMetadata(common.query_start_loc_cpu[1:].tolist())
            # Keep the layer metadata's CPU view independent of the batch-level
            # one, matching the real builder (see the "without_aliasing" case in
            # test_mla_dcp_metadata_v2.py). vLLM main also dropped the deprecated
            # CommonAttentionMetadata.seq_lens_cpu property, so fall back to the
            # device lengths for the upstream pass-through.
            seq_lens_cpu = getattr(common, "seq_lens_cpu", None)
            if seq_lens_cpu is None:
                seq_lens_cpu = common.seq_lens
            return SimpleNamespace(common=common, decode=decode, seq_lens_cpu=seq_lens_cpu.clone())

    spec.attn_groups = [
        [SimpleNamespace(get_metadata_builder=lambda _: RecordingBuilder(), layer_names=["draft.layer"])]
    ]

    def prepare_device(out, seq_lens, num_reqs, size, rank, interleave):
        out.zero_()
        out[:num_reqs].copy_(_local(seq_lens[:num_reqs].tolist(), rank, size, interleave))
        return out

    # vLLM main returns the prepared buffer; 0.29 calls the in-place helper.
    prepare_name = (
        "maybe_prepare_dcp_local_seq_lens"
        if hasattr(upstream_speculator, "maybe_prepare_dcp_local_seq_lens")
        else "prepare_dcp_local_seq_lens"
    )
    monkeypatch.setattr(upstream_speculator, prepare_name, prepare_device)
    monkeypatch.setattr(attn_utils, "get_dcp_group", lambda: SimpleNamespace(rank_in_group=1))
    return spec, target.clone(), device_lengths.clone()


@pytest.mark.parametrize(
    "architecture,padded,width,full_rebuild",
    [("MLA", 2, 3, False), ("MLA", 4, 4, True), ("SFA", 4, 3, False), ("SFA", 2, 4, True)],
)
def test_dspark_common_dcp_preparation(monkeypatch, architecture, padded, width, full_rebuild):
    spec, original_target, device_lengths = _speculator(monkeypatch, "dspark", architecture, width, padded, width)
    if full_rebuild:
        result = spec.build_draft_attn_metadatas(padded, spec.input_batch.seq_lens_cpu_upper_bound)[0]
    else:
        result = spec._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=padded,
            num_tokens_padded=padded * width,
            seq_lens_cpu_upper_bound=spec.input_batch.seq_lens_cpu_upper_bound,
            step=width,
            causal=spec._group_causal,
        )
    common = result["draft.layer"].common
    if architecture == "SFA" and not full_rebuild:
        # The direct SFA entry delegates to upstream without Ascend CPU
        # preparation, so the CPU view is the device lengths and the upstream
        # field names/values apply (upstream now forwards is_prefilling).
        torch.testing.assert_close(common.seq_lens, device_lengths[:padded])
        assert _dcp_local_cpu(common) is None
        assert common.is_prefilling.tolist() == [False, False]
    else:
        expected = [31 + width, 128] + [0] * (padded - 2)
        assert common.seq_lens_cpu.tolist() == expected
        assert torch.equal(common.dcp_local_seq_lens_cpu, _local(expected))
        assert common.is_prefilling.tolist() == [False] * padded
    assert torch.equal(common.dcp_local_seq_lens, _local(device_lengths[:padded].tolist()))
    assert common.causal is False
    assert torch.equal(spec.target_input_buffers.seq_lens_cpu, original_target)
    if architecture == "MLA":
        assert result["draft.layer"].decode.actual_seq_lengths_q == [(i + 1) * width for i in range(padded)]


@pytest.mark.parametrize("architecture,padded,step", [("MLA", 4, 0), ("MLA", 2, 3), ("SFA", 2, 0), ("SFA", 4, 1)])
def test_mtp_common_dcp_preparation(monkeypatch, architecture, padded, step):
    spec, original_target, device_lengths = _speculator(monkeypatch, "mtp", architecture, 1, padded, step)
    supplied_device_local = _local(device_lengths.tolist())
    with attn_utils.build_attn_metadata_wrapper():
        result = spec._build_draft_attn_metadata(
            num_reqs=2,
            num_reqs_padded=padded,
            num_tokens_padded=padded,
            seq_lens_cpu_upper_bound=spec.input_batch.seq_lens_cpu_upper_bound,
            step=step,
            dcp_local_seq_lens=supplied_device_local,
        )
    common = result["draft.layer"].common
    expected = [31 + step, min(127 + step, 128)] + [0] * (padded - 2)
    assert common.seq_lens_cpu.tolist() == expected
    assert torch.equal(common.dcp_local_seq_lens_cpu, _local(expected))
    assert common.dcp_local_seq_lens.data_ptr() == supplied_device_local.data_ptr()
    flags = [True, False] + [False] * (padded - 2) if step == 0 else [False] * padded
    assert common.is_prefilling.tolist() == flags
    assert torch.equal(spec.target_input_buffers.seq_lens_cpu, original_target)


@pytest.mark.parametrize("kind,full_rebuild", [("mtp", False), ("dspark", False), ("dspark", True)])
@pytest.mark.parametrize("architecture", ["MLA", "SFA"])
def test_non_dcp_preserves_existing_length_fallback(monkeypatch, kind, architecture, full_rebuild):
    width = 3 if kind == "dspark" else 1
    spec, original_target, device_lengths = _speculator(monkeypatch, kind, architecture, width, 2, width, use_dcp=False)

    def unexpected(*args, **kwargs):
        raise AssertionError("Non-DCP must not enter DCP CPU preparation")

    spec.dcp_manager.prepare_draft_dcp_metadata_inputs = unexpected
    if full_rebuild:
        # Called directly by the graph manager, without an outer wrapper.
        result = spec.build_draft_attn_metadatas(2, spec.input_batch.seq_lens_cpu_upper_bound)[0]
    else:
        with attn_utils.build_attn_metadata_wrapper() if kind == "mtp" else nullcontext():
            result = spec._build_draft_attn_metadata(
                num_reqs=2,
                num_reqs_padded=2,
                num_tokens_padded=2 * width,
                seq_lens_cpu_upper_bound=spec.input_batch.seq_lens_cpu_upper_bound,
                step=width,
            )
    common = result["draft.layer"].common
    assert _dcp_local_cpu(common) is None
    assert common.dcp_local_seq_lens is None
    if kind == "dspark" and architecture == "SFA" and not full_rebuild:
        # Upstream derives its CPU view from the device lengths; the deprecated
        # seq_lens_cpu property was removed in vLLM main.
        torch.testing.assert_close(common.seq_lens, device_lengths[:2])
    else:
        assert common.seq_lens_cpu.tolist() == [128, 128]
    assert torch.equal(common.seq_lens, device_lengths[:2])
    assert torch.equal(spec.target_input_buffers.seq_lens_cpu, original_target)


def test_sfa_consumer_uses_device_local_lengths_and_ignores_cpu(monkeypatch):
    builder = object.__new__(AscendSFADCPMetadataBuilder)
    builder.device = torch.device("cpu")  # Device stand-in; no NPU kernels run.
    builder.dcp_local_seq_lens_buf = torch.empty(4, dtype=torch.int32)
    builder.decode_threshold = 4
    builder.vllm_config = None
    monkeypatch.setattr(AscendSFADCPMetadataBuilder, "dcp_enabled", False, raising=False)
    monkeypatch.setattr(sfa_cp, "split_decodes_and_prefills", lambda *a, **kw: (4, 0, 4, 0))
    block_table = torch.zeros((4, 2), dtype=torch.int32)
    slots = torch.arange(4)
    builder._get_dcp_local_block_table = lambda *_: block_table
    builder._build_block_table_replicated_view = lambda *_: block_table
    builder._build_slot_mapping_replicated_view = lambda *_: slots
    builder._update_parallel_slot_mapping = lambda *_: None
    common = SimpleNamespace(
        slot_mapping=slots,
        block_table_tensor=block_table,
        num_reqs=4,
        num_input_tokens=4,
        seq_lens=torch.tensor([36, 126, 0, 0], dtype=torch.int32),
        dcp_local_seq_lens=torch.tensor([4, 16, 0, 0], dtype=torch.int32),
        dcp_local_seq_lens_cpu=torch.full((4,), 999, dtype=torch.int32),
    )
    metadata = object.__new__(AscendSFADCPMetadata)
    metadata.seq_lens = common.seq_lens
    result = builder._build_with_metadata_view(common, lambda: metadata)
    assert result.dcp_context.seq_lens.tolist() == [4, 16, 0, 0]
    assert common.dcp_local_seq_lens_cpu.tolist() == [999] * 4


def test_draft_decode_hooks_forward_parallel_config(monkeypatch):
    """DCP draft decode calls these hooks directly and needs parallel_config."""
    spec, _, _ = _speculator(monkeypatch, "mtp", "SFA", 1, 2, 1)
    seen: list[object] = []

    @contextmanager
    def factory(*_args, **kwargs):
        seen.append(kwargs["parallel_config"])
        yield

    monkeypatch.setattr(
        "vllm_ascend.worker.v2.spec_decode.autoregressive.speculator.build_draft_attn_metadata_factory",
        factory,
    )
    monkeypatch.setattr(AutoRegressiveSpeculator, "_build_uniform_attn_metadata", lambda *a, **k: None)
    monkeypatch.setattr(AutoRegressiveSpeculator, "_build_attn_metadata", lambda *a, **k: None)
    batch = SimpleNamespace(num_tokens=2, num_reqs=2)
    seq_lens = spec.input_batch.seq_lens_cpu_upper_bound
    dcp_local = spec.input_buffers.dcp_local_seq_lens
    spec._build_uniform_attn_metadata(batch, 2, 1, seq_lens, 1, dcp_local_seq_lens=dcp_local)
    spec._build_attn_metadata(2, batch, np.array([0, 1, 2]), seq_lens, 1, dcp_local_seq_lens=dcp_local)

    assert seen == [spec.draft_vllm_config.parallel_config, spec.draft_vllm_config.parallel_config]
