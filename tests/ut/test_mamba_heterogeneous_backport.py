import inspect
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import vllm_ascend.patch.worker.patch_mamba_utils  # noqa: F401
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
    is_conv_state_dim_first,
)
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker import mamba_utils
from vllm_ascend.patch.platform.patch_kv_cache_coordinator import (
    AscendHybridKVCacheCoordinator,
    _num_blocks_for_reconciled_hit,
)
from vllm_ascend.patch.platform.patch_kv_delivery_preemption import (
    KVDeliveryScheduler,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


def _spec(
    mamba_type,
    shapes,
    *,
    dtypes=None,
    block_size=768,
    mode="align",
    spec_blocks=3,
):
    return MambaSpec(
        block_size=block_size,
        shapes=shapes,
        dtypes=(tuple(torch.bfloat16 for _ in shapes) if dtypes is None else tuple(dtypes)),
        mamba_type=mamba_type,
        mamba_cache_mode=mode,
        num_speculative_blocks=spec_blocks,
    )


def _config(groups):
    return KVCacheConfig(num_blocks=8, kv_cache_tensors=[], kv_cache_groups=groups)


def _buffer(n, dtype):
    np_dtype = np.int64 if dtype == torch.int64 else np.int32
    return SimpleNamespace(np=np.zeros(n, dtype=np_dtype), gpu=None)


GDN = MambaAttentionBackendEnum.GDN_ATTN
PLE = MambaAttentionBackendEnum.SHORT_CONV
COPY_FUNCS = {
    GDN: (get_conv_copy_spec, get_temporal_copy_spec),
    PLE: (get_conv_copy_spec,),
}


def test_composite_qsa_group_keeps_three_768_token_physical_blocks():
    blocks = [object(), object(), object()]
    assert _num_blocks_for_reconciled_hit(blocks, 2304, 2304) == 3
    assert _num_blocks_for_reconciled_hit(blocks, 2304, 1536) == 2
    assert _num_blocks_for_reconciled_hit(blocks, 2304, 768) == 1
    assert _num_blocks_for_reconciled_hit(blocks, 2304, 2303) == 2
    assert _num_blocks_for_reconciled_hit(blocks, 2304, 767) == 0


def test_full_attention_reconcile_keeps_physical_blocks_for_mla_representative():
    # Qwen3.8's QSA/compressed group is represented at lookup time by an
    # MLAAttentionSpec-like object with a 4x logical compression ratio. It does
    # not retain the composite ``kv_cache_specs`` attribute, while ``blocks``
    # still contains scheduler physical block IDs at 768-token granularity.
    representative = SimpleNamespace(block_size=768 * 4, compress_ratio=4)
    assert not hasattr(representative, "kv_cache_specs")
    assert representative.compress_ratio == 4
    physical_blocks = [object(), object(), object()]
    assert _num_blocks_for_reconciled_hit(physical_blocks, 2304, 2304) == 3
    assert _num_blocks_for_reconciled_hit(physical_blocks, 2304, 1536) == 2

    # Lock the production call site as well as the helper arithmetic: do not
    # allow a future branch to classify the representative by kv_cache_specs or
    # reinterpret its physical IDs with the 3072-token logical block size.
    source = inspect.getsource(
        AscendHybridKVCacheCoordinator.find_longest_cache_hit
    )
    reconcile = source.split(
        "# Truncate every full-attention group to the reconciled hit length.", 1
    )[1].split("cache_hit_blocks =", 1)[0]
    compact_reconcile = "".join(reconcile.split())
    assert "_num_blocks_for_reconciled_hit" in compact_reconcile
    assert 'getattr(spec,"kv_cache_specs",None)' in compact_reconcile
    assert 'hasattr(spec,"compress_ratio")' in compact_reconcile
    assert "_get_effective_block_size" in reconcile
    assert "cdiv" in reconcile


def test_composite_constituents_share_physical_table_and_restored_tokens():
    physical_blocks = [object(), object(), object()]
    num_blocks = _num_blocks_for_reconciled_hit(physical_blocks, 2304, 1536)
    qsa_block_table = physical_blocks[:num_blocks]
    compressed_key_block_table = physical_blocks[:num_blocks]

    assert len(qsa_block_table) == len(compressed_key_block_table) == 2
    assert qsa_block_table == compressed_key_block_table
    scheduler_num_computed_tokens = 1536
    actual_restored_tokens = len(qsa_block_table) * 768
    assert actual_restored_tokens == scheduler_num_computed_tokens


def test_reconciled_hit_rejects_non_integral_lookup_span():
    with pytest.raises(AssertionError, match="non-integral token span"):
        _num_blocks_for_reconciled_hit([object(), object()], 1537, 768)


def test_homogeneous_single_spec_compatibility():
    spec = _spec(GDN, ((6, 1280), (6, 128, 128)))
    config = _config([KVCacheGroupSpec(["gdn.0", "gdn.1"], spec)])
    groups = mamba_utils.get_mamba_groups(config)
    assert groups == {spec: [0]}
    bufs = mamba_utils.MambaCopyBuffers.create(2, config, {GDN: COPY_FUNCS[GDN]}, _buffer)
    assert bufs.entries_per_req == 4
    assert bufs.sizes.np.shape == (8,)


def test_qwen_gdn_ple_uses_73_copy_entries_not_74():
    gdn_spec = _spec(GDN, ((6, 1280), (6, 128, 128)))
    ple_spec = _spec(PLE, ((12, 10240),))
    groups = [KVCacheGroupSpec([f"gdn.{i}"], gdn_spec) for i in range(36)]
    groups.append(KVCacheGroupSpec(["ple.1"], ple_spec))
    config = _config(groups)
    mapping = mamba_utils.get_mamba_groups(config)
    assert mapping[gdn_spec] == list(range(36))
    assert mapping[ple_spec] == [36]
    bufs = mamba_utils.MambaCopyBuffers.create(2, config, COPY_FUNCS, _buffer)
    assert bufs.entries_per_req == 36 * 2 + 1 == 73
    assert bufs.sizes.np.shape == (146,)
    ctx = mamba_utils.MambaSpecDecodeGPUContext.create(2, config, COPY_FUNCS, torch.device("cpu"), _buffer)
    assert ctx.num_states == 73
    assert ctx.state_base_addrs.shape == (73,)


def test_model_runner_selects_copy_funcs_by_mamba_type():
    gdn_spec = _spec(GDN, ((6, 1280), (6, 128, 128)))
    ple_spec = _spec(PLE, ((12, 10240),))
    config = _config(
        [
            KVCacheGroupSpec(["gdn.0"], gdn_spec),
            KVCacheGroupSpec(["ple.1"], ple_spec),
        ]
    )

    class Model:
        def get_mamba_state_copy_funcs(self, mamba_types):
            assert mamba_types == {GDN, PLE}
            return {mamba_type: COPY_FUNCS[mamba_type] for mamba_type in mamba_types}

        def get_mamba_state_copy_func(self):
            raise AssertionError("heterogeneous model must not use legacy funcs")

    runner = SimpleNamespace(kv_cache_config=config, model=Model())
    copy_funcs = NPUModelRunner._get_mamba_state_copy_funcs(runner)
    assert copy_funcs == COPY_FUNCS
    assert runner._mamba_state_copy_funcs_by_type is copy_funcs


def test_distinct_specs_inside_one_uniform_group_are_mapped_per_layer():
    gdn_spec = _spec(GDN, ((6, 1280), (6, 128, 128)))
    ple_spec = _spec(PLE, ((12, 10240),))
    grouped = UniformTypeKVCacheSpecs(
        block_size=768,
        kv_cache_specs={"gdn.0": gdn_spec, "ple.1": ple_spec},
    )
    config = _config([KVCacheGroupSpec(["gdn.0", "ple.1"], grouped)])
    mapping = mamba_utils.get_mamba_groups(config)
    assert mapping == {gdn_spec: [0], ple_spec: [0]}
    assert mamba_utils.get_mamba_group_ids(mapping) == [0]
    bufs = mamba_utils.MambaCopyBuffers.create(1, config, COPY_FUNCS, _buffer)
    assert bufs.entries_per_req == 3


def test_missing_copy_func_is_rejected():
    ple_spec = _spec(PLE, ((12, 10240),))
    config = _config([KVCacheGroupSpec(["ple.1"], ple_spec)])
    with pytest.raises(AssertionError, match="missing state copy funcs"):
        mamba_utils.MambaCopyBuffers.create(1, config, {GDN: COPY_FUNCS[GDN]}, _buffer)


def test_too_many_copy_funcs_is_rejected():
    ple_spec = _spec(PLE, ((12, 10240),))
    config = _config([KVCacheGroupSpec(["ple.1"], ple_spec)])
    with pytest.raises(AssertionError, match="declares 1 states"):
        mamba_utils.MambaCopyBuffers.create(
            1,
            config,
            {PLE: (get_conv_copy_spec, get_temporal_copy_spec)},
            _buffer,
        )


@pytest.mark.parametrize(
    "different",
    [
        {"block_size": 384},
        {"spec_blocks": 1},
        {"mode": "all"},
    ],
)
def test_scheduling_parameter_mismatch_is_rejected(different):
    gdn_spec = _spec(GDN, ((6, 1280), (6, 128, 128)))
    ple_spec = _spec(PLE, ((12, 10240),), **different)
    config = _config(
        [
            KVCacheGroupSpec(["gdn.0"], gdn_spec),
            KVCacheGroupSpec(["ple.1"], ple_spec),
        ]
    )
    with pytest.raises(AssertionError, match="cache scheduling parameters"):
        mamba_utils.MambaCopyBuffers.create(1, config, COPY_FUNCS, _buffer)


def test_metadata_detects_missing_runtime_state():
    gdn_spec = _spec(GDN, ((6, 1280), (6, 128, 128)))
    config = _config([KVCacheGroupSpec(["gdn.0"], gdn_spec)])
    ctx = mamba_utils.MambaSpecDecodeGPUContext.create(1, config, {GDN: COPY_FUNCS[GDN]}, torch.device("cpu"), _buffer)
    forward_context = {"gdn.0": SimpleNamespace(kv_cache=[torch.zeros((2, 6, 1280))])}
    with pytest.raises(ValueError, match="Expected at least 2"):
        ctx.initialize_from_forward_context(
            config,
            forward_context,
            {GDN: COPY_FUNCS[GDN]},
            [torch.zeros((1, 4), dtype=torch.int32)],
        )


def test_metadata_preserves_gdn_bf16_fp32_then_ple_bf16_order():
    gdn_spec = _spec(
        GDN,
        ((2, 4), (2, 3, 4)),
        dtypes=(torch.bfloat16, torch.float32),
    )
    ple_spec = _spec(PLE, ((3, 5),), dtypes=(torch.bfloat16,))
    grouped = UniformTypeKVCacheSpecs(
        block_size=768,
        kv_cache_specs={"gdn.0": gdn_spec, "ple.1": ple_spec},
    )
    config = _config([KVCacheGroupSpec(["gdn.0", "ple.1"], grouped)])
    ctx = mamba_utils.MambaSpecDecodeGPUContext.create(1, config, COPY_FUNCS, torch.device("cpu"), _buffer)
    forward_context = {
        "gdn.0": SimpleNamespace(
            kv_cache=[
                torch.zeros((2, 2, 4), dtype=torch.bfloat16),
                torch.zeros((2, 2, 3, 4), dtype=torch.float32),
            ]
        ),
        "ple.1": SimpleNamespace(kv_cache=[torch.zeros((2, 3, 5), dtype=torch.bfloat16)]),
    }
    ctx.initialize_from_forward_context(
        config,
        forward_context,
        COPY_FUNCS,
        [torch.zeros((2, 4), dtype=torch.int32)],
    )
    assert ctx.num_states == 3
    assert ctx.state_elem_sizes.tolist() == [2, 4, 2]
    assert ctx.state_group_indices.tolist() == [0, 0, 0]
    expected_conv_widths = [4, 0, 5] if is_conv_state_dim_first() else [2, 0, 3]
    assert ctx.state_conv_widths.tolist() == expected_conv_widths


def test_legacy_tuple_only_allowed_for_single_mamba_type():
    gdn_spec = _spec(GDN, ((6, 1280), (6, 128, 128)))
    ple_spec = _spec(PLE, ((12, 10240),))
    homogeneous = _config([KVCacheGroupSpec(["gdn.0"], gdn_spec)])
    resolved = mamba_utils.patch_mamba_utils_resolve_copy_funcs(COPY_FUNCS[GDN], homogeneous)
    assert resolved == {GDN: COPY_FUNCS[GDN]}
    heterogeneous = _config(
        [
            KVCacheGroupSpec(["gdn.0"], gdn_spec),
            KVCacheGroupSpec(["ple.1"], ple_spec),
        ]
    )
    with pytest.raises(AssertionError, match="keyed by mamba_type"):
        mamba_utils.patch_mamba_utils_resolve_copy_funcs(COPY_FUNCS[GDN], heterogeneous)


def test_all_mode_postprocess_remains_compatible():
    spec = _spec(GDN, ((6, 1280), (6, 128, 128)), mode="all")
    config = _config([KVCacheGroupSpec(["gdn.0"], spec)])
    scheduler = SimpleNamespace(num_scheduled_tokens={"r": 4})
    batch = SimpleNamespace(req_ids=["r"])
    requests = {"r": SimpleNamespace(num_computed_tokens=768)}
    state_idx = {}
    mamba_utils.postprocess_mamba_all(scheduler, config, batch, requests, state_idx, 3, 1)
    assert state_idx == {"r": 1}


@pytest.mark.parametrize(
    ("prompt_len", "use_eagle", "expected_first_chunk"),
    [
        (1536, True, 768),
        (1600, True, 768),
        (2400, True, 1536),
        (1536, False, 1536),
        (1600, False, 1536),
        (2400, False, 2304),
    ],
)
def test_qwen_hybrid_prefill_splits_on_768_scheduler_page(prompt_len, use_eagle, expected_first_chunk):
    scheduler = SimpleNamespace(
        block_size=768,
        cache_config=SimpleNamespace(block_size=8),
        hash_block_size=768,
        mamba_partial_cache_hit=False,
        use_eagle=use_eagle,
    )
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        shared_prefix_boundary=0,
    )
    split = KVDeliveryScheduler._mamba_block_aligned_split(scheduler, request, prompt_len)
    assert split == expected_first_chunk


@pytest.mark.parametrize(
    ("prompt_len", "use_eagle", "expected_first_chunk"),
    [(16, True, 8), (17, True, 8), (16, False, 16), (17, False, 16)],
)
def test_aligned_split_supports_eight_token_scheduler_pages(prompt_len, use_eagle, expected_first_chunk):
    scheduler = SimpleNamespace(
        block_size=8,
        cache_config=SimpleNamespace(block_size=1),
        hash_block_size=8,
        mamba_partial_cache_hit=False,
        use_eagle=use_eagle,
    )
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=prompt_len,
        num_tokens=prompt_len,
        shared_prefix_boundary=0,
    )
    assert KVDeliveryScheduler._mamba_block_aligned_split(scheduler, request, prompt_len) == expected_first_chunk


def test_equal_scheduler_and_kv_block_size_uses_unmodified_upstream_path():
    scheduler = object.__new__(KVDeliveryScheduler)
    scheduler.block_size = 8
    scheduler.cache_config = SimpleNamespace(block_size=8)
    scheduler.hash_block_size = 8
    scheduler.mamba_partial_cache_hit = False
    scheduler.use_eagle = True
    request = SimpleNamespace(
        num_computed_tokens=0,
        num_prompt_tokens=17,
        num_tokens=17,
        shared_prefix_boundary=0,
    )
    expected = Scheduler._mamba_block_aligned_split(scheduler, request, 17)
    assert scheduler._mamba_block_aligned_split(request, 17) == expected
