# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project

from types import SimpleNamespace

import pytest
import torch
import vllm.v1.core.kv_cache_utils as upstream_kv_cache_utils
import vllm.v1.engine.core as engine_core
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import MambaSpec, UniformTypeKVCacheSpecs

import vllm_ascend.patch.platform.patch_kv_cache_utils as kv_cache_utils_patch
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec

BLOCK_SIZE = 768
MAIN_PAGE_SIZE = 537_600
DRAFT_PAGE_SIZE = BLOCK_SIZE * 576 * get_dtype_size(torch.bfloat16)
MAX_MODEL_LEN = BLOCK_SIZE * 2
NUM_SPECULATIVE_BLOCKS = 7
NUM_BLOCKS_PER_REQUEST = 2 + 2 * (2 + NUM_SPECULATIVE_BLOCKS)
PROFILED_NUM_BLOCKS = 12
EXPECTED_CONCURRENCY = PROFILED_NUM_BLOCKS / NUM_BLOCKS_PER_REQUEST
EXPECTED_TOKEN_CAPACITY = int(EXPECTED_CONCURRENCY * MAX_MODEL_LEN)


def _make_vllm_config(
    *,
    num_gpu_blocks_override: int | None = None,
    method: str = "dspark",
    draft_model_type: str = "k3_dspark",
    use_v2_model_runner: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        use_v2_model_runner=use_v2_model_runner,
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="wrapped_multimodal_model"),
            hf_text_config=SimpleNamespace(
                model_type="kimi_linear",
                attn_res_block_size=3,
            ),
            max_model_len=MAX_MODEL_LEN,
            original_max_model_len=MAX_MODEL_LEN,
        ),
        speculative_config=SimpleNamespace(
            method=method,
            draft_model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type=draft_model_type),
            ),
        ),
        scheduler_config=SimpleNamespace(
            disable_hybrid_kv_cache_manager=False,
        ),
        cache_config=SimpleNamespace(
            mamba_cache_mode="align",
            num_gpu_blocks_override=num_gpu_blocks_override,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
        ),
        kv_transfer_config=None,
    )


def _make_kimi_k3_kv_cache_specs() -> dict[str, AscendMLAAttentionSpec | MambaSpec]:
    target_spec = AscendMLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=640,
        dtype=torch.int8,
        page_size_padded=MAIN_PAGE_SIZE,
    )
    draft_spec = AscendMLAAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=576,
        dtype=torch.bfloat16,
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,),),
        dtypes=(torch.bfloat16,),
        page_size_padded=MAIN_PAGE_SIZE,
        mamba_cache_mode="align",
        num_speculative_blocks=NUM_SPECULATIVE_BLOCKS,
    )
    assert target_spec.page_size_bytes == MAIN_PAGE_SIZE
    assert target_spec.unpadded_page_size_bytes < MAIN_PAGE_SIZE
    assert target_spec.compress_ratio == draft_spec.compress_ratio == 1
    assert draft_spec.page_size_bytes == DRAFT_PAGE_SIZE
    assert mamba_spec.num_speculative_blocks == NUM_SPECULATIVE_BLOCKS
    return {
        "target.0": target_spec,
        "target.1": target_spec,
        "draft.0": draft_spec,
        "mamba.0": mamba_spec,
        "mamba.1": mamba_spec,
        "mamba.2": mamba_spec,
        "mamba.3": mamba_spec,
    }


def _make_groups():
    vllm_config = _make_vllm_config()
    specs = _make_kimi_k3_kv_cache_specs()
    groups = kv_cache_utils_patch.get_kv_cache_groups(vllm_config, specs)
    return vllm_config, specs, groups


def test_kimi_k3_dspark_draft_shares_target_logical_group_only() -> None:
    _, specs, groups = _make_groups()

    assert len(groups) == 3
    target_draft_group = groups[0]
    assert target_draft_group.layer_names == ["target.0", "target.1", "draft.0"]
    assert target_draft_group.is_eagle_group
    assert isinstance(target_draft_group.kv_cache_spec, UniformTypeKVCacheSpecs)
    assert target_draft_group.kv_cache_spec.kv_cache_specs == {
        "target.0": specs["target.0"],
        "target.1": specs["target.1"],
        "draft.0": specs["draft.0"],
    }

    assert [group.layer_names for group in groups[1:]] == [
        ["mamba.0", "mamba.2"],
        ["mamba.1", "mamba.3"],
    ]
    assert all(not group.is_eagle_group for group in groups[1:])
    assert all(group.kv_cache_spec.page_size_bytes == MAIN_PAGE_SIZE for group in groups[1:])


def test_kimi_k3_dspark_uses_main_hybrid_tensors_then_contiguous_draft_tensor() -> None:
    vllm_config, _, groups = _make_groups()
    bytes_per_global_block = 2 * MAIN_PAGE_SIZE + DRAFT_PAGE_SIZE
    kv_cache_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=bytes_per_global_block * 11,
    )

    assert kv_cache_config.num_blocks == 11
    assert [tensor.shared_by for tensor in kv_cache_config.kv_cache_tensors] == [
        ["target.0", "mamba.0", "mamba.1"],
        ["target.1", "mamba.2", "mamba.3"],
        ["draft.0"],
    ]
    assert [tensor.size for tensor in kv_cache_config.kv_cache_tensors] == [
        MAIN_PAGE_SIZE * 11,
        MAIN_PAGE_SIZE * 11,
        DRAFT_PAGE_SIZE * 11,
    ]
    assert all(tensor.offset == 0 for tensor in kv_cache_config.kv_cache_tensors)
    assert all(tensor.block_stride == 0 for tensor in kv_cache_config.kv_cache_tensors)


def test_kimi_k3_dspark_pool_memory_and_concurrency_use_global_block_ids() -> None:
    vllm_config, _, groups = _make_groups()
    bytes_per_global_block = 2 * MAIN_PAGE_SIZE + DRAFT_PAGE_SIZE

    assert kv_cache_utils_patch._pool_bytes_per_block(vllm_config, groups) == bytes_per_global_block
    assert (
        kv_cache_utils_patch._max_memory_usage_bytes_from_groups(vllm_config, groups)
        == bytes_per_global_block * NUM_BLOCKS_PER_REQUEST
    )

    kv_cache_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=bytes_per_global_block * PROFILED_NUM_BLOCKS,
    )
    concurrency = kv_cache_utils_patch.get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)
    assert concurrency == pytest.approx(EXPECTED_CONCURRENCY)
    token_capacity, concurrency = upstream_kv_cache_utils.get_kv_cache_capacity(vllm_config, kv_cache_config)
    assert token_capacity == EXPECTED_TOKEN_CAPACITY
    assert concurrency == pytest.approx(EXPECTED_CONCURRENCY)


def test_kimi_k3_dspark_num_blocks_override_uses_bucketed_pool_size() -> None:
    vllm_config, _, groups = _make_groups()
    vllm_config.cache_config.num_gpu_blocks_override = 7
    bytes_per_global_block = 2 * MAIN_PAGE_SIZE + DRAFT_PAGE_SIZE

    kv_cache_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=bytes_per_global_block * PROFILED_NUM_BLOCKS,
    )

    assert kv_cache_config.num_blocks == 7
    assert [tensor.size for tensor in kv_cache_config.kv_cache_tensors] == [
        MAIN_PAGE_SIZE * 7,
        MAIN_PAGE_SIZE * 7,
        DRAFT_PAGE_SIZE * 7,
    ]


def test_kimi_k3_dspark_scheduler_uses_target_spec_for_uniform_group(monkeypatch) -> None:
    vllm_config, _, groups = _make_groups()
    bytes_per_global_block = 2 * MAIN_PAGE_SIZE + DRAFT_PAGE_SIZE
    worker_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        groups,
        available_memory=bytes_per_global_block * PROFILED_NUM_BLOCKS,
    )

    scheduler_config = upstream_kv_cache_utils.generate_scheduler_kv_cache_config([worker_config])

    scheduler_target_spec = scheduler_config.kv_cache_groups[0].kv_cache_spec
    assert isinstance(scheduler_target_spec, AscendMLAAttentionSpec)
    assert scheduler_target_spec.dtype == torch.int8
    assert scheduler_target_spec.page_size_bytes == MAIN_PAGE_SIZE
    assert scheduler_config.kv_cache_groups[0].is_eagle_group
    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_orig_get_max_concurrency_for_kv_cache_config",
        lambda *_: pytest.fail("K3 scheduler layout should use group-aware concurrency"),
    )
    token_capacity, concurrency = upstream_kv_cache_utils.get_kv_cache_capacity(vllm_config, scheduler_config)
    assert token_capacity == EXPECTED_TOKEN_CAPACITY
    assert concurrency == pytest.approx(EXPECTED_CONCURRENCY)


def test_kimi_k3_dspark_pp_projection_handles_draft_only_worker_and_empty_groups(monkeypatch) -> None:
    vllm_config, specs, global_groups = _make_groups()
    projected_groups = upstream_kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups,
        {"draft.0": specs["draft.0"]},
    )

    assert projected_groups[0].layer_names == ["draft.0"]
    assert all(not group.layer_names for group in projected_groups[1:])
    assert kv_cache_utils_patch._pool_bytes_per_block(vllm_config, projected_groups) == DRAFT_PAGE_SIZE
    assert (
        kv_cache_utils_patch._max_memory_usage_bytes_from_groups(vllm_config, projected_groups)
        == DRAFT_PAGE_SIZE * NUM_BLOCKS_PER_REQUEST
    )

    kv_cache_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        projected_groups,
        available_memory=DRAFT_PAGE_SIZE * PROFILED_NUM_BLOCKS,
    )
    assert kv_cache_config.num_blocks == PROFILED_NUM_BLOCKS
    assert len(kv_cache_config.kv_cache_tensors) == 1
    assert kv_cache_config.kv_cache_tensors[0].shared_by == ["draft.0"]
    assert kv_cache_config.kv_cache_tensors[0].size == DRAFT_PAGE_SIZE * PROFILED_NUM_BLOCKS
    assert kv_cache_utils_patch.get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config) == pytest.approx(
        EXPECTED_CONCURRENCY
    )

    scheduler_config = upstream_kv_cache_utils.generate_scheduler_kv_cache_config([kv_cache_config])
    assert scheduler_config.kv_cache_groups[0].kv_cache_spec.dtype == torch.bfloat16
    monkeypatch.setattr(
        kv_cache_utils_patch,
        "_orig_get_max_concurrency_for_kv_cache_config",
        lambda *_: pytest.fail("draft-only K3 scheduler layout should use group-aware concurrency"),
    )
    token_capacity, concurrency = upstream_kv_cache_utils.get_kv_cache_capacity(vllm_config, scheduler_config)
    assert token_capacity == EXPECTED_TOKEN_CAPACITY
    assert concurrency == pytest.approx(EXPECTED_CONCURRENCY)


def test_kimi_k3_dspark_pp_projection_handles_mamba_only_worker() -> None:
    vllm_config, specs, global_groups = _make_groups()
    projected_groups = upstream_kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups,
        {"mamba.0": specs["mamba.0"]},
    )

    assert not projected_groups[0].layer_names
    assert kv_cache_utils_patch._pool_bytes_per_block(vllm_config, projected_groups) == MAIN_PAGE_SIZE
    assert (
        kv_cache_utils_patch._max_memory_usage_bytes_from_groups(vllm_config, projected_groups)
        == MAIN_PAGE_SIZE * NUM_BLOCKS_PER_REQUEST
    )

    kv_cache_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        projected_groups,
        available_memory=MAIN_PAGE_SIZE * PROFILED_NUM_BLOCKS,
    )
    assert kv_cache_config.num_blocks == PROFILED_NUM_BLOCKS
    assert len(kv_cache_config.kv_cache_tensors) == 1
    assert kv_cache_config.kv_cache_tensors[0].shared_by == ["mamba.0"]


def test_scheduler_eagle_flag_is_or_reduced_across_pp_workers() -> None:
    vllm_config, specs, global_groups = _make_groups()
    mamba_only_groups = upstream_kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups,
        {"mamba.0": specs["mamba.0"]},
    )
    target_draft_groups = upstream_kv_cache_utils._project_kv_cache_groups_to_worker(
        global_groups,
        {
            "target.0": specs["target.0"],
            "target.1": specs["target.1"],
            "draft.0": specs["draft.0"],
        },
    )
    mamba_only_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        mamba_only_groups,
        available_memory=MAIN_PAGE_SIZE * PROFILED_NUM_BLOCKS,
    )
    target_draft_bytes_per_block = 2 * MAIN_PAGE_SIZE + DRAFT_PAGE_SIZE
    target_draft_config = kv_cache_utils_patch.get_kv_cache_config_from_groups(
        vllm_config,
        target_draft_groups,
        available_memory=target_draft_bytes_per_block * PROFILED_NUM_BLOCKS,
    )

    assert [group.is_eagle_group for group in mamba_only_config.kv_cache_groups] == [False, False, False]
    assert [group.is_eagle_group for group in target_draft_config.kv_cache_groups] == [True, False, False]
    assert (
        upstream_kv_cache_utils.generate_scheduler_kv_cache_config
        is kv_cache_utils_patch.generate_scheduler_kv_cache_config
    )
    assert engine_core.generate_scheduler_kv_cache_config is kv_cache_utils_patch.generate_scheduler_kv_cache_config

    scheduler_config = engine_core.generate_scheduler_kv_cache_config([mamba_only_config, target_draft_config])

    assert [group.is_eagle_group for group in scheduler_config.kv_cache_groups] == [True, False, False]
    token_capacity, concurrency = upstream_kv_cache_utils.get_kv_cache_capacity(vllm_config, scheduler_config)
    assert token_capacity == EXPECTED_TOKEN_CAPACITY
    assert concurrency == pytest.approx(EXPECTED_CONCURRENCY)


@pytest.mark.parametrize(
    ("config_overrides"),
    [
        pytest.param({"method": "mtp"}, id="not-dspark"),
        pytest.param({"draft_model_type": "deepseek_mtp"}, id="not-k3-draft"),
        pytest.param({"use_v2_model_runner": True}, id="v2-fails-closed"),
    ],
)
def test_kimi_k3_special_layout_is_narrow(config_overrides: dict[str, object]) -> None:
    vllm_config = _make_vllm_config(**config_overrides)

    assert (
        kv_cache_utils_patch._get_kimi_k3_c8_dspark_spec_partition(
            vllm_config,
            _make_kimi_k3_kv_cache_specs(),
        )
        is None
    )


def test_non_kimi_grouping_delegates_to_existing_dsv4_mtp_path(monkeypatch) -> None:
    sentinel = [object()]
    vllm_config = _make_vllm_config(draft_model_type="deepseek_mtp")
    monkeypatch.setattr(kv_cache_utils_patch, "_orig_get_kv_cache_groups", lambda *_: sentinel)

    assert kv_cache_utils_patch.get_kv_cache_groups(vllm_config, _make_kimi_k3_kv_cache_specs()) is sentinel
