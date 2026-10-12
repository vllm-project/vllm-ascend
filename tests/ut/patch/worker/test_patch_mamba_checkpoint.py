# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.model_executor.layers.mamba import checkpoint
from vllm.model_executor.layers.mamba.checkpoint import MambaPrefillCheckpointBuilder
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.kv_cache_interface import MambaSpec, get_mamba_prefill_checkpoint_position

from vllm_ascend.patch.platform.patch_kv_cache_coordinator import _effective_eagle_block_drop
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager
from vllm_ascend.patch.worker import patch_mamba_checkpoint


def _make_builder(role, num_checkpoint_blocks=1, use_eagle=True):
    speculative_config = SimpleNamespace(
        use_eagle_block_drop=lambda: use_eagle,
        use_eagle=lambda: use_eagle,
        num_speculative_tokens=5,
    )
    config = SimpleNamespace(
        speculative_config=speculative_config,
        kv_transfer_config=role,
        cache_config=SimpleNamespace(
            prefix_match_unit=384,
            hash_block_size=384,
            cache_hit_alignment_tokens=384,
            mamba_cache_mode="align",
        ),
    )
    spec = SimpleNamespace(
        num_prefill_checkpoint_blocks=num_checkpoint_blocks,
        block_size=384,
        prefill_checkpoint_alignment=128,
    )
    return MambaPrefillCheckpointBuilder(config, spec)


def _make_metadata(prompt_length):
    query_start_loc = torch.tensor([0, prompt_length], dtype=torch.int32)
    return SimpleNamespace(
        seq_lens_cpu_upper_bound=torch.tensor([prompt_length]),
        query_start_loc_cpu=query_start_loc,
        query_start_loc=query_start_loc,
        block_table_tensor=torch.arange(342, dtype=torch.int64).unsqueeze(0) + 100,
    )


@pytest.mark.parametrize("prompt_length", [131082, 131328])
@pytest.mark.parametrize(
    "role, expected_drop",
    [
        (None, False),
        (SimpleNamespace(is_kv_producer=True, is_kv_consumer=False), False),
        (SimpleNamespace(is_kv_producer=False, is_kv_consumer=True), True),
        (SimpleNamespace(is_kv_producer=True, is_kv_consumer=True), True),
        (SimpleNamespace(is_kv_producer=False, is_kv_consumer=False), True),
    ],
    ids=["standalone", "producer", "consumer", "kv_both", "neither"],
)
def test_checkpoint_export_uses_effective_drop_without_changing_mtp(role, expected_drop, prompt_length):
    builder = _make_builder(role)
    original_config = builder.vllm_config
    original_speculative_config = original_config.speculative_config
    metadata = _make_metadata(prompt_length)
    # Exercise the installed method on the class imported by attention backends.
    assert MambaPrefillCheckpointBuilder.build is patch_mamba_checkpoint.build_mamba_prefill_checkpoint
    with patch.object(
        checkpoint, "async_tensor_h2d", side_effect=lambda data, *, dtype, device: torch.tensor(data, dtype=dtype)
    ):
        actual = builder.build(metadata, [0])

    assert actual is not None
    if not expected_drop:
        assert actual.checkpoint_offsets.tolist() == [130944]
    else:
        expected_position = get_mamba_prefill_checkpoint_position(prompt_length, 384, drop_eagle_block=True)
        assert actual.checkpoint_offsets.tolist() == [expected_position]
    # Both aligned and non-aligned prompts reserve column 340 for this export.
    assert actual.state_indices.tolist() == [440]
    assert builder.vllm_config is original_config
    assert builder.vllm_config.speculative_config is original_speculative_config
    assert original_speculative_config.use_eagle_block_drop() is True
    assert original_speculative_config.use_eagle() is True
    assert original_speculative_config.num_speculative_tokens == 5


@pytest.mark.parametrize("use_eagle, num_checkpoint_blocks", [(False, 1), (True, 0)])
def test_checkpoint_no_drop_or_disabled_keeps_original_builder(use_eagle, num_checkpoint_blocks):
    builder = _make_builder(None, num_checkpoint_blocks, use_eagle)
    metadata = _make_metadata(131082)
    sentinel = object()
    with patch.object(patch_mamba_checkpoint, "_ORIGINAL_BUILD", return_value=sentinel) as original_build:
        assert builder.build(metadata, [0]) is sentinel
    original_build.assert_called_once_with(builder, metadata, [0])


def test_checkpoint_without_speculation_keeps_original_builder():
    builder = _make_builder(None)
    builder.vllm_config.speculative_config = None
    metadata = _make_metadata(131082)
    sentinel = object()
    with patch.object(patch_mamba_checkpoint, "_ORIGINAL_BUILD", return_value=sentinel) as original_build:
        assert builder.build(metadata, [0]) is sentinel
    original_build.assert_called_once_with(builder, metadata, [0])


@pytest.mark.parametrize("prompt_length", [131082, 131328], ids=["unaligned", "aligned"])
def test_checkpoint_allocation_export_and_sparse_registration_use_same_state(prompt_length):
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=384)
    spec = MambaSpec(
        block_size=384,
        shapes=((1,),),
        dtypes=(torch.int64,),
        mamba_cache_mode="align",
        num_speculative_blocks=7,
        num_prefill_checkpoint_blocks=1,
        prefill_checkpoint_alignment=128,
    )
    manager = AscendMambaManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=384,
    )
    manager.cache_hit_alignment_tokens = 384
    manager.skips_eagle_block_drop = True
    manager.drop_eagle_checkpoint_block = _effective_eagle_block_drop(True, None)
    request = SimpleNamespace(
        request_id="prime",
        num_prompt_tokens=prompt_length,
        num_computed_tokens=0,
        shared_prefix_boundary=0,
        block_hashes=[i.to_bytes(32, "little") for i in range(1, prompt_length // 384 + 1)],
    )

    # Select and reserve the internal checkpoint through the real admission
    # and allocation paths rather than seeding the manager's checkpoint map.
    num_blocks = manager.get_num_blocks_to_allocate(
        request_id=request.request_id,
        num_tokens=prompt_length,
        new_computed_blocks=(),
        total_computed_tokens=0,
        num_local_computed_tokens=0,
        num_tokens_main_model=prompt_length,
        prefill_end=prompt_length,
    )
    checkpoint_position, checkpoint_col = manager._checkpoints[request.request_id]
    assert checkpoint_position == 130944
    allocated = manager.allocate_new_blocks(request.request_id, prompt_length, prompt_length)
    # The returned logical table includes null gaps before the live states.
    assert sum(not block.is_null for block in allocated) == num_blocks
    blocks = manager.req_to_blocks[request.request_id]
    checkpoint_block = blocks[checkpoint_col]
    assert not checkpoint_block.is_null

    builder = _make_builder(None)
    builder.kv_cache_spec = spec
    original_config = builder.vllm_config
    original_speculative_config = original_config.speculative_config
    metadata = _make_metadata(prompt_length)
    metadata.block_table_tensor = torch.tensor([[block.block_id for block in blocks]], dtype=torch.int64)
    with patch.object(
        checkpoint, "async_tensor_h2d", side_effect=lambda data, *, dtype, device: torch.tensor(data, dtype=dtype)
    ):
        exported = builder.build(metadata, [0])
    assert exported is not None
    assert exported.checkpoint_offsets.tolist() == [checkpoint_position]
    assert exported.state_indices.tolist() == [checkpoint_block.block_id]

    # Synthetic CPU state records the processed-token count. This checks the
    # allocation/export/registration metadata contract, not an NPU kernel.
    state = torch.zeros(pool.num_gpu_blocks, dtype=torch.int64)
    state[exported.state_indices] = exported.checkpoint_offsets.to(torch.int64)
    manager.cache_blocks(
        request,
        prompt_length // 384 * 384,
        retention_interval=0,
        replay_boundaries=(prompt_length - 1, prompt_length),
    )
    checkpoint_hash = request.block_hashes[checkpoint_position // 384 - 1]
    assert pool.get_cached_block(checkpoint_hash, [0]) == [checkpoint_block]
    assert checkpoint_block.block_hash_num_tokens == checkpoint_position
    assert state[checkpoint_block.block_id].item() == checkpoint_position
    assert builder.vllm_config is original_config
    assert original_config.speculative_config is original_speculative_config
    assert original_speculative_config.use_eagle_block_drop() is True
    assert original_speculative_config.num_speculative_tokens == 5
