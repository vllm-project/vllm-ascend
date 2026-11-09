# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import create_request, create_vllm_config
from vllm_ascend.core.kv_cache_interface import AscendIndexerKPoolTailSpec, register_ascend_kv_cache_specs
from vllm_ascend.core.recompute_scheduler import (
    AsyncRecomputeScheduler,
)
from vllm_ascend.models.glm5next.kv_cache import KpoolTailManager
from vllm_ascend.patch.platform.patch_mamba_manager import AscendMambaManager


@pytest.fixture(autouse=True)
def _oproj_tp_config(monkeypatch):
    # _preempt_or_recompute reads the live AscendConfig singleton; default it to off.
    finegrained_tp_config = SimpleNamespace(oproj_tensor_parallel_size=0, mlp_tensor_parallel_size=0)
    monkeypatch.setattr(
        "vllm_ascend.core.recompute_scheduler.get_ascend_config",
        lambda: SimpleNamespace(finegrained_tp_config=finegrained_tp_config),
    )
    return finegrained_tp_config


@pytest.mark.parametrize("reject_drafts", [False, True])
def test_registered_hybrid_transition_headroom_completes_decode(reject_drafts):
    config = create_vllm_config(
        max_num_seqs=7,
        max_num_batched_tokens=65536,
        block_size=8704,
        speculative_method="mtp",
        num_speculative_tokens=5,
    )
    config.model_config.max_model_len = 65536
    config.scheduler_config.max_num_batched_tokens = 192
    config.scheduler_config.async_scheduling = True
    config.scheduler_config.watermark = 0.0
    config.cache_config.enable_prefix_caching = False
    config.cache_config.mamba_cache_mode = "align"
    config.kv_transfer_config = None
    groups = [
        KVCacheGroupSpec(
            ["attention"], FullAttentionSpec(block_size=8704, num_kv_heads=1, head_size=1, dtype=torch.float32)
        ),
        KVCacheGroupSpec(
            ["ring"],
            AscendIndexerKPoolTailSpec(
                block_size=9, sliding_window=4, compress_ratio=4, num_kv_heads=1, head_size=128, dtype=torch.float32
            ),
        ),
        *[
            KVCacheGroupSpec(
                [f"mamba_{i}"],
                MambaSpec(
                    block_size=8704,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=5,
                ),
            )
            for i in range(3)
        ],
    ]
    kv_config = KVCacheConfig(num_blocks=149, kv_cache_tensors=[], kv_cache_groups=groups)
    config.cache_config.num_gpu_blocks = kv_config.num_blocks
    register_all_kvcache_specs(config)
    register_ascend_kv_cache_specs()
    scheduler = AsyncRecomputeScheduler(
        vllm_config=config,
        kv_cache_config=kv_config,
        log_stats=False,
        block_size=8704,
        hash_block_size=128,
        structured_output_manager=StructuredOutputManager(config),
    )
    scheduler.use_v2_model_runner = False
    managers = scheduler.kv_cache_manager.coordinator.single_type_managers
    assert type(managers[1]) is KpoolTailManager
    assert all(isinstance(manager, AscendMambaManager) for manager in managers[2:])
    assert scheduler.num_lookahead_tokens == 5
    requests = [create_request(i, num_tokens=8499, max_tokens=1024, block_size=128) for i in range(7)]
    for request in requests:
        scheduler.add_request(request)
        assert (
            scheduler.kv_cache_manager.allocate_slots(
                request, 0, num_external_computed_tokens=8498, delay_cache_blocks=True
            )
            is not None
        )
        scheduler.waiting.remove_requests([request])
        request.status = RequestStatus.RUNNING
        request.append_output_token_ids([7] * 196)
        request.num_computed_tokens = 8694
        request.spec_token_ids = [7] * 5
        request.is_prefill_chunk = False
        scheduler.running.append(request)
    pool = scheduler.kv_cache_manager.block_pool
    assert pool.get_num_free_blocks() == 8
    for _ in range(7 * (1024 - 196) + 1):
        if not scheduler.running:
            break
        output = scheduler.schedule()
        assert not output.preempted_req_ids
        assert not output.recomputed_reqs
        assert output.total_num_scheduled_tokens > 0
        ids = list(output.num_scheduled_tokens)
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=[
                    [7] * (1 if reject_drafts else 1 + len(output.scheduled_spec_decode_tokens.get(rid, [])))
                    for rid in ids
                ],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            ),
        )
    assert not scheduler.running and not scheduler.waiting
    assert all(request.num_output_tokens == 1024 and request.num_preemptions == 0 for request in requests)
    assert pool.get_num_free_blocks() == 148
