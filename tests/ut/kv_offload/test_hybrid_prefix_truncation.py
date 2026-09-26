# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Raw-request scheduler regression; model compute and the remote ACK are fake."""

import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash, resolve_kv_cache_block_sizes
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import assert_scheduler_empty, create_model_runner_output, create_vllm_config
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec


@pytest.mark.parametrize("speculative_method", [None, "mtp", "dspark"])
def test_prefill_truncation_precedes_local_prefix_lookup(speculative_method, monkeypatch):
    monkeypatch.setattr(KVConnectorFactory, "_registry", KVConnectorFactory._registry.copy())
    if "MooncakeHybridConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "MooncakeHybridConnector",
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector",
            "MooncakeConnector",
        )
    config = create_vllm_config(
        max_num_batched_tokens=256,
        speculative_method=speculative_method,
        num_speculative_tokens=5,
        kv_transfer_config=KVTransferConfig(kv_connector="MooncakeHybridConnector", kv_role="kv_producer"),
    )
    config.model_config.hf_config.compress_ratios = [1, 2]
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    full = AscendMLAAttentionSpec(block_size=128, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    swa = AscendSlidingWindowMLASpec(
        block_size=128, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=128
    )
    ring = CircularBufferSpec(block_size=32, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32)
    plan = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["full"], full),
            KVCacheGroupSpec(["state"], ring),
            KVCacheGroupSpec(["swa"], swa, is_eagle_group=speculative_method is not None),
        ],
    )
    plan.kv_transfer_config = config.kv_transfer_config
    config.cache_config.num_gpu_blocks = plan.num_blocks
    register_all_kvcache_specs(None)
    init_none_hash(sha256)
    block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    scheduler = Scheduler(
        config, plan, StructuredOutputManager(config), block_size=block_size, hash_block_size=hash_size
    )

    def run(req_id, length):
        request = Request(
            request_id=req_id,
            prompt_token_ids=list(range(length)),
            pooling_params=None,
            sampling_params=SamplingParams(max_tokens=1, extra_args={"kv_transfer_params": {"do_remote_decode": True}}),
            block_hasher=get_request_block_hasher(hash_size, sha256),
        )
        scheduler.add_request(request)
        starts = []
        for _ in range(12):
            output = scheduler.schedule()
            scheduled = output.num_scheduled_tokens.get(req_id, 0)
            if scheduled:
                starts.append(request.num_computed_tokens - scheduled)
            result = create_model_runner_output([request] if scheduled else [])
            if scheduled and request.num_computed_tokens < request.num_prompt_tokens:
                result.sampled_token_ids = [[]]
            scheduler.update_from_output(output, result)
            if request.is_finished():
                # Remote transport completion is downstream of the failing lookup.
                output = scheduler.schedule()
                result = create_model_runner_output([])
                result.kv_connector_output = KVConnectorOutput(finished_sending={req_id})
                scheduler.update_from_output(output, result)
                scheduler.schedule()
                break
        assert request.is_finished()
        assert request.num_prompt_tokens == length - 1
        assert_scheduler_empty(scheduler)
        return starts

    run("warm", 641)
    starts = run("hit", 513)
    assert starts[0] == 384
    assert run("shorter", 512)[0] == 384
    assert run("longer", 514)[0] == 512
