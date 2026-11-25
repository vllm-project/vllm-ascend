# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
"""Real-Scheduler regression for the V2 Mooncake pull connector with speculative decoding.

Mirrors tests/ut/kv_offload/test_hybrid_prefix_truncation.py: the producer
scheduler truncates the last prompt token before the local prefix-cache
lookup. Building the real Scheduler mutates process-global state (KVCacheSpec
registry, prefix-cache hash seed); the autouse fixture below restores it.
"""

from unittest.mock import MagicMock

import pytest
import torch
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.utils.hashing import sha256
from vllm.v1 import kv_cache_spec_registry
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import (
    get_request_block_hasher,
    init_none_hash,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.kv_offload.utils import assert_scheduler_empty, create_model_runner_output, create_vllm_config
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec, AscendSlidingWindowMLASpec

# Factory registration names -> registered class, mirroring
# vllm_ascend/distributed/kv_transfer/__init__.py: both load the same V2
# pull connector implementation.
CONNECTOR_CLASSES = {
    "MooncakeConnectorV2": "MooncakeConnector",
    "MooncakePullConnector": "MooncakePullConnector",
}
NUM_SPECULATIVE_TOKENS = 5
RING_BLOCK_SIZE = 32
WARM_PREFIX_BLOCKS = 5
HIT_BOUNDARY_BLOCKS = 4


@pytest.fixture(autouse=True)
def _preserve_global_kv_state():
    """Restore the KVCacheSpec registry and prefix-cache hash state on exit."""
    spec_registry = dict(kv_cache_spec_registry._REGISTRY_KVCACHESPEC_LIST)
    role_managers = dict(kv_cache_spec_registry._REGISTRY_ROLE_MANAGERS)
    none_hash_seed = kv_cache_utils._NONE_HASH_SEED
    none_hash = getattr(kv_cache_utils, "NONE_HASH", None)
    yield
    # In-place clear+update also covers code holding a direct reference to
    # the dicts (the registry class methods resolve them through the module
    # globals, so rebinding the attributes would work for them as well).
    kv_cache_spec_registry._REGISTRY_KVCACHESPEC_LIST.clear()
    kv_cache_spec_registry._REGISTRY_KVCACHESPEC_LIST.update(spec_registry)
    kv_cache_spec_registry._REGISTRY_ROLE_MANAGERS.clear()
    kv_cache_spec_registry._REGISTRY_ROLE_MANAGERS.update(role_managers)
    kv_cache_utils._NONE_HASH_SEED = none_hash_seed
    if none_hash is not None:
        kv_cache_utils.NONE_HASH = none_hash
    elif hasattr(kv_cache_utils, "NONE_HASH"):
        del kv_cache_utils.NONE_HASH


@pytest.mark.parametrize("block_size", [128, 64])
@pytest.mark.parametrize("kv_connector", list(CONNECTOR_CLASSES))
@pytest.mark.parametrize("speculative_method", [None, "mtp", "dspark"])
def test_prefill_truncation_precedes_local_prefix_lookup(speculative_method, kv_connector, block_size, monkeypatch):
    monkeypatch.setattr(KVConnectorFactory, "_registry", KVConnectorFactory._registry.copy())
    if kv_connector not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            kv_connector,
            "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake.connector",
            CONNECTOR_CLASSES[kv_connector],
        )
    config = create_vllm_config(
        max_num_batched_tokens=2 * block_size,
        block_size=block_size,
        speculative_method=speculative_method,
        num_speculative_tokens=NUM_SPECULATIVE_TOKENS,
        kv_transfer_config=KVTransferConfig(kv_connector=kv_connector, kv_role="kv_producer"),
    )
    config.model_config.hf_config.compress_ratios = [1, 2]
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    full = AscendMLAAttentionSpec(block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.bfloat16)
    swa = AscendSlidingWindowMLASpec(
        block_size=block_size, num_kv_heads=1, head_size=8, dtype=torch.bfloat16, sliding_window=block_size
    )
    ring = CircularBufferSpec(
        block_size=RING_BLOCK_SIZE, num_kv_heads=1, head_size=16, head_size_v=0, dtype=torch.float32
    )
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
    resolved_block_size, hash_size = resolve_kv_cache_block_sizes(plan, config)
    # Match EngineCore initialization because the connector reads the cache config.
    plan.hash_block_size = hash_size
    scheduler = Scheduler(
        config, plan, StructuredOutputManager(config), block_size=resolved_block_size, hash_block_size=hash_size
    )
    # The producer scheduler normally starts its metadata thread when workers
    # register; this regression exercises scheduling only, so stub the thread.
    scheduler.connector.connector_scheduler._sending_thread = MagicMock(get_and_clear_finished_requests=lambda: set())

    def run(req_id, length):
        request = Request(
            request_id=req_id,
            prompt_token_ids=list(range(length)),
            pooling_params=None,
            sampling_params=SamplingParams(max_tokens=1, extra_args={"kv_transfer_params": {"do_remote_decode": True}}),
            block_hasher=get_request_block_hasher(hash_size, sha256, config.num_hash_lookahead_tokens),
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

    def expected_first_start(raw_length):
        # One token truncated by the producer, one reserved by the
        # always-recomputed last token, then floored to the block boundary.
        return (raw_length - 2) // block_size * block_size

    run("warm", WARM_PREFIX_BLOCKS * block_size + 1)
    starts = run("hit", HIT_BOUNDARY_BLOCKS * block_size + 1)
    assert starts[0] == expected_first_start(HIT_BOUNDARY_BLOCKS * block_size + 1)
    assert run("shorter", HIT_BOUNDARY_BLOCKS * block_size)[0] == expected_first_start(HIT_BOUNDARY_BLOCKS * block_size)
    assert run("longer", HIT_BOUNDARY_BLOCKS * block_size + 2)[0] == expected_first_start(
        HIT_BOUNDARY_BLOCKS * block_size + 2
    )
