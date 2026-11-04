# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Release-lane configuration and bounded replay allocation regression tests."""

import argparse
import dataclasses
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import msgspec
import pytest
import torch
from vllm.config import CacheConfig, DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    SlidingWindowManager,
    register_all_kvcache_specs,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, UniformTypeKVCacheSpecs
from vllm.v1.request import Request, RequestStatus

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec, register_ascend_kv_cache_specs
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_connector import MooncakeConnector
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import (
    MooncakeConnector as MooncakeHybridConnector,
)
from vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy import (
    _initialize_replay_window,
    _mark_prefix_replay,
    _patch_scheduler,
    _rewrite_method,
    apply_legacy_replay_patch,
    register_legacy_replay_argument,
)

pytestmark = pytest.mark.skipif(
    hasattr(Scheduler, "_mark_prefix_replay")
    and Scheduler._mark_prefix_replay.__module__ != apply_legacy_replay_patch.__module__,
    reason="vLLM supplies native bounded replay",
)

BLOCK_SIZE = 16
WINDOW = 32
HIT_TOKENS = 96


@pytest.fixture(autouse=True)
def enable_legacy_replay():
    apply_legacy_replay_patch(SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=True)))


@pytest.mark.parametrize("enabled", [False, True])
def test_replay_installation_is_opt_in_and_idempotent_in_fresh_process(enabled):
    # A fresh interpreter checks opt-in installation after normal plugin
    # registration. Use the same device-boundary mocks as the parent CPU UT
    # session; the host may have torch_npu installed without CANN libraries.
    # The fixtures register wire fields but must not install scheduler replay.
    script = """
import tests.ut.conftest
from types import SimpleNamespace
import msgspec
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.core.sched.output import NewRequestData, CachedRequestData
originals = (Scheduler.schedule, KVCacheManager.allocate_slots, SingleTypeKVCacheManager.cache_blocks)
from vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy import apply_legacy_replay_patch
assert not hasattr(Scheduler, '_mark_prefix_replay')
assert originals == (Scheduler.schedule, KVCacheManager.allocate_slots, SingleTypeKVCacheManager.cache_blocks)
assert 'replay_start' in NewRequestData.__dataclass_fields__
assert 'replay_start' in CachedRequestData.__dataclass_fields__
config = SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=ENABLED))
apply_legacy_replay_patch(config)
installed = (Scheduler.schedule, KVCacheManager.allocate_slots, SingleTypeKVCacheManager.cache_blocks)
assert (originals != installed) == ENABLED
apply_legacy_replay_patch(config)
assert installed == (Scheduler.schedule, KVCacheManager.allocate_slots, SingleTypeKVCacheManager.cache_blocks)
""".replace("ENABLED", repr(enabled))
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_native_replay_is_preserved():
    config = SimpleNamespace(cache_config=SimpleNamespace(swa_bounded_replay=True))
    with (
        patch(
            "vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy.Scheduler",
            SimpleNamespace(_mark_prefix_replay=lambda *args: None),
        ),
        patch("vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy._patch_scheduler") as install,
    ):
        apply_legacy_replay_patch(config)
        install.assert_not_called()


def test_changed_scheduler_body_does_not_partially_patch_scheduler():
    class UnsupportedScheduler:
        def __init__(self):
            pass

        def schedule(self):
            return "unchanged"

    with patch("vllm_ascend.patch.platform.patch_swa_bounded_replay_legacy.Scheduler", UnsupportedScheduler):
        original_init = UnsupportedScheduler.__init__
        with pytest.raises(RuntimeError, match="unexpected UnsupportedScheduler.schedule body"):
            _patch_scheduler()
        assert UnsupportedScheduler.__init__ is original_init


@pytest.mark.parametrize("enabled", [False, True])
def test_uniform_group_propagates_replay_window(enabled):
    spec = AscendSlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        bounded_replay=enabled,
    )
    group = UniformTypeKVCacheSpecs(block_size=BLOCK_SIZE, kv_cache_specs={"swa": spec})
    assert group.prefix_replay_tokens == (WINDOW if enabled else 0)


@pytest.mark.parametrize(
    "flag,expected", [(None, False), ("--swa-bounded-replay", True), ("--no-swa-bounded-replay", False)]
)
def test_cli_switch_reaches_engine_args_and_cache_schema(flag, expected):
    parser = argparse.ArgumentParser()
    register_legacy_replay_argument(parser)
    register_legacy_replay_argument(parser)
    parsed = parser.parse_args([] if flag is None else [flag])
    args = AsyncEngineArgs.from_cli_args(parsed)
    assert args.swa_bounded_replay is expected
    cache = CacheConfig(swa_bounded_replay=args.swa_bounded_replay)
    assert cache.swa_bounded_replay is expected
    assert dataclasses.asdict(cache)["swa_bounded_replay"] is expected
    assert EngineArgs(swa_bounded_replay=expected).swa_bounded_replay is expected


@pytest.mark.parametrize("hit,start,replayed", [(0, 0, 0), (16, 0, 16), (32, 0, 32), (96, 64, 32)])
def test_admission_rewind_clears_previous_preemption_boundary(hit, start, replayed):
    scheduler = SimpleNamespace(prefix_replay_tokens=WINDOW)
    request = SimpleNamespace(replay_start=128)
    assert _mark_prefix_replay(scheduler, request, hit) == replayed
    assert request.replay_start == start


def test_disabled_replay_clears_boundary():
    request = SimpleNamespace(replay_start=64)
    assert _mark_prefix_replay(SimpleNamespace(prefix_replay_tokens=0), request, HIT_TOKENS) == 0
    assert request.replay_start == 0


def test_new_request_replay_boundary_survives_msgpack():
    payload = NewRequestData(
        req_id="replayed",
        prompt_token_ids=[1],
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([1],),
        num_computed_tokens=64,
        lora_request=None,
        replay_start=64,
    )
    restored = NewRequestData(**msgspec.msgpack.decode(msgspec.msgpack.encode(payload)))
    assert restored.replay_start == 64


def test_replay_window_is_private_and_does_not_cache_swa_blocks():
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=BLOCK_SIZE)
    spec = AscendSlidingWindowMLASpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.bfloat16,
        sliding_window=WINDOW,
        bounded_replay=True,
    )
    manager = SlidingWindowManager(
        kv_cache_spec=spec,
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )
    manager.add_local_computed_blocks("replayed", [], HIT_TOKENS, 0)
    manager.allocate_new_blocks("replayed", HIT_TOKENS, HIT_TOKENS)
    blocks = manager.req_to_blocks["replayed"]
    assert all(b.is_null for b in blocks[:4])
    assert all(not b.is_null for b in blocks[4:6])
    # No request hashes are needed: a replay SWA group must not enter caching.
    manager.cache_blocks(SimpleNamespace(request_id="replayed"), HIT_TOKENS)
    assert all(b.block_hash is None for b in blocks[4:6])
    assert not spec.prefix_cacheable
    assert spec.prefix_replay_tokens == WINDOW


def test_cacheable_group_adopts_entire_hit_when_first_replay_chunk_is_short():
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=BLOCK_SIZE)
    manager = FullAttentionManager(
        kv_cache_spec=FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=128, dtype=torch.bfloat16),
        block_pool=pool,
        enable_caching=True,
        kv_cache_group_id=0,
        scheduler_block_size=BLOCK_SIZE,
    )
    shared = pool.get_new_blocks(HIT_TOKENS // BLOCK_SIZE)
    manager.add_local_computed_blocks("replayed", shared, HIT_TOKENS, 0)
    assert manager.allocate_new_blocks("replayed", HIT_TOKENS, HIT_TOKENS) == []
    assert manager.req_to_blocks["replayed"] == shared
    assert all(block.ref_cnt == 2 for block in shared)


def test_changed_scheduler_body_fails_before_installing_method():
    class UnsupportedScheduler:
        def schedule(self):
            return "unchanged"

    original = UnsupportedScheduler.schedule
    with pytest.raises(RuntimeError, match="unexpected UnsupportedScheduler.schedule body"):
        _rewrite_method(UnsupportedScheduler, "schedule", (("missing anchor", "replacement"),))
    assert UnsupportedScheduler.schedule is original


@pytest.fixture(params=[(False, False), (True, False), (True, True)], ids=["v1", "v2", "v2-async"])
def replay_scheduler(request):
    use_v2_model_runner, async_scheduling = request.param
    with (
        patch.object(ModelConfig, "__post_init__", lambda self, *args, **kwargs: None),
        patch.object(VllmConfig, "__post_init__", lambda self, *args, **kwargs: None),
        patch.object(VllmConfig, "use_v2_model_runner", new_callable=PropertyMock, return_value=use_v2_model_runner),
        patch.object(ModelConfig, "is_encoder_decoder", new_callable=PropertyMock, return_value=False),
        patch.object(ModelConfig, "uses_mrope", new_callable=PropertyMock, return_value=False),
        patch.object(ModelConfig, "is_diffusion", new_callable=PropertyMock, return_value=False),
        patch.object(ModelConfig, "is_multimodal_model", new_callable=PropertyMock, return_value=False),
    ):
        model = ModelConfig(model="test", tokenizer="test", max_model_len=256)
        model.hf_config = MagicMock()
        model.hf_config.architectures = ["DeepseekV41ForCausalLM"]
        model.hf_config.canvas_length = None
        model.multimodal_config = None
        model.pooler_config = None
        cache = CacheConfig(block_size=BLOCK_SIZE, enable_prefix_caching=True, swa_bounded_replay=True)
        cache.num_gpu_blocks = 128
        config = VllmConfig(
            model_config=model,
            cache_config=cache,
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(
                max_model_len=256,
                max_num_seqs=1,
                max_num_batched_tokens=16,
                enable_chunked_prefill=True,
                is_encoder_decoder=False,
                async_scheduling=async_scheduling,
                watermark=0.0,
            ),
        )
        register_all_kvcache_specs(None)
        register_ascend_kv_cache_specs()
        full = FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=128, dtype=torch.bfloat16)
        swa = AscendSlidingWindowMLASpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=1,
            head_size=128,
            dtype=torch.bfloat16,
            sliding_window=WINDOW,
            bounded_replay=True,
        )
        kv = KVCacheConfig(
            num_blocks=128,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(["full"], full),
                KVCacheGroupSpec(["swa"], swa),
            ],
        )
        scheduler_cls = AsyncScheduler if async_scheduling else Scheduler
        yield scheduler_cls(config, kv, MagicMock(), BLOCK_SIZE)


@pytest.mark.parametrize("hit", [16, 32, 96])
def test_actual_scheduler_replays_across_chunks_and_preserves_shared_blocks(replay_scheduler, hit):
    scheduler = replay_scheduler
    init_none_hash(sha256)
    hasher = get_request_block_hasher(BLOCK_SIZE, sha256)
    warm = Request("warm", list(range(hit + 1)), SamplingParams(max_tokens=1), None, block_hasher=hasher)
    manager = scheduler.kv_cache_manager
    assert manager.allocate_slots(warm, hit) is not None
    warm_blocks = manager.get_blocks("warm").blocks[0][:]
    manager.free(warm)
    req = Request("replayed", list(range(hit + 1)), SamplingParams(max_tokens=1), None, block_hasher=hasher)
    scheduler.add_request(req)
    output = scheduler.schedule()
    data = output.scheduled_new_reqs[0]
    if hit <= WINDOW:
        assert data.num_computed_tokens == 0
        assert data.replay_start == 0
    else:
        assert data.num_computed_tokens == hit - WINDOW
        assert data.replay_start == hit - WINDOW
        assert output.num_scheduled_tokens[req.request_id] == 16
        adopted = manager.get_blocks(req.request_id).blocks[0]
        assert adopted == warm_blocks
        assert all(b.ref_cnt == 1 for b in adopted)
        # The next chunk still ends inside the same cached prefix; no extra
        # cacheable blocks are allocated or overwritten during replay.
        next_output = scheduler.schedule()
        assert next_output.num_scheduled_tokens[req.request_id] == 16
        assert manager.get_blocks(req.request_id).blocks[0] == warm_blocks
        # Simulate drained preemption and verify that readmission carries a
        # fresh boundary through the V1 resumed-request payload.
        scheduler.running.remove(req)
        req.num_in_flight_tokens = 0
        scheduler._preempt_request(req, time.monotonic())
        resumed = scheduler.schedule()
        assert req.replay_start == hit - WINDOW
        if scheduler.use_v2_model_runner:
            assert resumed.scheduled_new_reqs[0].replay_start == hit - WINDOW
        else:
            assert resumed.scheduled_cached_reqs.replay_start == {req.request_id: hit - WINDOW}


@pytest.mark.parametrize("enabled,connector", [(True, False), (True, True), (False, False)])
def test_scheduler_created_before_patch_initializes_replay_on_first_schedule(replay_scheduler, enabled, connector):
    scheduler = replay_scheduler
    scheduler.cache_config.swa_bounded_replay = enabled
    scheduler.connector = MagicMock() if connector else None
    # Simulate a scheduler whose original constructor ran before config
    # revalidation installed the replay constructor wrapper.
    del scheduler.prefix_replay_tokens
    if connector:
        with pytest.raises(ValueError, match="get_loaded_kv_cache_group_ids"):
            scheduler.schedule()
        assert not hasattr(scheduler, "prefix_replay_tokens")
    else:
        scheduler.schedule()
        if enabled:
            assert scheduler.prefix_replay_tokens == WINDOW
        else:
            assert not hasattr(scheduler, "prefix_replay_tokens")


@pytest.mark.parametrize("connector_cls", [MooncakeConnector, MooncakeHybridConnector])
def test_pd_transfer_declares_private_swa_groups(connector_cls):
    connector = object.__new__(connector_cls)
    connector.connector_scheduler = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=["global", "swa", "compressor"])
    )
    assert connector.get_loaded_kv_cache_group_ids(MagicMock()) == (0, 1, 2)


def _attach_pd_connector(scheduler, hit, asynchronous=True, loaded_groups=None):
    """Use the real Hybrid connector contract; only replace network work."""
    connector = object.__new__(MooncakeHybridConnector)
    connector.connector_scheduler = SimpleNamespace(kv_cache_config=scheduler.kv_cache_manager.kv_cache_config)
    if loaded_groups is not None:
        connector.get_loaded_kv_cache_group_ids = lambda request: loaded_groups
    loads = []
    connector.get_num_new_matched_tokens = lambda request, local: (
        (max(0, hit - local), asynchronous) if not loads else (0, False)
    )
    connector.update_state_after_alloc = lambda request, blocks, external: loads.append(external) if external else None
    connector.on_new_request = MagicMock()
    connector.build_connector_meta = MagicMock(return_value=None)
    connector.has_pending_block_frees = lambda: False
    connector.has_pending_push_work = lambda: False
    scheduler.connector = connector
    _initialize_replay_window(scheduler)
    return connector, loads


def _pd_request(scheduler, length):
    init_none_hash(sha256)
    request = Request(
        "pd",
        list(range(length)),
        SamplingParams(max_tokens=4),
        None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )
    scheduler.add_request(request)
    return request


@pytest.mark.parametrize("hit", [16, 32, 95, 96])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_pd_full_window_is_loaded_without_128_token_replay(replay_scheduler, hit, asynchronous):
    scheduler = replay_scheduler
    _, loads = _attach_pd_connector(scheduler, hit, asynchronous)
    req = _pd_request(scheduler, hit + 1)
    first = scheduler.schedule()
    if asynchronous:
        assert first.total_num_scheduled_tokens == 0
        assert req.status == RequestStatus.WAITING_FOR_REMOTE_KVS
        assert req.num_computed_tokens == hit
    assert loads == [hit]
    blocks = scheduler.kv_cache_manager.get_blocks(req.request_id).blocks
    full_blocks = blocks[0][:]
    # The transferred SWA tail is private but physically allocated.
    assert any(not block.is_null for block in blocks[1])
    output = first
    if asynchronous:
        scheduler.finished_recving_kv_req_ids.add(req.request_id)
        output = scheduler.schedule()
    data = output.scheduled_new_reqs[0]
    assert data.num_computed_tokens == hit
    assert data.replay_start == 0
    assert output.num_scheduled_tokens[req.request_id] == 1
    assert scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0][: len(full_blocks)] == full_blocks


def test_async_pd_full_prompt_hit_only_recomputes_last_token(replay_scheduler):
    scheduler = replay_scheduler
    _attach_pd_connector(scheduler, HIT_TOKENS)
    req = _pd_request(scheduler, HIT_TOKENS)
    scheduler.schedule()
    scheduler.finished_recving_kv_req_ids.add(req.request_id)
    output = scheduler.schedule()
    data = output.scheduled_new_reqs[0]
    assert data.num_computed_tokens == HIT_TOKENS - 1
    assert data.replay_start == 0
    assert output.num_scheduled_tokens[req.request_id] == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("hit", [95, 96])
def test_external_hit_without_window_replays_only_after_load(replay_scheduler, asynchronous, hit):
    scheduler = replay_scheduler
    _, loads = _attach_pd_connector(scheduler, hit, asynchronous, loaded_groups=(0,))
    req = _pd_request(scheduler, hit + 1)
    output = scheduler.schedule()
    adopted = hit - hit % BLOCK_SIZE
    assert loads == [adopted]
    full_blocks = scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0][:]
    if asynchronous:
        assert output.total_num_scheduled_tokens == 0
        assert req.replay_start == 0
        scheduler.finished_recving_kv_req_ids.add(req.request_id)
        output = scheduler.schedule()
    data = output.scheduled_new_reqs[0]
    assert data.num_computed_tokens == adopted - WINDOW
    assert data.replay_start == adopted - WINDOW
    assert output.num_scheduled_tokens[req.request_id] == BLOCK_SIZE
    assert scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0] == full_blocks
    # Another replay chunk must not allocate/overwrite the shared global hit.
    output = scheduler.schedule()
    assert output.num_scheduled_tokens[req.request_id] == BLOCK_SIZE
    assert scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[0] == full_blocks
    swa_blocks = scheduler.kv_cache_manager.get_blocks(req.request_id).blocks[1]
    assert not swa_blocks[(adopted - WINDOW) // BLOCK_SIZE].is_null


@pytest.mark.parametrize("hit", [15, 16, 32])
def test_short_external_hit_without_window_recomputes_cold(replay_scheduler, hit):
    scheduler = replay_scheduler
    _, loads = _attach_pd_connector(scheduler, hit, loaded_groups=(0,))
    req = _pd_request(scheduler, hit + 1)
    output = scheduler.schedule()
    assert loads == []
    assert output.scheduled_new_reqs[0].num_computed_tokens == 0
    assert req.replay_start == 0
    assert req.status == RequestStatus.RUNNING


@pytest.mark.parametrize("valid_tokens", [0, 16, 32, 64])
def test_failed_async_pd_load_does_not_trust_received_window(replay_scheduler, valid_tokens):
    scheduler = replay_scheduler
    _attach_pd_connector(scheduler, HIT_TOKENS)
    req = _pd_request(scheduler, HIT_TOKENS + 1)
    scheduler.schedule()
    # The upstream invalid-block handler has lowered the valid prefix.
    req.num_computed_tokens = valid_tokens
    scheduler.failed_recving_kv_req_ids.add(req.request_id)
    scheduler.finished_recving_kv_req_ids.add(req.request_id)
    output = scheduler.schedule()
    data = output.scheduled_new_reqs[0]
    expected = max(0, valid_tokens - WINDOW)
    assert data.num_computed_tokens == expected
    assert data.replay_start == expected
    assert req.request_id not in scheduler.failed_recving_kv_req_ids
    assert req.request_id not in scheduler.finished_recving_kv_req_ids


def test_pd_preemption_rebuilds_window_from_local_global_hit(replay_scheduler):
    scheduler = replay_scheduler
    _attach_pd_connector(scheduler, HIT_TOKENS)
    req = _pd_request(scheduler, HIT_TOKENS + 1)
    scheduler.schedule()
    scheduler.finished_recving_kv_req_ids.add(req.request_id)
    scheduler.schedule()
    assert req.replay_start == 0
    scheduler.running.remove(req)
    req.num_in_flight_tokens = 0
    scheduler._preempt_request(req, time.monotonic())
    output = scheduler.schedule()
    assert req.replay_start == HIT_TOKENS - WINDOW
    if scheduler.use_v2_model_runner:
        assert output.scheduled_new_reqs[0].replay_start == HIT_TOKENS - WINDOW
    else:
        assert output.scheduled_cached_reqs.replay_start == {req.request_id: HIT_TOKENS - WINDOW}
