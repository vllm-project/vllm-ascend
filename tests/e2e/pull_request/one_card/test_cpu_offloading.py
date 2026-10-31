# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import time
from typing import Any

import msgspec
import msgspec.msgpack
import pytest
import zmq
from vllm import LLM, SamplingParams, TokensPrompt
from vllm.config import KVEventsConfig, KVTransferConfig
from vllm.distributed.kv_events import BlockStored, KVEventBatch


class MockSubscriber:
    """Helper class to receive and verify published events"""

    def __init__(
        self,
        endpoint: str,
        topic: str,
    ):
        self.ctx = zmq.Context.instance()  # type: ignore
        self.topic_bytes = topic.encode("utf-8")

        # Set up subscriber socket
        self.sub = self.ctx.socket(zmq.SUB)  # type: ignore
        self.sub.setsockopt(zmq.SUBSCRIBE, self.topic_bytes)  # type: ignore
        self.sub.connect(endpoint)

        self.decoder = msgspec.msgpack.Decoder(type=KVEventBatch)

    def get_new_cpu_stored_events(self) -> list[BlockStored]:
        cpu_stored_events: list[BlockStored] = []

        poller = zmq.Poller()  # type: ignore
        poller.register(self.sub, zmq.POLLIN)  # type: ignore

        timeout = 1000  # 1 second
        while True:
            events = dict(poller.poll(timeout))

            if events.get(self.sub) != zmq.POLLIN:  # type: ignore
                return cpu_stored_events

            topic_bytes, _, payload = self.sub.recv_multipart()

            assert topic_bytes == self.topic_bytes

            event_batch = self.decoder.decode(payload)
            assert isinstance(event_batch, KVEventBatch)
            for event in event_batch.events:
                if isinstance(event, BlockStored) and event.medium == "CPU":
                    cpu_stored_events.append(event)
                    timeout = 100

    def close(self):
        """Clean up resources"""
        self.sub.close()


def _latency_test(
    llm: LLM,
    subscriber: MockSubscriber,
    require_cpu_speedup: bool,
) -> None:
    sampling_params = SamplingParams(max_tokens=1)

    num_times_cpu_better_than_cold = 0
    num_tests = 10
    total_cold_time = 0.0
    total_gpu_hit_time = 0.0
    total_cpu_hit_time = 0.0
    prompt_token_ids = [0] * 10001
    for i in range(num_tests):
        prompt_token_ids[0] = i
        prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

        # run generation - this should trigger saving KV cache
        start_time = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.perf_counter() - start_time
        total_cold_time += cold_time

        # run generation again - should hit the GPU prefix cache
        start_time = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        gpu_hit_time = time.perf_counter() - start_time
        total_gpu_hit_time += gpu_hit_time

        # reset prefix cache to avoid GPU hit.
        llm.reset_prefix_cache()

        assert subscriber.get_new_cpu_stored_events()

        # run generation again - this should trigger loading from CPU
        start_time = time.perf_counter()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cpu_hit_time = time.perf_counter() - start_time
        total_cpu_hit_time += cpu_hit_time

        if cpu_hit_time < cold_time:
            num_times_cpu_better_than_cold += 1

    print("Average times:")
    print(f"    Cold: {total_cold_time * 1000 / num_tests:.2f}ms")
    print(f"    GPU hit: {total_gpu_hit_time * 1000 / num_tests:.2f}ms")
    print(f"    CPU hit: {total_cpu_hit_time * 1000 / num_tests:.2f}ms")

    if require_cpu_speedup:
        assert num_times_cpu_better_than_cold >= 0.8 * num_tests


def _accuracy_test(llm: LLM, subscriber: MockSubscriber) -> None:
    sampling_params = SamplingParams(max_tokens=5, temperature=0)
    vllm_config = llm.llm_engine.vllm_config
    extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
    cpu_block_size = extra_config.get("block_size")
    if cpu_block_size is None:
        cpu_block_size = extra_config["blocks_per_chunk"] * vllm_config.cache_config.block_size

    subscriber.get_new_cpu_stored_events()

    # Build a prompt containing one complete offload chunk plus a tail token.
    # vLLM keeps the final prompt token for computation, so this shape ensures
    # that the second generation restores a complete chunk from CPU.
    tokenizer = llm.get_tokenizer()
    prompt_token_ids = list(tokenizer.encode("Let's count to 10. One, two, three, four,"))
    num_padding_tokens = (1 - len(prompt_token_ids)) % cpu_block_size
    prompt_token_ids = [0] * num_padding_tokens + prompt_token_ids
    prompts = [TokensPrompt(prompt_token_ids=prompt_token_ids)]

    cold_output = llm.generate(prompts, sampling_params, use_tqdm=False)[0]
    assert len(cold_output.prompt_token_ids) % cpu_block_size == 1
    assert subscriber.get_new_cpu_stored_events()

    llm.reset_prefix_cache()
    cpu_output = llm.generate(prompts, sampling_params, use_tqdm=False)[0]

    assert cpu_output.outputs[0].token_ids == cold_output.outputs[0].token_ids


@pytest.mark.parametrize("enable_tiering", [False, True])
def test_cpu_offloading(tmp_path, enable_tiering: bool) -> None:
    """
    Tests the native CPU-only and multi-tier offloading specs.
    """

    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    extra_config: dict[str, Any] = {
        # Keep CI host-memory and pinned-memory pressure bounded. The original
        # CPU-only test already exercised eviction with a 1 GiB tier, which is
        # also sufficient for validating the filesystem tiering path.
        "cpu_bytes_to_use": 1 << 30,
        # Match the established CPU-offloading workload. With the default
        # 16-token GPU blocks this batches eight blocks into each CPU chunk,
        # reducing scheduler and tier-manager overhead for long prompts.
        "block_size": 128,
        "spec_name": ("TieringOffloadingSpec" if enable_tiering else "CPUOffloadingSpec"),
    }
    if enable_tiering:
        extra_config["secondary_tiers"] = [
            {
                "type": "fs",
                "root_dir": str(tmp_path / "native_kv_offload"),
            }
        ]

    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config,
    )

    port: int
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        port = s.getsockname()[1]

    events_endpoint = f"tcp://*:{port}"
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=events_endpoint,
        topic="test",
    )

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        gpu_memory_utilization=0.5,
        kv_events_config=kv_events_config,
        kv_transfer_config=kv_transfer_config,
    )

    events_endpoint = events_endpoint.replace("*", "127.0.0.1")
    subscriber = MockSubscriber(events_endpoint, topic=kv_events_config.topic)

    try:
        # Tiering must use a shared mmap so the scheduler can move chunks to
        # secondary tiers. Ascend cannot pin an arbitrary mmap, so only the
        # CPU-only spec has the pinned-memory performance guarantee measured
        # by this assertion. Both specs still execute the same load workload.
        _latency_test(llm, subscriber, require_cpu_speedup=not enable_tiering)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm
