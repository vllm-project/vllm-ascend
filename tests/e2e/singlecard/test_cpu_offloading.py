# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import socket
import time

import msgspec
import msgspec.msgpack
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


def _latency_test(llm: LLM, subscriber: MockSubscriber):
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
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cold_time = time.time() - start_time
        total_cold_time += cold_time

        # run generation again - should hit the GPU prefix cache
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        gpu_hit_time = time.time() - start_time
        total_gpu_hit_time += gpu_hit_time

        # reset prefix cache to avoid GPU hit.
        llm.reset_prefix_cache()

        assert subscriber.get_new_cpu_stored_events()

        # run generation again - this should trigger loading from CPU
        start_time = time.time()
        llm.generate(prompts, sampling_params, use_tqdm=False)
        cpu_hit_time = time.time() - start_time
        total_cpu_hit_time += cpu_hit_time

        if cpu_hit_time < cold_time:
            num_times_cpu_better_than_cold += 1

    print("Average times:")
    print(f"    Cold: {total_cold_time * 1000 / num_tests:.2f}ms")
    print(f"    GPU hit: {total_gpu_hit_time * 1000 / num_tests:.2f}ms")
    print(f"    CPU hit: {total_cpu_hit_time * 1000 / num_tests:.2f}ms")

    assert num_times_cpu_better_than_cold >= 0.8 * num_tests


def _accuracy_test(llm: LLM, subscriber: MockSubscriber):
    sampling_params = SamplingParams(max_tokens=1)
    cpu_block_size = (llm.llm_engine.vllm_config.kv_transfer_config.
                      kv_connector_extra_config["block_size"])

    subscriber.get_new_cpu_stored_events()

    # prepend prompt to be cpu block aligned
    prompt = "Let's count to 10. One, two, three, four,"
    while (len(llm.generate(prompt, use_tqdm=False)[0].prompt_token_ids) %
           cpu_block_size != 0):
        prompt = ". " + prompt

    assert subscriber.get_new_cpu_stored_events()

    test_count = 100
    success_count = 0
    for i in range(test_count):
        if (llm.generate(prompt, sampling_params,
                         use_tqdm=False)[0].outputs[0].text == " five"):
            success_count += 1

    assert success_count >= 0.5 * test_count


def test_cpu_offloading() -> None:
    """
    Tests OffloadingConnector with CPUOffloadingSpec.
    """

    # configure OffloadingConnector (spec_name=CPUOffloadingSpec by default)
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "num_cpu_blocks": 1000,
            "block_size": 128,
            "spec_name": "NPUOffloadingSpec",
            "spec_module_path": "vllm_ascend.kv_offload.npu"
        },
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
        _latency_test(llm, subscriber)
        _accuracy_test(llm, subscriber)
    finally:
        subscriber.close()
        del llm
