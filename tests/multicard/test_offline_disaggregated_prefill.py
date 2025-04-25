"""Test the offline disaggregated prefill and decode with AscendHcclConnector.
This test is designed to run on a 4+4 Ascend cluster. The prefill node will
run on the first 4 Ascend devices, and the decode node will run on the last 4
Ascend devices. The prefill node will generate the kv-cache and transfer it to
the decode node. The decode node will then use the kv-cache to generate the
outputs.

Run `pytest tests/test_offline_disaggregated_prefill.py`.
"""
import gc
import multiprocessing as mp
import os
import time
from multiprocessing import Event, Process
from typing import List

import pytest
import torch
import vllm  # noqa: F401
from vllm import SamplingParams
from vllm.config import KVTransferConfig
from vllm.distributed.parallel_state import (destroy_distributed_environment,
                                             destroy_model_parallel)

from tests.conftest import VllmRunner

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "max_split_size_mb:256"


def clean_up():
    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


def run_prefill(prefill_done, process_close, prompts: List[str], model: str):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"
    os.environ["PROMPT_DEVICE_ID"] = "0,1"
    os.environ["DECODE_DEVICE_ID"] = "2,3"

    sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=1)
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"AscendHcclConnector","kv_buffer_device":"npu","kv_role":"kv_producer","kv_rank":0,"kv_parallel_size":2}'
    )
    with VllmRunner(
            model_name=model,
            trust_remote_code=True,
            kv_transfer_config=ktc,
            max_model_len=2000,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as llm:
        for _ in range(5):
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(
                    f"[Prefill] Prompt: {prompt!r}, Generated text: {generated_text!r}"
                )
            print("[Prefill] Done.")
            prefill_done.set()

    # To keep the prefill node running in case the decode node is not done
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    try:
        while not process_close.is_set():
            time.sleep(1)
    finally:
        print("Cleanup prefill resources")
        clean_up()


def run_decode(prefill_done, prompts: List[str], model: str):
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "2,3"
    os.environ["PROMPT_DEVICE_ID"] = "0,1"
    os.environ["DECODE_DEVICE_ID"] = "2,3"

    sampling_params = SamplingParams(temperature=0, top_p=0.9)
    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"AscendHcclConnector","kv_buffer_device":"npu","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":2}'
    )
    with VllmRunner(
            model_name=model,
            trust_remote_code=True,
            kv_transfer_config=ktc,
            max_model_len=2000,
            gpu_memory_utilization=0.8,
            tensor_parallel_size=2,
            distributed_executor_backend="mp",
    ) as llm:
        for _ in range(5):
            # Wait for the producer to start the consumer
            print("Waiting for prefill node to finish...")
            prefill_done.wait()

            # At this point when the prefill_done is set, the kv-cache should
            # have been transferred to this decode node, so we can start
            # decoding.
            outputs = llm.generate(prompts, sampling_params)
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(
                    f"[Decode] Prompt: {prompt!r}, Generated text: {generated_text!r}"
                )
            print("[Decode] Done.")

    clean_up()


@pytest.mark.parametrize("model", [
    ("Qwen/QwQ-32B"),
    ("deepseek-ai/DeepSeek-V2-Lite"),
])
def test_models_distributed(model: str) -> None:
    prompts = [
        "Hello, how are you today?",
        "Hi, what is your name?",
        "Tell me a very long story.",
        "what is your favourite book?",
    ]

    mp.get_context("spawn")
    prefill_done = Event()
    process_close = Event()
    prefill_process = Process(target=run_prefill,
                              args=(prefill_done, process_close, prompts,
                                    model))
    decode_process = Process(target=run_decode,
                             args=(prefill_done, prompts, model))

    # Start prefill node
    prefill_process.start()
    # Start decode node
    decode_process.start()
    # Terminate the prefill node when decode is finished
    decode_process.join()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
