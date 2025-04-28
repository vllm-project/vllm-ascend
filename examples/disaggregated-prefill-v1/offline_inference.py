"""
This file demonstrates the example usage of disaggregated prefilling We will
launch 2 vllm instances (NPU 0,1,3,4 for prefill and NPU 5,6,7,8 for decode),
and then transfer the KV cache between them.
"""

import multiprocessing as mp
import os
from multiprocessing import Event, Process, Queue
from typing import List, Literal


def get_kv_transfer_config(role: Literal["kv_producer", "kv_consumer"],
                           local_server_id: str):
    kv_rank = 0 if role == "kv_producer" else 1
    return f"""{{
        "kv_connector": "AscendHcclConnectorV1",
        "kv_buffer_device": "npu",
        "kv_role": "{role}",
        "kv_rank": {kv_rank},
        "kv_parallel_size": 2
    }}"""


def clean_up():
    import gc

    import torch
    from vllm.distributed.parallel_state import (
        destroy_distributed_environment, destroy_model_parallel)

    destroy_model_parallel()
    destroy_distributed_environment()
    gc.collect()
    torch.npu.empty_cache()


def run_prefill(
    prefill_done,
    process_close,
    prompt_q: Queue,
    prompts: List[str],
    model: str,
    local_server_id: str,
    visible_devices: str,
):
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    tensor_parallel_size = len(visible_devices.split(","))

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    ktc = KVTransferConfig.from_cli(
        get_kv_transfer_config(
            role="kv_producer",
            local_server_id=local_server_id,
        ))

    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_transfer_config=ktc,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=40,
    )

    result = llm.generate(prompts, sampling_params)
    for output in result:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"[Prefill] Prompt: {prompt!r}, Generated text: {generated_text!r}"
        )
        prompt_q.put(prompt + generated_text)
    prompt_q.close()

    print("[Prefill] DONE.")
    prefill_done.set()

    # To keep the prefill node running in case the decode node is not done;
    # otherwise, the script might exit prematurely, causing incomplete decoding.
    process_close.wait()

    del llm
    clean_up()


def run_decode(
    prefill_done,
    prompt_q: Queue,
    num_prompts: int,
    model: str,
    local_server_id: str,
    visible_devices: str,
):
    os.environ["VLLM_USE_V1"] = "1"
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    tensor_parallel_size = len(visible_devices.split(","))

    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    sampling_params = SamplingParams(temperature=0, top_p=0.95)

    ktc = KVTransferConfig.from_cli(
        get_kv_transfer_config(
            role="kv_consumer",
            local_server_id=local_server_id,
        ))

    llm = LLM(
        model=model,
        trust_remote_code=True,
        enforce_eager=True,
        enable_prefix_caching=False,
        kv_transfer_config=ktc,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=40,
    )

    # Wait for the producer to start the comsumer
    print("[Decode] Waiting for prefill node to finish...")
    prefill_done.wait()

    # Get the prompts from the queue
    prompts = []
    for _ in range(num_prompts):
        prompts.append(prompt_q.get())

    # At this point when the prefill_done is set, the kv-cache should have been
    # transferred to this decode node, so we can start decoding.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(
            f"[Decode] Prompt: {prompt!r}, Generated text: {generated_text!r}")
    print("[Decode] DONE.")

    # Must delete the llm instance, otherwise the process will not exit
    del llm
    clean_up()


if __name__ == "__main__":
    mp.get_context("spawn")

    model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    # Set the server id and device ids for prefill and decode nodes
    prompt_server_id = "server-0"
    prompt_deivce_ids = "0,1,2,3"
    decode_server_id = "server-1"
    decode_device_ids = "4,5,6,7"

    prompts = [
        "Hello, how are you today?",
        "Hi, what is your name?",
        "Tell me a very long story.",
        "what is your favourite book?",
    ]
    num_prompts = len(prompts)

    prompt_q: Queue = Queue(num_prompts)
    prefill_done = Event()
    process_close = Event()

    prefill_process = Process(
        target=run_prefill,
        args=(prefill_done, process_close, prompt_q, prompts, model,
              prompt_server_id, prompt_deivce_ids),
    )
    decode_process = Process(
        target=run_decode,
        args=(prefill_done, prompt_q, num_prompts, model, decode_server_id,
              decode_device_ids),
    )

    # Start prefill node
    prefill_process.start()
    # Start decode node
    decode_process.start()

    # Wait for decode process to finish
    decode_process.join()
    print("[Main] Decode process done.")

    # Terminate the prefill node, and wait for it to finish
    process_close.set()
    prefill_process.join()
    print("[Main] Prefill process done.")
