import os
import time
import argparse

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_ASCEND_ENABLE_CONTEXT_PARALLEL"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_len', type=int, default=1024)
    parser.add_argument('--output_len', type=int, default=128)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--pcp', type=int, default=2)
    parser.add_argument('--dcp', type=int, default=1)
    parser.add_argument('--iter_times', type=int, default=1)

    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is"
    ]

    sampling_params = SamplingParams(temperature = 0.8, top_p = 0.95, max_tokens=args.output_len)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        prefill_context_parallel_size=args.pcp,
        decode_context_parallel_size=args.dcp,
        enable_prefix_caching=False,
        enable_expert_parallel=True,
        enable_chunked_prefill=False,
        max_num_batched_tokens=2048,
        max_model_len=1024,
        max_num_seqs=1,
        block_size=128,
        gpu_memory_utilization=0.9
    )

    t0 = time.time()
    for _ in range(args.iter_times):
        outputs = llm.generate(prompts, sampling_params)
    t1 = time.time()
    print(f"TTFT: {(t1 - t0) * 1000 / (args.iter_times * args.bs)} ms")

    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"req_num: {i}\nGenerated text: {generated_text!r}")