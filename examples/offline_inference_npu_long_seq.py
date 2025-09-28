import os
import time
import argparse

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_len', type=int, default=1024)
    parser.add_argument('--output_len', type=int, default=128)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="deepseek-ai/DeepSeek-V2-Lite")
    parser.add_argument('--tp', type=int, default=2)
    parser.add_argument('--cp', type=int, default=2)
    parser.add_argument('--iter_times', type=int, default=1)

    args = parser.parse_args()

    prompts = [
        "The capital of France is",
        "Hello, my name is Tom, I am",
        "The president of United States is",
        "AI future is? What do you think about it? Can you give me some information or any thing you want?"
    ]

    sampling_params = SamplingParams(temperature = 0.8, top_p = 0.95, max_tokens=args.output_len)
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        context_parallel_size=args.cp,
        enable_prefix_caching=False,
        enable_expert_parallel=True,
        enable_chunked_prefill=False,
        enable_sequence_parallel=True,
        max_num_batched_tokens=args.input_len // args.cp // args.tp + 138,
        max_model_len=args.input_len + args.output_len + 138,
        additional_config={"ascend_scheduler_config": {"enabled": True}},
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
