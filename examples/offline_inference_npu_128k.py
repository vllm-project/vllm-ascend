import os
import time
import argparse
import random

from vllm import LLM, SamplingParams
from datasets import load_dataset, Features, Value, Sequence
from transformers import AutoTokenizer

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_USE_V1"] = "1"

def generate_prompts_128K(model_path):
    # Define the features schema
    ft = Features({
        "id": Value("int64"),
        "context": Value("string"),
        "input": Value("string"),
        "answer": Sequence(Value("string")),
        "options": Sequence(Value("string"))
    })
    # 100k datasets: https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench
    dataset_dict = load_dataset("/home/l00889328/vllm091/datasets/InfiniteBench", features=ft)
    dataset = dataset_dict["train"]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_ids = []

    prompt = str(dataset['context'][0]) + '\n' + dataset['input'][0]
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=100*1024,
        return_tensors="pt"
    )

    token_ids.extend(encoded["input_ids"].squeeze(0).tolist())

    prompt = str(dataset['context'][1]) + '\n' + dataset['input'][1]
    encoded = tokenizer(
        prompt,
        truncation=True,
        max_length=28*1024,
        return_tensors="pt"
    )
    token_ids.extend(encoded["input_ids"].squeeze(0).tolist())
    token_ids_text = tokenizer.decode(token_ids)
    return token_ids_text


def generate_prompt_token_ids(input_len, batchsize):
    token_ids = [[random.randint(1,128000) for _ in range(input_len)] for _ in range(batchsize)]
    return token_ids


# Performance testing function
def run_performance(args):
    """Run performance tests and return timing results."""

    sampling_params = SamplingParams(temperature = 0.8, top_p = 0.95, ignore_eos=True, max_tokens=args.output_len)

    prompt_token = generate_prompts_128K(args.model_path)

    # Create an LLM
    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        enforce_eager=True,
        tensor_parallel_size=args.tp,
        data_parallel_size=args.dp,
        context_parallel_size=args.cp,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enable_sequence_parallel=True,
        enable_expert_parallel=True,
        max_num_batched_tokens=args.input_len // args.cp // args.tp + 138,
        max_model_len=args.input_len + 138,
        quantization="ascend",
        additional_config={"ascend_scheduler_config": {"enabled": True}},
        max_num_seqs=1,
        block_size=128,
        gpu_memory_utilization=0.85
    )

    print("========================= Warmup =========================")
    t0 = time.time()
    llm.generate(prompts=prompt_token, sampling_params=sampling_params)
    t1 = time.time()
    dt0 = t1 - t0
    print(f"E2E: {dt0} s")
    print("============================= Warmup finished. ============================")

    # Second run for comparison
    print("========================= Infer ===========================")
    t2 = time.time()
    for _ in range(args.iter_times):
        llm.generate(prompts=prompt_token, sampling_params=sampling_params)
    t3 = time.time()

    # Give engines time to pause their processing loops before exiting.
    time.sleep(1)
    dt1 = t3 - t2
    print(f"E2E: {dt1} s")
    print("============================= Infer finished. ============================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_len', type=int, default=128*1024)
    # current output_len only suppot 1 for long_seq prefill stage
    parser.add_argument('--output_len', type=int, default=1)
    parser.add_argument('--bs', type=int, default=1)
    parser.add_argument('--model_path', type=str, default="/mnt/nfs/weight/deepseek_diff/deepseek_r1_w8a8_vllm_lzy")
    parser.add_argument('--tp', type=int, default=8)
    parser.add_argument('--cp', type=int, default=2)
    parser.add_argument('--dp', type=int, default=1)
    parser.add_argument('--iter_times', type=int, default=1)

    args = parser.parse_args()
    # Run performance test using our new function
    run_performance(args)
