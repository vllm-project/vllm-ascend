
import os

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from vllm import LLM, SamplingParams


def main():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create a sampling params object.
    sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
    # Create an LLM.
    llm = LLM(
        model="deepseek-ai/DeepSeek-V2-Lite",
        tensor_parallel_size=8,
        decode_context_parallel_size=2,
        enforce_eager=True,
        speculative_config={"method": "eagle3", "num_speculative_tokens": 3, "model": "deepseek-ai/DeepSeek-V2-Lite"},
        trust_remote_code=True,
        max_model_len=1024,
    )

    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    main()
