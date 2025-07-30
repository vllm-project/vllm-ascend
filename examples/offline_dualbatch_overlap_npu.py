import os
import time

from vllm import LLM, SamplingParams

os.environ["VLLM_USE_MODELSCOPE"] = "True"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
# enable dual-batch overlap for vllm ascend
os.environ["VLLM_ASCEND_ENABLE_DBO"] = "1"

# Sample prompts.
prompts = ["The president of the United States is"] * 41
# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)


def main():
    # Create an LLM.
    llm = LLM(model="deepseek-ai/DeepSeek-V3-Lite-base-latest-w8a8-dynamic",
              enforce_eager=True,
              tensor_parallel_size=2,
              max_model_len=4096,
              trust_remote_code=True,
              enable_expert_parallel=True,
              additional_config={
                  "torchair_graph_config": {
                      "enabled": False
                  },
                  "ascend_scheduler_config": {
                      "enabled": True
                  },
              })

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)

    # Print the outputs.
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)

    # Add a buffer to wait for profiler in the background process
    # (in case MP is on) to finish writing profiling output.
    time.sleep(10)


if __name__ == "__main__":
    main()
