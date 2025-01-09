from vllm import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=100, temperature=0.0)
# Create an LLM.
# TODO (cmq): ray is not supported currently, need some fixes
llm = LLM(
    model="facebook/opt-125m",
    tensor_parallel_size=2,
    distributed_executor_backend="mp",
    trust_remote_code=True,
)

# Generate texts from the prompts.
outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
