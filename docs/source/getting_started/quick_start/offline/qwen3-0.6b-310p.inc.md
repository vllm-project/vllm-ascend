The following Qwen3-0.6B example has been validated on Atlas 300I DUO and Atlas 200I Pro.

???+ important "Runtime requirements for this hardware path"

    - Use `float16`.
    - Use `FULL_DECODE_ONLY` and limit the graph capture sizes.
    - `enable_npugraph_ex` is not supported. Set `--additional-config '{"ascend_compilation_config": {"enable_npugraph_ex":false}}'`.

In the container terminal, create `example.py` with the following code:

<!-- doctest: quickstart-300i-duo-offline -->
```python
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-0.6B"

prompts = [
    "Hello, my name is",
    "The future of AI is",
]

llm = LLM(
    model=MODEL,
    dtype="float16",
    max_num_seqs=8,
    compilation_config={
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4, 8],
    },
    additional_config={
        "ascend_compilation_config": {
            "enable_npugraph_ex": False,
        },
    },
)

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
)

outputs = llm.generate(prompts, sampling_params)

assert len(outputs) == len(prompts)

for output in outputs:
    generated_text = output.outputs[0].text
    assert generated_text.strip()
    print(f"Prompt: {output.prompt!r}, Generated text: {generated_text!r}")
```

Run the example:

<!-- doctest: quickstart-300i-duo-offline-run -->
```bash
python3 example.py
```

The following output shows that vLLM has successfully detected the Ascend platform:

```text
INFO 05-27 11:40:38 [__init__.py:44] Available plugins for group vllm.platform_plugins:
INFO 05-27 11:40:38 [__init__.py:46] - ascend -> vllm_ascend:register
INFO 05-27 11:40:38 [__init__.py:49] All plugins in this group will be loaded. Set `VLLM_PLUGINS` to control which plugins to load.
INFO 05-27 11:40:38 [__init__.py:238] Platform plugin ascend is activated
```

The following output shows the generated results:

```text
Prompt: 'Hello, my name is', Generated text: ' Lucy and I am an 8 year old who loves to draw and write stories'
Prompt: 'The future of AI is', Generated text: ' a topic that is being discussed in various contexts. In the business world, AI'
```

The following messages show the process exiting after offline inference and do not affect the inference results:

```text
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1201] Shutdown initiated (timeout=0)
(EngineCore pid=970) INFO 05-12 11:36:00 [core.py:1224] Shutdown complete
ERROR 05-12 11:36:01 [core_client.py:704] Engine core proc EngineCore died unexpectedly, shutting down client.
sys:1: DeprecationWarning: builtin type swigvarlink has no __module__ attribute
```
