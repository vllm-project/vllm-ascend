    ### 3.3 Run vLLM

{% include "getting_started/quick_start/model_download.inc.md" %}

    === "Offline batch inference"

        Run the following command in the container terminal:

        ```bash
        vi example.py
        ```

        In `vi`, press `i` to enter insert mode and paste the following code. After pasting, press `Esc`, type `:wq`, and then press Enter to save and exit.

        If the image includes an editor that you are more familiar with, you can also create the same file by using a command such as `nano example.py`.

        ```python
        from vllm import LLM, SamplingParams

        prompts = [
            "Hello, my name is",
            "The future of AI is",
        ]

        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=32,
        )

        llm = LLM(model="Qwen/Qwen3-0.6B")
        outputs = llm.generate(prompts, sampling_params)

        assert len(outputs) == len(prompts)

        for output in outputs:
            generated_text = output.outputs[0].text
            assert generated_text.strip()
            print(f"Prompt: {output.prompt!r}")
            print(f"Generated text: {generated_text!r}")
            print()
        ```

        Run the example:

        ```bash
        python3 example.py
        ```

    === "Online serving"

        ```bash
        vllm serve Qwen/Qwen3-0.6B \
            --host 0.0.0.0 \
            --port 8000 \
            --served-model-name quickstart-model
        ```

        Verify the service from another terminal on the host:

        ```bash
        curl --fail --silent --show-error \
            http://localhost:8000/v1/models \
            | python3 -m json.tool
        ```

        ```bash
        curl --fail --silent --show-error \
            http://localhost:8000/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "quickstart-model",
                "messages": [
                    {
                        "role": "user",
                        "content": "Introduce vLLM in one sentence."
                    }
                ],
                "max_tokens": 32,
                "temperature": 0
            }' \
            | python3 -m json.tool
        ```
