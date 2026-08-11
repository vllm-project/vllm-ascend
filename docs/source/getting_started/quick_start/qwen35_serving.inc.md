    ### 3.3 Run vLLM

{% include "getting_started/quick_start/model_download.inc.md" %}

    === "Offline batch inference"

        !!! info "Currently unsupported"

            This Quickstart does not currently provide a standalone offline inference example for Atlas 300I DUO or Atlas 200I Pro. When support is added, commands will be added directly to this tab without changing the page structure.

    === "Online serving"

        ```bash
        vllm serve Qwen/Qwen3.5-2B \
            --host 0.0.0.0 \
            --port 8000 \
            --tensor-parallel-size 1 \
            --served-model-name quickstart-model \
            --max-num-seqs 32 \
            --max-model-len 16384 \
            --trust-remote-code \
            --gpu-memory-utilization 0.90 \
            --mamba-ssm-cache-dtype float16 \
            --dtype float16 \
            --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":1}' \
            --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}' \
            --additional-config '{"ascend_compilation_config":{"enable_npugraph_ex":false}}'
        ```

        Verify the service from another terminal on the host:

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
