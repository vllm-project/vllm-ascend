    ### 3.2 Verify the Container Environment

    Run the following commands in the container:

    ```bash
    npu-smi info

    python3 - <<'PY'
    import torch
    import vllm
    import vllm_ascend

    assert torch.npu.is_available(), "No available Ascend NPU detected in the container"
    print("vLLM Ascend environment: OK")
    PY
    ```
