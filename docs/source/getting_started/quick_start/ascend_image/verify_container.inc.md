#### Verify the container environment

Run the following commands in the container. The container is ready when the output includes `vLLM Ascend environment: OK`.

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
