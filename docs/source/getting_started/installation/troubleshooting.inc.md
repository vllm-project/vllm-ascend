### 5.3 Troubleshooting {: #installation-troubleshooting }

#### 5.3.1 `npu-smi` Works, but Python Cannot Detect the NPU

Check:

```bash
python - <<'PY'
import torch
import torch_npu

print("has torch.npu:", hasattr(torch, "npu"))
print("npu available:", torch.npu.is_available())
PY
```

Also confirm that you have loaded the environment:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
```

#### 5.3.2 `libatb.so` Cannot Be Found

This usually means that the NNAL environment is not installed or has not been loaded.

Confirm:

```bash
ls -l /usr/local/Ascend/nnal/atb/
```

Then load it:

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh
```

#### 5.3.3 The Device Is Not Visible in Docker

First, check the host:

```bash
npu-smi info
ls -l /dev/davinci*
```

Then verify that:

- `docker run --device` passes the correct device nodes;
- the required driver libraries are mounted;
- Atlas A3 uses the complete dual-DIE device pair;
- Atlas 200I Pro uses the additional product-specific devices and mounts.

#### 5.3.4 `docker pull` Fails

The Docker daemon and the current shell use separate network configurations.

Check the following first:

- daemon proxy;
- registry mirror;
- DNS;
- organization network policy.

#### 5.3.5 Device Type Mismatch After pip / Wheel Installation

Check:

```bash
python - <<'PY'
from vllm_ascend._build_info import __device_type__

print(__device_type__)
PY
```

If the wheel does not match the target hardware:

- switch to WheelNext;
- use a pre-built image for the hardware;
- or build from source for the target hardware.

#### 5.3.6 `pip check` Reports Dependency Conflicts

First distinguish between:

- core software conflicts;
- optional feature dependencies;
- known metadata conflicts in the current release.

Do not ignore additional conflicts involving any of the following components:

```text
vllm
vllm-ascend
torch
torch-npu
triton
triton-ascend
```

To avoid wheel metadata issues, use a pre-built image.

#### 5.3.7 ModelScope Download Compatibility Issues

If Hugging Face is inaccessible, you can use:

```bash
export VLLM_USE_MODELSCOPE=True
pip install "modelscope>=1.18.1,<1.38"
```

ModelScope 1.38 and later releases changed Hub behavior, the cache layout, and repository metadata, which may cause older vLLM model-ID integrations to fail.

If you need to use a newer version, explicitly download the model first:

```bash
pip install "modelscope>=1.38"

python - <<'PY'
from modelscope import snapshot_download

snapshot_download(
    "Qwen/Qwen3-0.6B",
    local_dir="/data/models/Qwen3-0.6B",
)
PY

unset VLLM_USE_MODELSCOPE
```

Then replace the model ID in the Quickstart with:

```text
/data/models/Qwen3-0.6B
```

#### 5.3.8 Installation Verification Passes, but the Model Fails to Run

If the following checks pass:

- NPU tensor operation;
- vLLM Ascend plugin registration;
- core package check;

then troubleshoot the failure at the model layer:

- whether the model supports the target hardware;
- dtype/quantization;
- TP/DP/EP;
- graph parameters;
- device memory;
- model weights and tokenizer.

Continue with:

- [Quickstart](quick_start.md)
- [Supported Models](../user_guide/support_matrix/supported_models.md)
- [Model Tutorials](../tutorials/models/index.md)
- [FAQs](../faqs.md)
