### 5.1 CPU-only Build Verification {: #cpu-only-build-verification }

CPU-only build verification checks only the Python/package build. It does not verify:

- the NPU runtime;
- custom kernels;
- NPU inference;
- NPU-specific tests.

Even without an NPU, the build stage generally still requires matching CANN Toolkit headers and libraries.

Install the build dependencies:

```bash
pip install --upgrade \
    pip \
    "setuptools>=64" \
    "setuptools-scm>=8" \
    wheel \
    attrs \
    googleapis-common-protos \
    "cmake>=3.26" \
    ninja
```

After installing the matching dependencies, run:

```bash
export ASCEND_TOOLKIT_HOME="${
    ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit/latest
}"

export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export COMPILE_CUSTOM_KERNELS=0
```

Set `SOC_VERSION` for the target hardware.

For example:

```bash
# Atlas A2
export SOC_VERSION=ascend910b1

# Atlas A3
# export SOC_VERSION=ascend910_9391

# Atlas 300I DUO
# export SOC_VERSION=ascend310p1
```

Then run:

```bash
pip install --no-build-isolation -e .
PYTHONPATH= pip check
```

!!! note

    Passing CPU-only verification does not mean that the runtime environment is ready.
    Before release or runtime use, you must still complete the installation verification and Quickstart inference on the corresponding NPU.
