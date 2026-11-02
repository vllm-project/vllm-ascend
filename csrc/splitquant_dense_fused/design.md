# SplitQuant native build design

The SplitQuant implementation is built as one shared library,
`libvllm_ascend_splitquant_dense.so`, from the sources in this directory. CMake
installs it under `vllm_ascend/lib`; the Python quantization layer loads that
library in each worker process.

The library registers the dense, multi-projection, state-cache, router, and MoE
operators used by the Qwen3 SplitQuant model. The public signatures and the
group-size-128 contract are documented in `README.md` and `ops.h`. Inputs must
be on the NPU, finite, and have the documented dtype and shape; autograd is not
supported.

The model integration keeps the fused router as the default for supported BF16
top-8, 128-expert batches. Unsupported shapes fall back to the reference
router, and the evaluator records those fallbacks. The native library is loaded
from the package installation and is not selected through an external library
override.

Correctness is checked with `test/run_equivalence.py`. Model-level validation
must record the model and dataset checksums, native-library checksum, runtime
flags, and the router/MoE call counters so that a result can be reproduced.
