# Qualifying NumPy and PyArrow dependency candidates

The **CPU dependency compatibility probe** workflow resolves the repository's
Triton-Ascend pin together with explicit NumPy and PyArrow candidate versions
in a clean Python 3.12 Linux x86_64 environment. It runs `pip check`, NumPy/Arrow
array conversions, and a Parquet roundtrip. It records pip resolution and
installation reports, installed metadata, the checkout revision, and logs.

Run the workflow manually from GitHub Actions to select candidate versions.
Changes to the workflow or its smoke tests also run the default baseline:
NumPy 1.26.4 and PyArrow 16.1.0. These are probe inputs, not new production
dependency declarations. The workflow reads the Triton-Ascend version from
`requirements.txt` rather than maintaining another version pin.

An incompatible candidate must fail normal resolution. The workflow uses
binary wheels from PyPI and the Ascend package index and does not override
published requirements. If resolution fails, inspect the uploaded
`resolution.log`; pip may not produce a JSON report for a failed resolution.
Installed metadata after a failed install may only describe the probe tools,
so it must not be interpreted as a successfully installed candidate stack.

To repeat the probe locally in a clean environment, install the repository's
Triton-Ascend pin and the selected NumPy/PyArrow versions using normal pip
resolution, then run:

```bash
python -m pip check
python -m unittest tests.dependency_compatibility.numpy_pyarrow_smoke -v
```

The smoke module is deliberately opt-in: ordinary pytest discovery does not
collect it or require PyArrow for unrelated tests. Both candidate libraries
must be installed before running it; missing imports fail rather than skip.

This verifies a narrow CPU dependency graph. It does not install the full
vLLM-Ascend runtime, import torch-npu, execute CANN tools, or compile/run NPU
kernels. The current Triton-Ascend wheel's NumPy constraint may reject NumPy
2.x even when NumPy/PyArrow alone passes these CPU tests. A successful probe
does not establish an ABI-compatible version range or authorize relaxing
production pins. Before changing those pins, run the supported native/NPU
tests and coordinate release metadata with the upstream project.

See [the short-term compatibility task](https://github.com/vllm-project/vllm-ascend/issues/18802)
for the remaining qualification work.
