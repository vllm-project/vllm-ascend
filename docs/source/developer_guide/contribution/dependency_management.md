# Python dependency ownership and verification

## Sources of truth

`requirements.txt` is the runtime dependency input to `setup.py`'s
`get_requirements()` and therefore to wheel `Requires-Dist` metadata.
`pyproject.toml`'s `[build-system].requires` describes the PEP 517 isolated
build environment. These environments have different responsibilities;
the two lists should not be mechanically mirrored.

`requirements-dev.txt` includes the runtime and lint requirements and adds
development/test dependencies. `requirements-lint.txt`,
`docs/requirements-docs.txt`, and `benchmarks/requirements-bench.txt` serve
their respective workflows. Requirements declarations are not complete
environment locks: they do not record all transitive versions, wheel hashes,
Python/platform tags, CANN, or the verified upstream vLLM revision.

Native builds need the matching torch/torch-npu installation as well as
CMake, pybind11, and the CANN toolchain. Preserve paired versions when
changing dependencies. Reducing build isolation requirements requires
native-build evidence; a Python-only wheel cannot prove native compatibility.
See [the build-isolation discussion](https://github.com/vllm-project/vllm-ascend/issues/17791)
and [the existing build/runtime contract PR](https://github.com/vllm-project/vllm-ascend/pull/17908).

## Verify a dependency change

Run the focused regression checks and verify the actual built artifact:

```bash
python -m unittest tests.ut.test_wheel_dependencies -v
python -m build --wheel --no-isolation
python -m tools.check_wheel_dependencies dist/*.whl
```

Use an environment prepared for the supported hardware stack. For an
offline metadata-only smoke test, `SOC_VERSION=ascend910b1
COMPILE_CUSTOM_KERNELS=0 python -m build --wheel --no-isolation
--skip-dependency-check` avoids native compilation and the build dependency
precheck. The dependency metadata CI job uses this restricted path. It
checks the wheel contract and installation with `--no-deps`; it does not
verify resolution, imports, CANN, or NPU kernels.

In a clean supported Linux environment with the required package indexes,
also validate resolution and the installed dependency graph:

```bash
python -m pip install --dry-run --report dependency-report.json dist/*.whl
python -m pip install dist/*.whl
python -m pip check
python -c 'from importlib.metadata import requires; print(requires("vllm-ascend"))'
```

Run native build, NumPy/torch tensor conversion, and the relevant NPU tests
before claiming hardware compatibility. `pip check` verifies declared
requirements, not ABI compatibility. Record Python, architecture, CANN,
torch, torch-npu, Triton-Ascend, NumPy, PyArrow, package indexes, and the
verified vLLM revision with the result. Keep environment locks separate
from published runtime compatibility declarations.

## Investigating transitive NumPy constraints

Inspect the exact installed distribution or downloaded wheel, rather than
inferring published metadata from a development requirements file:

```bash
python -c 'from importlib.metadata import requires; print(requires("triton-ascend"))'
python -m pip check
```

For example, the Huawei index's Triton-Ascend 3.2.2 CPython 3.12 x86_64
wheel declares `numpy==1.26.4`. This matches
[the release source](https://github.com/triton-lang/triton-ascend/blob/v3.2.2/python/setup.py#L875-L889),
while current main has a different packaging layout. Always distinguish
release, architecture, index, and revision when comparing requirements.
Do not override that pin with `--no-deps` and claim compatibility from a
successful resolver run. NumPy 2.x validation must include compiled
extensions and the CANN/NPU stack.
