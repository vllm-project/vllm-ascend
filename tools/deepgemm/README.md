# Building DeepGEMM on Ascend 950

The standard vLLM Ascend build fetches a pinned DeepGEMM-Ascend revision and
builds it with the dependency's native Torch extension. The resulting wheel
contains `vllm_ascend.third_party.deep_gemm`, including the extension and the
headers needed by the device JIT. The same package is staged in editable builds.
The pin and CMake target live in
[`deepgemm.cmake`](../../cmake/external_projects/deepgemm.cmake).

Use the normal `pip install .` or `pip install -e .` from the repository root
in a supported 950 Torch/CANN build environment. For a local checkout, set
`DEEPGEMM_SRC_DIR` to its path and initialize its `third-party/deep-jit` and
`third-party/magic_enum` submodules. `FETCHCONTENT_BASE_DIR` controls the default
download directory.

The A5 Dockerfiles install GCC with C++20 `<format>` support and elfutils headers.
An explicit `CXX` overrides automatic compiler selection for DeepGEMM. The
native setup generates Python stubs using the two parser packages in
`tools/deepgemm/requirements-build.txt`. Isolated A5 native builds request these
automatically. With `--no-build-isolation`, install them first using
`pip install -r tools/deepgemm/requirements-build.txt`.

Run `python tools/deepgemm/verify_install.py` to check the installed extension
and JIT headers. See the [configuration guide](../../docs/source/user_guide/configuration/additional_config.md)
to enable `deepgemm_mega_moe` with `enable_fused_mc2=3`.
