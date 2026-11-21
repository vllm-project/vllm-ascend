# Building the Ascend DeepGEMM backend

The 950 build vendors a pinned DeepGEMM-Ascend revision and its recursive
dependencies into `vllm_ascend.third_party.deep_gemm`. It does not import an
unrelated system `deep_gemm` package. Editable installation copies the native
extension, device JIT headers and build manifest into the source package.

The pinned revision includes the packed FP4 weight layout from
[DeepGEMM-Ascend #32](https://github.com/deepseek-ai/DeepGEMM-Ascend/pull/32).
This framework change depends on that upstream change. Packing occurs once
when weights load; it preserves expert placement and routing. The Python
transform and device JIT headers must come from the same pinned revision.

## Prerequisites

Use the framework's supported Torch, torch_npu and CANN environment. The
DeepGEMM build additionally requires:

- A C++20 compiler with `<format>` support (GCC 13 or newer).
- The elfutils development headers and libraries (`libdw-dev` on Ubuntu,
  `elfutils-devel` on openEuler).
- `tree-sitter==0.25.2` and `tree-sitter-cpp==0.23.4`, as declared in
  `pyproject.toml`. These are build dependencies, not serving requirements.

`install_build_deps.sh` installs these prerequisites for a builder image. When
building with `--no-build-isolation`, install the parser dependencies in the
build environment explicitly. A virtual environment with system site packages
can reuse the image's Torch and CANN packages without changing serving clients'
parser versions.

An explicit `CXX` takes precedence. Otherwise the builder selects an installed
GCC 13/14 toolchain, or the GCC 14 shipped under
`$ASCEND_HOME_PATH/tools/hcc`. The latter uses the host elfutils headers and
libraries and a private static libstdc++; these flags apply only to the
DeepGEMM extension. No compiler environment is exported into serving.

## Build and inspect

For a standalone build of the same vendored payload used by `setup.py`:

```bash
python tools/deepgemm/build.py \
  --source .deps/deepgemm-ascend \
  --work-dir build/deepgemm-ascend \
  --destination build/deepgemm-payload
```

The source checkout must match the pinned revision and have no tracked edits.
The generated `ascend_build.json` records the revision, submodules, framework
versions, compiler flags and the native wheel SHA-256. The wheel payload is
copied unchanged: CANN include fixes and optional TileLang loading live in the
pinned dependency source. Build-only `DG_JIT_*` variables are removed before
creating the wheel. Keep the manifest with the binary when reproducing a build.

After the normal editable framework build, verify import from the source tree:

```bash
python -c 'from vllm_ascend.ops.fused_moe.deepgemm_loader import load_deepgemm; print(load_deepgemm().__file__)'
```

This verifies package delivery only. Multi-rank numerical checks, graph replay
and serving throughput remain separate validation requirements.

## Selecting the backend

Enable `--additional-config '{"deepgemm_mega_moe":true,"enable_fused_mc2":0}'`
for DeepSeek V4.1 on Ascend 950 with TP1, EP8 and linear expert placement.
EPLB, MoE LoRA and overlapping microbatches are rejected before weights are
converted. The default is disabled. The typed key is documented in the
[additional configuration reference](../../docs/source/user_guide/configuration/additional_config.md).

This selects both fused communication and expert execution. The existing
`enable_fused_mc2` option selects CANN's different native backend; the upstream
CUDA expert-kernel selector does not describe this Ascend symmetric-memory
communication contract. The backend follows Ascend's existing
prepare/fused-experts/finalize extension points instead of changing upstream
routing or collective order. Weight packing belongs in the quantization
post-load hook, before the ordinary CANN NZ conversion, and runs once. Shared
linear modules defer their conversion to that same owner.

The existing routed-expert capture patch classifies this communication method
with other token-sharded methods; it adds no new patch registration. Remove
that classification when upstream capture learns the prepare/finalize token
layout. Hardware acceptance must cover zero-token ranks, target/draft reuse,
shared experts, changed-input graph replay and the disabled-backend path.

Each new target/draft layout is compiled in dry-run mode during initialization,
before its symmetric buffer is published for reuse. Ranks rendezvous on the CPU
process group after compilation so a faster compiler cannot enter the device
rank barrier while a peer is still compiling. The previous dry-run setting is
restored on success or failure. This runs once per layout; ordinary expert
execution adds no host synchronization.
