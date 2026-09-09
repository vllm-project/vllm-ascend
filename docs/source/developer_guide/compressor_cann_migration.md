# Validating the CANN Compressor migration

DeepSeek V4 now calls the formal CANN `aclnnCompressor` for the KV/Gate projections,
state-cache update, overlap and softmax-weighted compression. The model applies
FP32 RMSNorm and partial interleave RoPE afterwards. The previous locally built
Compressor fused these stages and exposed an incompatible ACLNN signature.

## Environment

Use a clean Linux environment with an Atlas A3 or Ascend 950 device. One device
is sufficient for the operator tests; model tests need the model's full hardware
and weight requirements. The beta.2 Compressor documentation does not list A2 as
supported. On A3 it also excludes circular state caches and noncontiguous axis 0.

| Component | Validation baseline |
| --- | --- |
| OS / Python | Ubuntu 22.04, Python 3.12 (same baseline as the repository Dockerfiles) |
| Driver / firmware | Ascend HDK 26.1.1, with firmware matched to the hardware |
| CANN | 9.2.0-beta.2 Toolkit and matching ops packages |
| ops-transformer | 9.2.0-beta.2; source tag `v9.2.0-beta.2`, commit `5f33f1e23d41fe0a047d01b7c7ae278777cac1f5` |
| NNAL / ATB | Matching 9.2.0-beta.2 if required by the selected hardware build |
| vLLM | `b2f685834a6456197e7033966fdef52a23f1abcd`, the repository's `.github/vllm-main-verified.commit` when this change was prepared |
| vLLM Ascend | The migration PR's exact head commit; rebuild its C++ extension and custom operators |
| PyTorch | 2.10.0, pinned by this repository |
| torch-npu | Repository pin: 2.10.0.post4; compatibility with CANN beta.2 still needs confirmation |
| Other Python dependencies | This checkout's `requirements.txt` and `pyproject.toml`, including triton-ascend 3.2.2 and transformers 5.14.1 |

The [CANN release notes](https://gitcode.com/cann/release-management/blob/2d0509b826141bfa4187aced485907413c0b2d5c/9.2.0-beta.2/release-notes.md)
specify the HDK and package versions. The public
[TorchNPU compatibility matrix](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.md)
currently pairs torch-npu 2.10.0.post4 with CANN 9.1.0, not beta.2. The table above
therefore describes a migration validation environment, **not a certified or
hardware-tested complete stack**. Confirm a beta.2-compatible TorchNPU build with
the package provider and record its exact version if it differs from the pin.

The repository's generic Dockerfiles still default to CANN 9.1.0. Do not use an
unchanged prebuilt image to validate this migration, or mix a beta.2 Toolkit with
older ops packages. Other models' dependency defaults are not upgraded by this
operator migration.

## Build and check the installed libraries

Activate the beta.2 Toolkit, ops and, where needed, ATB environment scripts from
their actual installation paths. Use a fresh checkout and build directory so
the removed custom Compressor is not retained in an old OPP installation.

```bash
git clone https://github.com/vllm-project/vllm.git /vllm-workspace/vllm
git -C /vllm-workspace/vllm checkout b2f685834a6456197e7033966fdef52a23f1abcd
VLLM_TARGET_DEVICE=empty python -m pip install -e /vllm-workspace/vllm

# Check out the PR's exact head in /vllm-workspace/vllm-ascend first.
cd /vllm-workspace/vllm-ascend
COMPILE_CUSTOM_KERNELS=1 python -m pip install -e . --no-build-isolation
python -m pip check
```

Use the normal hardware-specific `SOC_VERSION` and build prerequisites described
in the installation guide. Install the pinned dependencies before using
`--no-build-isolation`; if testing a provider-supplied TorchNPU build, ensure pip
does not replace it with the repository pin during installation.

The new binding searches `libopapi_transformer.so`, then `libopapi.so`, without
falling back to the experimental `libcust_opapi.so`. Check that the installed
formal library exports both `aclnnCompressorGetWorkspaceSize` and
`aclnnCompressor`. Absence of either produces a dependency error.

```bash
npu-smi info
python -m pip show torch torch-npu vllm vllm-ascend triton-ascend transformers
git -C /vllm-workspace/vllm rev-parse HEAD
git -C /vllm-workspace/vllm-ascend rev-parse HEAD
```

Also record the Toolkit and ops package `version.info` files, loaded library
paths, and whether the state cache has a padded first-axis stride.

## Tests

```bash
python -m pytest -sv tests/ut/models/test_deepseek_v4_compressor.py tests/ut/attention/test_dsa_cp.py
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor.py \
    --junitxml=compressor-cann-results.xml
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_compressor_metadata.py
```

The new operator tests cover C4Li, C4A and C128A, FP16/BF16 prefill, ragged batches,
chunked prefill followed by decode, state-cache updates (including padded pages), empty input, Meta shapes,
invalid attributes, and full-chain graph replay. Latency cases record
`compressor_chain_ms` and `peak_allocated_bytes` in JUnit output. They measure the
new full chain; they do not assert a performance improvement over the old kernel.

Before merging, compare the old and new full chains on the same hardware,
requests, weights and graph mode. Run DeepSeek V4 Attention and Indexer model
accuracy/performance tests, including asynchronous metadata, prefix caching,
graph padding and the actual padded state-cache layout used by the model.

Two changes require explicit measurement:

- Formal Compressor writes BF16/FP16 before RMSNorm/RoPE. Widening afterwards
  does not recover the precision lost at that boundary; bitwise parity is not
  promised.
- The formal beta.2 API declares two required FP32 intermediate outputs, even
  with `gradEnabled=false`. The binding allocates their documented shapes and
  does not consume them. These allocations and the separate postprocessing
  launches can increase peak memory and latency. Do not replace them with null
  or undersized buffers without verifying that the installed API permits it.

Hardware compilation, numerical accuracy and performance validation are pending.
