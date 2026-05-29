# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

vllm-ascend is a hardware plugin that enables vLLM to run on Huawei Ascend NPU. It integrates with upstream vLLM via the pluggable hardware interface (entry points in `setup.py`). The plugin does not add new model files directly; instead it patches upstream vLLM behavior and extends components via inheritance.

## Build and Install

```bash
# Editable install (triggers CMake build for custom kernels)
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# Compile custom kernels (enabled by default via COMPILE_CUSTOM_KERNELS=1)
# The build requires CANN toolkit and NPU drivers. On CPU-only environments:
export COMPILE_CUSTOM_KERNELS=0

# Build targets a specific chip; auto-detected via npu-smi, or set manually:
export SOC_VERSION=ascend910b1   # Atlas A2
export SOC_VERSION=ascend910_9391 # Atlas A3
export SOC_VERSION=ascend310p1   # Atlas 300I
```

Build artifacts:
- C++ extensions are built via CMake and installed into `vllm_ascend/`
- AclNN custom ops are built via `csrc/build_aclnn.sh`
- `_build_info.py` is auto-generated with the detected device type

## Lint and Format

```bash
# Run all linting and formatting checks (ruff, codespell, typos, markdownlint, shellcheck)
bash format.sh

# Run the same checks as CI
bash format.sh ci

# Run only Python linting
ruff check vllm_ascend/
ruff format vllm_ascend/

# Install lint tools
pip install -r requirements-lint.txt
pre-commit install
```

## Testing

```bash
# Run all unit tests
pytest tests/ut/

# Run a specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run a specific unit test function
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs

# Run system tests (requires NPU hardware)
pytest -sv tests/e2e/singlecard/test_piecewise_res_cons_consistency.py
```

## High-Level Architecture

### Plugin Registration

`vllm_ascend/__init__.py` defines entry points registered in `setup.py`:
- `register()` → `NPUPlatform`
- `register_model_loader()` → custom net/rfork loaders
- `register_connector()` → KV transfer connector
- `register_service_profiling()` → profiling config
- `register_model()` → model registrations in `vllm_ascend/models/`

Global patches are applied via `adapt_patch()` before engine initialization.

### Platform Layer

`vllm_ascend/platform.py` — `NPUPlatform` is the central platform abstraction that vLLM dispatches to for NPU-specific behavior (device selection, attention backend, communication, etc.).

### Worker and Model Runner

- `vllm_ascend/worker/worker.py` — `NPUWorker` (vLLM v1 worker)
- `vllm_ascend/worker/model_runner_v1.py` — `NPUModelRunner` (v1 model runner)
- `vllm_ascend/worker/v2/model_runner.py` — v2 model runner
- `vllm_ascend/_310p/model_runner_310p.py` — Ascend 310P specific runner

### Attention Backends

`vllm_ascend/attention/` contains NPU-specific attention implementations:
- `attention_v1.py` — main v1 attention backend dispatch
- `dsa_v1.py` — DeepSeek Attention
- `mla_v1.py` — Multi-Head Latent Attention
- `sfa_v1.py` — another attention variant
- `context_parallel/` — context parallel attention
- `kvcomp_attn/` — KV cache compressed attention

### Custom Operators

Three layers of custom ops:
1. **Python custom ops** — `vllm_ascend/ops/` (rotary embedding, layernorm, linear, GDN, etc.)
2. **Triton kernels** — `vllm_ascend/ops/triton/` (rms_norm, penalty, fused ops)
3. **C++/AscendC kernels** — `csrc/` built via CMake; includes kernels for MLA preprocess, batch matmul, MoE, etc.

Ops are registered in `vllm_ascend/ops/register_custom_ops.py`.

### Patching System

Upstream vLLM behavior is modified via patches in `vllm_ascend/patch/`:
- `patch/platform/` — platform-level patches (distributed, scheduling, KV cache, speculative config)
- `patch/worker/` — model-specific patches (DeepSeek, Qwen, MiniMax, LLaMA-Eagle, etc.)

Patches monkey-patch upstream classes/methods. New patches require architectural review.

### Compilation and Graph Fusion

`vllm_ascend/compilation/` manages ACL graph capture and fusion:
- `acl_graph.py` — graph parameter management
- `graph_fusion_pass_manager.py` — fusion pass orchestration
- `passes/` — individual graph fusion passes

### Configuration

`vllm_ascend/ascend_config.py` parses vLLM's `additional_config` into `AscendConfig`, which contains sub-configs for:
- `xlite_graph_config`
- `ascend_compilation_config`
- `ascend_fusion_config`
- `finegrained_tp_config`
- `eplb_config`
- `weight_prefetch_config`
- `profiling_chunk_config`

### Environment Variables

All environment variables must be defined in `vllm_ascend/envs.py` in the `env_variables` dict. Never hardcode env var names elsewhere; import from `vllm_ascend.envs`.

## Development Conventions

- **Commits**: Use Conventional Commits format with sign-off: `git commit -s`
- **PR titles**: `[Type][Module] Description` (e.g., `[BugFix][Worker] Fix CPU binding`)
- **Imports**: Keep at top of file; inline imports only for circular dependencies or lazy loading
- **No magic numbers**: Use named constants
- **Avoid `tensor.item()` on device tensors** in hot paths — it causes CPU-NPU sync overhead
- **No new global mutable state** without justification

## Important Notes

- Model-specific behavior should be implemented via **patching** or **inheritance**, not by adding new model files.
- All new patches and model runner changes require strict architectural review.
- NPU-specific tests must be verified on actual Ascend hardware.
- The PR description template in `.github/PULL_REQUEST_TEMPLATE.md` must be completed.
