# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

IMPORTANT: Thoroughly review [AGENTS.md](AGENTS.md) before beginning any work — it contains the authoritative contributor guidelines (env var registration, patch review requirements, commit sign-off, NPU-specific practices).

## What this repo is

vLLM-Ascend is a **hardware plugin** for [vLLM](https://github.com/vllm-project/vllm) that enables serving on Huawei Ascend NPUs. It does not modify upstream vLLM; it registers itself via the `vllm.platform_plugins` entry point (`vllm_ascend:register`) and monkey-patches vLLM classes at runtime. C++ custom operators live in `csrc/` and are built into `vllm_ascend/_cann_ops_custom/`.

The matching vLLM commit is recorded in `.github/vllm-main-verified.commit`; dev work pairs an editable vLLM checkout at that commit with an editable install of this repo. On this machine the three repos live in `/home/z00886386/{vllm,vllm-ascend,flash-linear-attention-npu}`.

## Build and install

Requires a container with Ascend driver + CANN (`source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh`), torch/torch-npu matched to CANN, and cmake/ninja. All installs use `--no-build-isolation --no-deps` so the container's torch/torch-npu is never replaced.

```bash
# vLLM (upstream, pinned commit from .github/vllm-main-verified.commit)
cd /home/z00886386/vllm
git checkout ba07e4a48fc951300d97eb506217dd530583dea3   # example pinned commit
VLLM_TARGET_DEVICE=empty python3 -m pip install --no-build-isolation --no-deps -e .
# needs setuptools-rust in the env (pip install setuptools-rust)

# vLLM-Ascend: SOC_VERSION selects the target hardware family
export SOC_VERSION=ascend950dt_9582   # A5. A2: ascend910b1, A3: ascend910_9391, 310P: ascend310p1
git submodule update --init --recursive   # csrc/third_party/catlass
python3 -m pip install --no-build-isolation --no-deps -e .
```

**Do NOT set `VLLM_VERSION`** when running against the pinned vLLM main commit. `vllm_version_is("0.27.1")` (in `vllm_ascend/utils.py`) switches patch branches based on this env var, but the pinned commit is post-0.27.1 (e.g. its pcp module moved to `vllm/v1/attention/ops/pcp`), so forcing `0.27.1` picks stale branches and crashes EngineCore with `ModuleNotFoundError: vllm.model_executor.layers.attention.pcp`. Leave it unset so the installed vLLM's real version string is used.

The build writes `vllm_ascend/_build_info.py` (auto-generated `__device_type__`); device detection at runtime derives everything (A2/A3/A5/310P, `get_fla_gdn_soc()`) from it.

## Tests and lint

```bash
# Unit tests (no NPU needed for most)
pytest -q tests/ut/ops/test_gdn_fla.py
pytest -q tests/ut/device/test_device_config.py

# Real-NPU operator tests
pytest -s -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py

# Single test case
pytest -sv tests/ut/ops/test_gdn_fla.py -k 'backend_config'
pytest -sv tests/ut/ops/test_gdn_fla.py::test_parse_per_operator_overrides

# Lint (required before commit, covers all file types incl. markdown)
bash format.sh ci
ruff check vllm_ascend/
ruff format vllm_ascend/
```

Serve smoke test (A5, single card):

```bash
export ASCEND_RT_VISIBLE_DEVICES=3           # physical card -> logical npu:0
export VLLM_ASCEND_GDN_BACKEND=fla_npu        # strict FLA GDN mode
vllm serve /mnt/weight/Qwen3.6-35B-A3B --dtype bfloat16 --tensor-parallel-size 1 \
  --enforce-eager --max-model-len 2048 --max-num-seqs 1 --gpu-memory-utilization 0.95
```

## Architecture: patch system

Patches are applied in two waves, never at import of `vllm_ascend` itself:

- `adapt_patch(is_global_patch=True)` → imports `vllm_ascend/patch/platform/` — scheduler/engine-level patches, applied from the plugin entry points (`_ensure_global_patch` in `vllm_ascend/__init__.py`).
- `adapt_patch()` → imports `vllm_ascend/patch/worker/` — model/worker-level monkey-patches, applied in `NPUWorker.__init__` (`vllm_ascend/worker/worker.py`).

Worker patches replace methods on upstream vLLM classes at module level (e.g. `patch/worker/patch_qwen3_5.py` sets `QwenGatedDeltaNetAttention.forward/_forward_core = AscendGatedDeltaNetAttention.*`). Import-time class identity matters: the patch target must be the exact class object the model module instantiates.

## Architecture: GDN (Qwen3.5/3.6 linear attention) FLA path

Active feature area on the current branch. Key files:

- `vllm_ascend/ops/gdn.py` — `AscendGatedDeltaNetAttention`, the patched forward/_forward_core. Runs on A2/A3/A5 via `get_fla_gdn_soc()`; falls back to native AscendC ops when the FLA adapter is unavailable.
- `vllm_ascend/ops/gdn_fla.py` — `FlaGDNAdapter` / `FlaGDNOperatorDispatcher`: resolves FLA operators (from the `flash-linear-attention-npu` wheel), probes them on scratch tensors, logs one selection per operator, and enforces backend policy.
- Backend policy: `VLLM_ASCEND_GDN_BACKEND` (`auto`|`fla_npu`|`native`) plus per-op `VLLM_ASCEND_GDN_OP_BACKENDS="op=backend,..."`. Strict `fla_npu` raises on any resolve/probe failure; `auto` silently falls back (native path logs nothing).
- Prefill preferred path is a single fused `fla_npu.ops.ascendc.gdn_core_fwd_phase6` call (device kernel `ChunkGdnCoreFwd`); ordinary decode uses `fla_npu.ops.ascendc.recurrent_gated_delta_rule`; both require bf16 activations (strict mode raises otherwise) and PCP world size 1.
- FLA wheels are SoC-specific: `FLA_NPU_SOC=ascend950` (A5), `ascend910b` (A2), `ascend910_93` (A3). The FLA repo used on this machine is `/home/z00886386/flash-linear-attention-npu` branch `chw_new_cumsum_kkt_solve_tri_newest` (A5), or `chw_new_cumsum_kkt_solve_tri_simple` per the A2/A3 guide.
- GDN design/validation docs live under `docs/superpowers/` (guides, plans, specs, reports).

## Environment variables

All env vars must be registered in `vllm_ascend/envs.py` (`VLLM_ASCEND_*` naming, documented default) — never hardcoded elsewhere. Notable ones: `VLLM_ASCEND_GDN_BACKEND`, `VLLM_ASCEND_GDN_OP_BACKENDS`, `SOC_VERSION` (build-time), `VLLM_VERSION` (see pitfall above).

## Logging pitfall

vLLM only configures the `vllm` logger namespace by default. `vllm_ascend.*` INFO logs (e.g. `GDN FLA operator selected: ...` from `gdn_fla.py`) are dropped unless you either rely on `configure_ascend_logging` (not always effective in EngineCore) or launch with `VLLM_LOGGING_CONFIG_PATH` pointing to a JSON config that adds `"vllm_ascend": {"handlers": ["vllm"], "level": "INFO", "propagate": false}` (a working example exists at `/tmp/vllm-logging-config.json` on this machine). Ascend-specific logs also go to rotating files under `~/ascend/log/vllm_ascend/`.

## Profiling

Serve with `--profiler-config '{"profiler": "torch", "torch_profiler_dir": "/home/z00886386/profile", "torch_profiler_with_stack": false}'`, then drive `/start_profile` and `/stop_profile` via HTTP POST on the main API server port (this vLLM version has no separate profiler port; there is no 8908). Worker NPU traces land under `torch_profiler_dir/` in `dp0_..._ascend_pt/` directories (check `ASCEND_PROFILER_OUTPUT/trace_view.json` for device kernel names like `ChunkGdnCoreFwd`); `stop_profile` can take minutes to export.
