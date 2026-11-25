# Architecture Context Map

Orientation for the reviewing agent: what this repository is, where things
live, and which upstream surfaces the plugin inherits from. This file is
context, not rules — rules live in the L1/L2 checklists with citable IDs.

## What this repository is

vLLM Ascend is the hardware plugin that runs upstream vLLM on Huawei
Ascend NPUs (A2/910B, A3, A5, 310P). It is NOT a fork: it integrates via
vLLM's pluggable platform interface (NPUPlatform registered as an OOT
platform) plus a bounded set of monkey-patches under `vllm_ascend/patch/`.
Most classes here inherit an upstream vLLM base class and override only
the NPU-specific parts — that inheritance contract is the single most
important thing to keep in mind when reviewing a diff here.

## Module map (coarse, for finding context while reviewing a diff)

- `vllm_ascend/__init__.py` — plugin entry: register() returns NPUPlatform
- `vllm_ascend/platform.py` — NPUPlatform: central platform abstraction (device 'npu', dispatch PrivateUse1)
- `vllm_ascend/worker/` — NPUWorker(GPUWorker) in `worker.py`; the model runner is NPUModelRunner(GPUModelRunner) in `worker/v2/model_runner.py` (V2 only — the legacy V1 runner has been removed); `worker/v2/` also carries the sampling helpers (`v2/sample/`), speculative-decoding proposers (`v2/spec_decode/`: eagle, dflash, dflash2, dspark, ...) and ACL-graph utilities
- `vllm_ascend/mrv2_utils.py` — shared helpers for the V2 runner
- `vllm_ascend/_310p/` — self-contained 310P implementation: NPUWorker310(NPUWorker) in `worker_310p.py`, NPUModelRunner310V2(NPUModelRunner) in `worker/v2/`, plus its own attention, fused_moe, ops and quantization
- `vllm_ascend/attention/` — Ascend attention backends and metadata builders: attention_v1 (GQA/FIA), mla_v1, sfa_v1 (+ sfa_hisparse, sfa_kv_offload), dsa_v1/dsa_v41, fa_v1, fa3_v1, flashmla, indexer/indexer_kpool, context_parallel
- `vllm_ascend/compilation/` — ACL graph capture/replay (ACLGraphWrapper, API-compatible with CUDA graph) and the inductor compiler bridge (AscendCompiler, graph fusion passes)
- `vllm_ascend/core/` — scheduler extensions and Ascend KV-cache specs (batch-job-aware scheduler, dyntra LB, kv_cache_interface)
- `vllm_ascend/device/`, `device_allocator/` — hardware profiles (A2/A3/A5/310P) and the CaMem allocator
- `vllm_ascend/distributed/` — parallel state, NPUCommunicator, MC2/MLP-TP, KV transfer (Mooncake/AscendStore/UCM), EPLB-related comms
- `vllm_ascend/quantization/` — W8A8/W4A8/W4A4/WNA16 and KV-cache quant methods
- `vllm_ascend/models/` — Ascend-specific model implementations registered into ModelRegistry (deepseek_v4/v41, glm5*, kimi_k3, llama_eagle3, ...)
- `vllm_ascend/ops/` — operator layer: fused MoE (routers/dispatchers/quant/eplb), triton kernels, custom op registration
- `vllm_ascend/patch/` — upstream monkey-patches; `platform/` = global (pre-worker), `worker/` = per-worker; catalog in `patch/__init__.py`
- `vllm_ascend/model_executor/` — weight offloader/prefetch and kernel warmup
- `vllm_ascend/model_loader/` — non-standard loaders (netloader, rfork, bigtensorloader)
- `vllm_ascend/eplb/`, `ec_manager/` — EPLB policies and expert-parallel manager
- `vllm_ascend/_cann_ops_custom/` — CANN custom operator kernels and compilation pass utilities
- `vllm_ascend/ascend_config.py` — AscendConfig singleton and typed sub-configs (`--additional-config`)
- `vllm_ascend/envs.py` — the ONLY place VLLM_ASCEND_* env vars are defined

## Inheritance surfaces (why signature changes here are dangerous)

NPUPlatform <- vllm Platform; NPUWorker <- GPUWorker; NPUModelRunner <-
GPUModelRunner (V2 only); NPUWorker310 <- NPUWorker; NPUModelRunner310V2 <-
NPUModelRunner; Ascend attention backends <- AttentionBackend /
AttentionImpl / MLAAttentionImpl; ACLGraphWrapper <- CUDA-aware graph
wrapper; AscendCompiler <- CompilerInterface; NPUCommunicator <-
DeviceCommunicatorBase. A signature or call-order change on the upstream
side of any arrow must be reflected in every Ascend override — see
`L2/model-execution/checklist.md` (`L2-exec#override-contract`) for the
review rule.
