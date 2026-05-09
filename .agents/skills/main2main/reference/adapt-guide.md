# Adapt Guide — How to Analyze Upstream Changes

This is the reference material for the adapt phase. Come here when you need the file mapping table, subsystem-specific guidance, or the concrete commands for analyzing upstream diffs. The thinking framework is in SKILL.md — this document is for lookup, not for reading end-to-end.

---

## vLLM Key Areas to Focus On

When analyzing vLLM changes, pay special attention to these areas that typically require vLLM Ascend adaptation:

1. **Platform Interface** (`vllm/platforms/`)
   - New abstract methods that must be implemented
   - Method signature changes
   - New platform features

2. **MoE (Mixture of Experts)** (`vllm/model_executor/layers/fused_moe/`)
   - FusedMoE layer changes
   - Activation function changes
   - Router changes

3. **Attention** (`vllm/model_executor/layers/attention/`)
   - Attention backend changes
   - New parameters or interfaces
   - MLA (Multi-Head Latent Attention) updates

4. **Speculative Decoding** (`vllm/v1/worker/gpu/spec_decode/`, `vllm/config/speculative.py`)
   - Import path changes
   - Config field changes
   - New speculative methods

5. **Distributed** (`vllm/distributed/`)
   - Parallel state changes
   - KV transfer changes
   - Device communicator updates

6. **Models** (`vllm/model_executor/models/`)
   - New model architectures
   - Model interface changes

7. **Worker/Model Runner** (`vllm/v1/worker/gpu/model_runner.py`)
   - New worker methods
   - Model runner changes

8. **Quantization** (`vllm/model_executor/layers/quantization/`)
   - Quantization config changes
   - compress-tensor method changes

---

## vLLM Ascend Key File Locations

| Project | Path |
|---------|------|
| vLLM Ascend version compatibility | `vllm-ascend/docs/source/conf.py` |
| vLLM Ascend source code | `vllm_ascend/` |
| **Core Modules** | |
| Ascend-specific attention | `vllm_ascend/attention/` |
| Ascend-specific executor | `vllm_ascend/worker/` |
| Ascend-specific ops | `vllm_ascend/ops/` |
| **Specialized Implementations** | |
| Ascend 310P specific | `vllm_ascend/_310p/` |
| EPLB load balancing | `vllm_ascend/eplb/` |
| XLite compiler | `vllm_ascend/xlite/` |
| **Compilation & Fusion** | |
| Graph fusion pass manager | `vllm_ascend/compilation/` |
| Compilation passes | `vllm_ascend/compilation/passes/` |
| **Quantization** | |
| Quantization methods | `vllm_ascend/quantization/` |
| ModelSlim integration | `vllm_ascend/quantization/methods/modelslim/` |
| **Distributed & KV Cache** | |
| KV transfer | `vllm_ascend/distributed/kv_transfer/` |
| Device communicators | `vllm_ascend/distributed/device_communicators/` |
| **Speculative Decoding** | |
| MTP proposer | `vllm_ascend/spec_decode/mtp_proposer.py` |
| Eagle proposer | `vllm_ascend/spec_decode/eagle_proposer.py` |
| **Utility Modules** | |
| Common utilities | `vllm_ascend/utils.py` |
| Ascend config | `vllm_ascend/ascend_config.py` |
| Platform detection | `vllm_ascend/platform.py` |
| Environment variables | `vllm_ascend/envs.py` |

---

## File Mapping Table

When upstream changes a file on the left, the corresponding vllm-ascend file on the right likely needs updating. Use this table to scope your `grep` searches.

| vLLM upstream path | vllm-ascend path | Notes |
|:---|:---|:---|
| `vllm/platforms/` | `vllm_ascend/platform.py` | Abstract methods, platform capabilities |
| `vllm/v1/worker/` | `vllm_ascend/worker/` | Worker lifecycle, model loading, execute_model |
| `vllm/v1/worker/gpu/model_runner.py` | `vllm_ascend/worker/model_runner_v1.py`, `worker/v2/model_runner.py` | Model runner is heavily overridden |
| `vllm/v1/attention/` | `vllm_ascend/attention/` | Attention backend interface |
| `vllm/model_executor/layers/attention/` | `vllm_ascend/attention/`, `vllm_ascend/ops/mm_encoder_attention.py` | Attention layer API |
| `vllm/model_executor/layers/fused_moe/` | `vllm_ascend/ops/fused_moe/` | MoE kernel interface, router |
| `vllm/distributed/` | `vllm_ascend/distributed/` | Collective ops, TP/PP, KV transfer |
| `vllm/config*.py` | `vllm_ascend/ascend_config.py` + many files that read config | Config class fields, constructor args |
| `vllm/compilation/` | `vllm_ascend/compilation/` | Compilation passes, fusion rules |
| `vllm/model_executor/models/` | `vllm_ascend/models/` | Model forward signatures |
| `vllm/model_executor/layers/quantization/` | `vllm_ascend/quantization/` | Quantization kernels, compress-tensor |
| `vllm/model_executor/layers/layernorm.py` | `vllm_ascend/ops/layernorm.py` | Normalization ops |
| `vllm/model_executor/custom_op.py` | `vllm_ascend/ops/` (any file registering custom ops) | Custom op registration |
| `vllm/v1/worker/gpu/spec_decode/` | `vllm_ascend/spec_decode/` | MTP/Eagle proposer |

Files not in this table — `docs/`, `tests/`, `benchmarks/`, `.github/` — usually don't need adaptation unless they hint at an interface change (e.g., a new test that exercises an API your code overrides).

---

## Dependency Changes

If upstream changes `pyproject.toml` or `setup.py` dependencies, check whether vllm-ascend's own dependency declarations need updating. This is easy to miss because dependency errors often surface as confusing runtime ImportErrors rather than clean install failures.

---

## Commit Reference Updates

After all steps complete, update `docs/source/conf.py`:
- `main_vllm_commit` → final target commit
- `main_vllm_tag` → new version tag (if changed)

This is the source of truth for which vLLM commit vllm-ascend is adapted to. If this is wrong, the next main2main run will re-do work or skip commits.
