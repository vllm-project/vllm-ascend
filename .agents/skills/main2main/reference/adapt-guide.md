# Adapt Guide — How to Analyze Upstream Changes

This is the reference material for the adapt phase. Come here when you need the file mapping table, subsystem-specific guidance, or the concrete commands for analyzing upstream diffs. The thinking framework is in SKILL.md — this document is for lookup, not for reading end-to-end.

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

## Subsystem Notes

When you see changes in one of these areas, here are the things most likely to need attention. This list is not exhaustive — if you encounter patterns not covered here, use your judgment and update this section for future runs.

### Platform (`vllm/platforms/`)

AscendPlatform must implement every abstract method in the base class — miss one and you get `Can't instantiate abstract class`.

- New abstract methods: check base class for signature + return type, implement in `vllm_ascend/platform.py`.
- Signature changes on existing methods: grep for the method name in `vllm_ascend/platform.py`.

### Worker / Model Runner (`vllm/v1/worker/`)

The model runner is heavily overridden in vllm-ascend. Changes here frequently require adaptation.

- `execute_model()` signature or flow changes: trace through `vllm_ascend/worker/model_runner_v1.py`.
- New lifecycle methods or input preparation changes: check if the worker base class calls them.

### Attention (`vllm/model_executor/layers/attention/`)

Attention backends are replaced entirely on Ascend.

- `forward_oot()` parameter changes: this is vllm-ascend's primary extension point — any upstream signature change breaks it.

### MoE (`vllm/model_executor/layers/fused_moe/`)

FusedMoE is fully reimplemented for Ascend NPU. Interface changes cascade into `vllm_ascend/ops/fused_moe/`.

### Config (`vllm/config*.py`)

Config changes are deceptively dangerous — a single field rename can break 5+ files. grep `vllm_config.<class>.<field>` across all of `vllm_ascend/` to find every access point.

### Distributed (`vllm/distributed/`)

Changes here may affect `vllm_ascend/distributed/`, including KV transfer and device communicators.

### Speculative Decoding (`vllm/v1/worker/gpu/spec_decode/`)

Import path changes are common in this area — spec decode is actively refactored upstream. Check `vllm_ascend/spec_decode/`.

---

## Analysis Commands

These are the concrete git commands for investigating upstream changes. Run them against the local vLLM repo.

```bash
# Overview of all changed files
git diff <start>..<end> --name-only

# Changes in a specific subsystem
git diff <start>..<end> -- vllm/platforms/
git diff <start>..<end> -- vllm/model_executor/layers/fused_moe/
git diff <start>..<end> -- vllm/model_executor/layers/attention/

# Commit-level history (useful for understanding intent)
git log --oneline <start>..<end>

# Find breaking changes in commit messages
git log --oneline <start>..<end> | grep -iE "(refactor|breaking|api|rename|remove|deprecate)"

# Detect renamed/moved files
git diff <start>..<end> --name-status | grep -E "^R"

# Detailed diff for a specific file
git diff <start>..<end> -- <FILE_PATH>
```

---

## Dependency Changes

If upstream changes `pyproject.toml` or `setup.py` dependencies, check whether vllm-ascend's own dependency declarations need updating. This is easy to miss because dependency errors often surface as confusing runtime ImportErrors rather than clean install failures.

---

## Commit Reference Updates

After all steps complete, update `docs/source/conf.py`:
- `main_vllm_commit` → final target commit
- `main_vllm_tag` → new version tag (if changed)

This is the source of truth for which vLLM commit vllm-ascend is adapted to. If this is wrong, the next main2main run will re-do work or skip commits.
