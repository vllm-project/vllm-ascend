# Adapt Guide

Use this guide during the adapt phase of each main2main step. The goal is not
to copy upstream vLLM changes into vllm-ascend. The goal is to understand which
upstream contracts changed, then update the Ascend implementation that depends
on those contracts.

This file is only about adaptation decisions and code changes. Mechanical
pipeline work, such as updating the pinned vLLM commit reference, is handled by
`SKILL.md` and `scripts/update_commit_reference.py`.

## Inputs

For each step, use these files:

- `/tmp/main2main/steps/<step-id>/changed-files.txt` — file paths changed by the upstream step
- `/tmp/main2main/steps/<step-id>/upstream.patch` — full upstream diff for the step

Read `changed-files.txt` first. It is a cheap routing signal that tells you
which parts of `upstream.patch` deserve attention.

---

## Adapt Workflow

### 1. Triage Changed Files

Classify each changed upstream file using the Key Areas table below.

Files such as docs, tests, examples, benchmarks, and CI config usually do not
need vllm-ascend code changes. They still matter when they reveal an interface
shift, for example a new test calling a method that vllm-ascend overrides.

At the end of triage, write down the affected areas, such as:

- `platform`
- `worker/model_runner`
- `attention`
- `moe`
- `config`
- `distributed`
- `dependencies`

This keeps the rest of the adapt phase focused.

### 2. Inspect Only Relevant Patch Chunks

Use the affected paths from triage to find the corresponding diff chunks in
`upstream.patch`:

```bash
grep -n "^diff --git" /tmp/main2main/steps/<step-id>/upstream.patch
```

For each relevant chunk, identify the concrete contract change:

- class or abstract method added/removed
- function signature changed
- config field renamed, moved, or made required
- import path moved
- constructor arguments changed
- dependency version changed
- return type or data structure shape changed

Internal implementation changes only need adaptation when vllm-ascend overrides
the method, imports the symbol, reads the field, or depends on the behavior.

### 3. Map Upstream Symbols to vllm-ascend

Use the File Mapping Table to find likely vllm-ascend locations. Then search
for the changed symbol or field:

```bash
grep -rn '<symbol_or_field_name>' vllm_ascend/
```

Decide based on actual dependency, not path similarity. A vLLM change requires
vllm-ascend adaptation when vllm-ascend:

- subclasses or implements the changed interface
- overrides the changed method
- calls the changed function
- imports the moved symbol
- reads or writes the changed config field
- registers a backend that must satisfy the changed protocol

If none of those are true, record that no adaptation is needed for that change
and move on.

A no-op adapt conclusion only means no additional vllm-ascend code change was
identified. It does not complete the step and must not skip CI. Return to the
pipeline in `SKILL.md`; the updated commit reference still has to be verified.

### 4. Apply the Adaptation

Make the smallest vllm-ascend change that restores the contract.

Common patterns:

- Add newly required platform or worker methods.
- Update overridden method signatures to match upstream.
- Update imports after upstream moves modules.
- Read config fields from their new location.
- Update Ascend-specific kernels or wrappers when upstream call sites changed.
- Update vllm-ascend dependency declarations when upstream dependency files changed.

When a change must support both the release version and upstream main, use the
version compatibility rules from `SKILL.md`:

```python
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.19.0"):
    # release version API
else:
    # upstream main API
```

Use exact `vllm_version_is("<tag>")` checks. Do not use `hasattr()`,
`try/except`, boolean capability flags, or comparison strings to hide version
boundaries.

If Ascend cannot support a new upstream feature yet, add a narrow stub or
guarded error with a `# TODO` comment that names the unsupported feature. Avoid
silent no-ops for required interfaces; they make CI pass for the wrong reason.

### 5. Self-Check Before CI

Before starting CI, check the likely failure points:

- Grep for old import paths, old method names, and old config field names.
- Confirm overridden method signatures match upstream.
- Confirm dependency changes were mirrored when requirements or project metadata changed.
- Confirm new version guards use the current `main_vllm_tag` from `docs/source/conf.py`.
- If no code adaptation was needed, explicitly record that conclusion before
  returning to the mandatory CI step.
- Keep temporary notes, patches, and summaries in `/tmp/main2main/`, not in the repo.

Commit reference updates are not part of this self-check; the pipeline handles
them with `scripts/update_commit_reference.py` before CI.

---

## Key Areas

When these upstream paths appear in `changed-files.txt`, inspect the diff for
the listed contract changes.

1. **Platform Interface** (`vllm/platforms/`)
   - New abstract methods
   - Method signature changes
   - New platform capability flags

2. **Worker / Model Runner** (`vllm/v1/worker/`, `vllm/v1/worker/gpu/model_runner.py`)
   - New or removed parameters in `execute_model`, `load_model`, or runner initialization
   - New lifecycle methods
   - Changes to scheduler, executor, or worker result objects

3. **Attention** (`vllm/model_executor/layers/attention/`, `vllm/v1/attention/`)
   - New parameters in `forward()`
   - Attention backend interface changes
   - MLA or metadata layout changes

4. **MoE** (`vllm/model_executor/layers/fused_moe/`)
   - FusedMoE layer signature changes
   - Router or expert interface changes
   - Activation, quantization, or expert parallel behavior changes

5. **Config** (`vllm/config*.py`)
   - Field renames or moves between config classes
   - New required fields
   - Constructor changes

6. **Distributed** (`vllm/distributed/`)
   - Collective op interface changes
   - KV transfer protocol changes
   - Device communicator changes

7. **Speculative Decoding** (`vllm/v1/worker/gpu/spec_decode/`, `vllm/config/speculative.py`)
   - Import path changes
   - Config field changes
   - Proposer interface changes

8. **Compilation** (`vllm/compilation/`)
   - Pass manager interface changes
   - Required pass changes
   - Pass registration changes

9. **Quantization** (`vllm/model_executor/layers/quantization/`)
   - Quantization config changes
   - Kernel wrapper changes
   - `compress-tensor` or weight loading behavior changes

10. **Models** (`vllm/model_executor/models/`)
    - Forward signature changes for models vllm-ascend overrides
    - New model architectures that need Ascend-specific support
    - Changes to model loader assumptions

11. **Dependencies** (`requirements*`, `constraints*`, `pyproject.toml`, `setup.py`, `setup.cfg`, `uv.lock`, `poetry.lock`)
    - Version bumps required by upstream API changes
    - New runtime dependencies
    - Removed or renamed dependencies

---

## File Mapping Table

Use this table after identifying a changed upstream symbol. It points to likely
vllm-ascend locations, not guaranteed locations. Always grep for the symbol.

| vLLM upstream path | vllm-ascend path | What to check |
|:---|:---|:---|
| `vllm/platforms/` | `vllm_ascend/platform.py` | Abstract methods, platform capabilities |
| `vllm/v1/worker/` | `vllm_ascend/worker/` | Worker lifecycle, model loading, `execute_model` |
| `vllm/v1/worker/gpu/model_runner.py` | `vllm_ascend/worker/model_runner_v1.py`, `vllm_ascend/worker/v2/model_runner.py` | Runner initialization and execution |
| `vllm/v1/attention/` | `vllm_ascend/attention/` | Backend interface and metadata |
| `vllm/model_executor/layers/attention/` | `vllm_ascend/attention/`, `vllm_ascend/ops/mm_encoder_attention.py` | Attention wrappers and kernels |
| `vllm/model_executor/layers/fused_moe/` | `vllm_ascend/ops/fused_moe/` | MoE kernel interface, router, experts |
| `vllm/distributed/` | `vllm_ascend/distributed/` | Collective ops, TP/PP, KV transfer |
| `vllm/config*.py` | `vllm_ascend/ascend_config.py`, plus call sites under `vllm_ascend/` | Config fields and constructor args |
| `vllm/compilation/` | `vllm_ascend/compilation/` | Passes, fusion rules, registration |
| `vllm/model_executor/models/` | `vllm_ascend/models/` | Model forward signatures and loaders |
| `vllm/model_executor/layers/quantization/` | `vllm_ascend/quantization/` | Quantization methods and kernels |
| `vllm/model_executor/layers/layernorm.py` | `vllm_ascend/ops/layernorm.py` | LayerNorm op interface |
| `vllm/model_executor/custom_op.py` | `vllm_ascend/ops/` | Custom op registration |
| `vllm/v1/worker/gpu/spec_decode/` | `vllm_ascend/spec_decode/` | MTP/Eagle proposer interfaces |
| `requirements*`, `constraints*`, `pyproject.toml`, `setup.py`, `setup.cfg` | Matching dependency files in vllm-ascend | Dependency versions |

---

## vllm-ascend Directory Reference

Use this as a quick orientation map when grep returns many results.

| Area | Path |
|:---|:---|
| Version metadata | `docs/source/conf.py` |
| Core source | `vllm_ascend/` |
| Platform | `vllm_ascend/platform.py` |
| Attention | `vllm_ascend/attention/` |
| Worker / executor | `vllm_ascend/worker/` |
| Models | `vllm_ascend/models/` |
| Ops | `vllm_ascend/ops/` |
| Distributed / KV transfer | `vllm_ascend/distributed/` |
| Compilation | `vllm_ascend/compilation/` |
| Quantization | `vllm_ascend/quantization/` |
| Speculative decoding | `vllm_ascend/spec_decode/` |
| Ascend config | `vllm_ascend/ascend_config.py` |
| Environment variables | `vllm_ascend/envs.py` |
| Utilities | `vllm_ascend/utils.py` |
| 310P-specific code | `vllm_ascend/_310p/` |
| EPLB | `vllm_ascend/eplb/` |
| XLite | `vllm_ascend/xlite/` |
