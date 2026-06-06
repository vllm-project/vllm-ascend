# Periodic CI Refactoring Execution Plan

## 1. Goal

Refactor the current nightly / weekly CI into a unified periodic CI routing system.

The target architecture is:

```text
schedule_config.yaml
  -> parse_schedule_config.py
  -> unified PeriodicCase list
  -> framework routers
  -> framework-specific matrices
  -> reusable workflow / pytest execution
```

The refactoring should support all current nightly test types:

```text
model tests
accuracy tests
ops tests
single-node tests
multi-node internal tests
multi-node external tests
```

The workflow should no longer hardcode large test matrices.  
New cases should be added by placing files under `tests/e2e/schedule/` and registering them in `schedule_config.yaml`.

---

## 1A. Review Findings & Resolutions

This plan was hardened after a review (see `modify.md`). Each finding below is now a
binding requirement, and its resolution is reflected in the sections that follow.

| # | Finding (from review) | Resolution | Status |
|---|---|---|---|
| 1 | Parser was a `_classify()` scope-splitter, not `PeriodicCase` + routers | Parser builds a `PeriodicCase` per file, then dispatches through a `ROUTERS` registry (§6, §13) | Done |
| 2 | Workflow consumed `ops_matrix` but parser never emitted it (`fromJSON('')` crash) | Parser **always** writes all four matrices, empty as `[]` (§15) | Done |
| 3 | Ops had no independent route; entries were dict-style with inline `chip`/`name` | `OpsRouter` emits `ops_matrix`; config entries are plain paths (§10.3, §14.2) | Done |
| 4 | accuracy still lived under `tests/e2e/accuracy/` and was matched loosely | Files moved to `tests/e2e/schedule/accuracy/`; parser **rejects** `tests/e2e/accuracy/` and requires the `tests/e2e/schedule/` prefix (§7, §21) | Done |
| 5 | `external_dp` detected by a directory part | Detected **only** by filename stem containing `external_dp`; the `external_dp/` directory is deleted (§11) | Done |
| 6 | `build-image` passed `branch_tag` as the checkout branch | `build-image` passes `branch_ref`; `branch_tag` is used only for image tag / artifact names (§18) | Done |
| 7 | `selected_cases_summary` was consumed but never produced | Parser emits `selected_cases_summary` (name, framework, route, chip, runner, path, multi_node_type) (§15) | Done |
| 8 | Parser accepted `chip` / `npu_num` / `extra_components` overrides | Parser accepts **plain path strings only**; everything is inferred. `extra_components` is fixed to `false` this version (§5) | Done |
| 9 | `_e2e_periodic_ops.yaml` had a broken multi-line `run:` | Install/run steps use valid single-line / block scalars; container + checkout mirror the model e2e setup (§20) | Done |
| 10 | Runner rule table was incomplete (e.g. `four_node + a2`) | `_NPU_NUM` enumerates every `(resource_type, chip, resource_num)`, including `four_node + a2 -> 0` (LWS) (§12) | Done |
| 11 | Per-model configs in `tests/e2e/models/configs/` should not be symlinked | The group `model_list` YAMLs moved to `tests/e2e/schedule/accuracy/`. **Superseded by §1D:** `tests/e2e/models/` (configs *and* the test harness) has since been physically moved/deleted — the harness now lives at `tests/e2e/schedule/scripts/accuracy/` and the workflow reads from there. | Superseded by §1D |

Self-optimizations applied beyond the review:

```text
- Removed orphaned source files left by the move (old tests/e2e/accuracy/ tree
  and the old GLM external_dp/ config) so no duplicate/dead config remains.
- accuracy group YAMLs are plain top-level lists of model names; the parser
  accepts both a bare list and the legacy {model_list: [...]} mapping (§14.1).
- All file I/O in the parser is UTF-8 explicit (Windows-safe for model YAMLs
  that contain non-ASCII characters).
```

---

## 1B. Script Layout Migration & Routing Round

A follow-up review ("Script Layout Without `common/`") added these requirements,
all now implemented:

| # | Requirement | Resolution | Status |
|---|---|---|---|
| 1 | No `common/` directories; shared files in nearest parent | Scripts consolidated under `tests/e2e/schedule/scripts/`; shared helpers placed directly in `multi_node/` etc. — no `common/` (§3A) | Done |
| 2 | Move periodic execution scripts out of `tests/e2e/nightly/...` | All scripts moved; old `tests/e2e/nightly/` tree deleted (§3A) | Done |
| 3 | Update all references (workflows, imports, path/template constants) | Module paths, run.sh test paths, lws default config path, both reusable workflows, and doc comments updated; repo grep for `nightly` is clean in code (§3A) | Done |
| 4 | 310p supported like a2/a3 via filename/dir token | `310p` is a first-class chip, checked before a2/a3 (§9, §12) | Done |
| 5 | Directory entries expanded before routing | ops + accuracy dirs → per-chip groups; model dirs → per-file (§10.4; accuracy grouping refined in round 3) | Done |
| 6 | Move `test_recurrent_gated_delta_rule_v310.py` to the right place | Now at `tests/e2e/schedule/ops/one_card/` (ops case home) | Done |
| 7 | Invariants intact: external_dp by filename, accuracy YAML = config, PeriodicCase + Router | Unchanged and re-verified by the test suite | Done |

---

## 1C. Direct Accuracy YAML Integration Round

A third review ("Direct Accuracy YAML Integration") replaced the indirect
model-list accuracy flow with direct config paths. All implemented:

| # | Requirement | Resolution | Status |
|---|---|---|---|
| 1 | Real accuracy YAMLs live under `accuracy/<resource>/<chip>/` | Configs copied from `tests/e2e/models/configs/`; chip is the explicit `a2/a3/310p` dir (§3, §2-migration) | Done |
| 2 | Parser emits `config_paths`, not `model_list` | `PeriodicCase.config_paths`; `AccuracyRouter` outputs `config_paths`; `model_list` + `_load_model_list()` removed (§6, §14.1) | Done |
| 3 | accuracy `.yaml`→`config_paths=[path]`, `.py`→`tests`, else error | Implemented in `_parse_to_case` (§14.1) | Done |
| 4 | Accuracy directory grouped by chip (not per-file) | `_expand_directory` groups accuracy files by chip into one case per chip (§10.4) | Done |
| 5 | Old list-based group YAML rejected | `_validate_accuracy_config()` raises on non-dict YAML; group files deleted (§14.1) | Done |
| 6 | Workflow runs `pytest --config <config_path>` directly | `_e2e_nightly_single_node_models.yaml` consumes `config_paths`; effective-config-paths + per-config run + stem-based summary; `model_list` input removed | Done |
| 7 | schedule_config.yaml registers dirs/files, not groups | accuracy entries are `accuracy/<resource>/<chip>/` dirs | Done |

The old path (`group YAML -> model_list -> tests/e2e/models/configs/${model}.yaml`)
no longer exists. (Superseded by §1D: the `tests/e2e/models/` source tree has since
been deleted and the accuracy harness moved under
`tests/e2e/schedule/scripts/accuracy/`.)

---

## 1D. Accuracy Framework Consolidation & Cleanup Round

A fourth round consolidated the accuracy test framework under
`tests/e2e/schedule/` and removed residue left by the earlier rounds. This round
intentionally expands the §2 scope (which had deferred "accuracy execution logic"):
the accuracy harness is now a first-class member of the schedule scripts tree.

| # | Change | Resolution | Status |
|---|---|---|---|
| 1 | `test_qwen3_30b_acc.py` was a hand-written pytest accuracy case | Converted to a single_node YAML at `model/Qwen/four_card/Qwen3-30B-A3B-W8A8-eagle3-mooncake.yaml`; runs via `test_single_node.py` (TP1 + TP4 cases) | Done |
| 2 | single_node framework lacked a Mooncake sidecar | `single_node_config.py` gained a `mooncake` field + generalized `DEFAULT_PORT` assignment; `test_single_node.py` wraps the server in `MooncakeLauncher` via `ExitStack` and writes `mooncake.json` | Done |
| 3 | single_node loader could not resolve a full-path `CONFIG_YAML_PATH` | `_load_yaml` uses the path directly when it resolves, else joins `CONFIG_BASE_PATH` (mirrors multi_node `load_yaml_mapping`) | Done |
| 4 | accuracy harness still lived at `tests/e2e/models/` | Moved `test_{lm_eval,asr,rm}_eval_correctness.py` + `conftest.py` + `report_template.md` to `tests/e2e/schedule/scripts/accuracy/`; `_e2e_nightly_single_node_models.yaml` repointed | Done |
| 5 | `.py`-accuracy → `tests` route was unused | Removed from `_parse_to_case` (accuracy is YAML-only now), dropped `tests` from `AccuracyRouter`, removed the `tests` input + pytest step from the accuracy workflow | Done |
| 6 | residual / dead trees | Deleted `tests/e2e/models/` (configs already mirrored in `accuracy/`), `tests/e2e/weekly/single_node/configs/` (5 superseded YAMLs), and the empty `model/Qwen/one_card/` | Done |
| 7 | docs + model-adapter skill pointed at old paths | Repointed accuracy-config refs to `schedule/accuracy/<resource>/<chip>/`, harness refs to `schedule/scripts/accuracy/`, and stale `tests/e2e/nightly/...` multi_node/single_node doc refs to `schedule/scripts/...` | Done |

The 4 configs that lived only in `tests/e2e/models/configs/` (`gemma-3-4b-it`,
`internlm3-8b-instruct`, `Qwen2.5-Math-RM-72B`, `Qwen3-ASR-1.7B`) plus
`accuracy.txt` / `accuracy_groups_a2.json` were dropped (unregistered anywhere);
the 18 active configs were verified **byte-identical** to their `schedule/accuracy/`
copies before deletion. The zh `multi_node_test.po` translation is left for i18n
regeneration from the updated `.md`.

---

## 2. Scope

This task refactors the CI scheduling and routing layer.

Do refactor:

```text
.github/workflows/schedule_periodic_test.yaml
.github/workflows/scripts/schedule_config.yaml
.github/workflows/scripts/parse_schedule_config.py
tests/e2e/schedule/ directory layout
```

Do not refactor:

```text
existing single-node test framework
existing multi-node test framework
existing LWS execution logic
existing internal / external DP runtime logic
existing accuracy execution logic
existing model test implementation
```

Ops may get its own lightweight reusable workflow if needed.

---

## 3. Target Directory Layout

All periodic CI cases must be placed under:

```text
tests/e2e/schedule/
```

Expected layout — the three frameworks (`model`, `accuracy`, `ops`) sit at the same
depth so the framework is always the 4th path segment:

```text
tests/e2e/schedule/
  model/
    DeepSeek/
      one_card/
      two_card/
      four_card/
      eight_card/
      one_node/
      two_node/
      four_node/

    Qwen/
      one_card/
      two_card/
      four_card/
      eight_card/
      one_node/
      two_node/
      four_node/

    GLM/
      ...   (same resource_dir set)

    Kimi/ MiniMax/ Hy/ ...

  accuracy/
    one_card/
      a2/   <Model>.yaml ...   (real executable accuracy configs)
      a3/   <Model>.yaml ...
      310p/ <Model>-310p.yaml ...
    two_card/
      a2/ ...
    four_card/
      a2/ ...

  ops/
    one_card/
    two_card/
    four_card/
    eight_card/
    one_node/
```

Examples:
- model:    `tests/e2e/schedule/model/Kimi/one_node/Kimi-K2.5.yaml`
- accuracy: `tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml`

Accuracy configs are real executable YAMLs (not model-list groups); chip is the
explicit `a2/a3/310p` directory. A directory entry is grouped into one job per chip
(`config_paths` list). See §1C and §14.1.

Important rules:

```text
1. Model cases must be under tests/e2e/schedule/model/<model_family>/<resource_dir>/.
2. The <model_family> segment must not be a resource directory name.
3. accuracy must be under tests/e2e/schedule/accuracy/.
4. ops must be under tests/e2e/schedule/ops/.
5. Do not use tests/e2e/accuracy/.
6. The bare layout tests/e2e/schedule/<Family>/ (without the model/ layer) is rejected.
```

---

## 3A. Execution Script Layout (no `common/`)

The periodic execution scripts (runners, configs, helpers) live under a single
`scripts/` tree, separate from the case files. Shared helpers go in the **nearest
suitable parent directory** — there are **no `common/` directories**.

```text
tests/e2e/schedule/scripts/
  __init__.py
  multi_node/
    __init__.py
    utils.py
    benchmark_results.py
    lws.yaml.jinja2
    run.sh
    internal_dp/
      __init__.py
      multi_node_config.py
      test_multi_node.py
      utils.py
    external_dp/
      __init__.py
      external_dp_config.py
      runtime.py
      test_external_dp.py
      utils.py
      config/
        template.md
  single_node/
    __init__.py
    single_node_config.py
    test_single_node.py
    GUIDE_AND_TEMPLATE.md
  accuracy/                       (added in §1D — accuracy harness home)
    __init__.py
    conftest.py
    test_lm_eval_correctness.py
    test_asr_eval_correctness.py
    test_rm_eval_correctness.py
    report_template.md
```

Migration rules:

```text
1. Move scripts from tests/e2e/nightly/... (and weekly/accuracy) into
   tests/e2e/schedule/scripts/, collapsing the per-tier scripts/ subdirectories.
2. Do NOT create scripts/common/ or scripts/multi_node/common/.
3. Update every reference after the move:
   - workflow paths   (_e2e_nightly_multi_node.yaml, _e2e_nightly_single_node.yaml)
   - python imports   (tests.e2e.nightly.* -> tests.e2e.schedule.scripts.*)
   - path constants   (run.sh test paths, *_config.py base paths)
   - template paths   (lws.yaml.jinja2 default config path)
4. schedule/ stays a namespace package (no __init__.py); scripts/ and its
   subpackages are regular packages (with __init__.py), mirroring the old
   nightly(ns)/multi_node(pkg)/scripts(pkg) shape shifted under schedule.
```

Module-path mapping applied:

```text
tests.e2e.nightly.multi_node.scripts                -> tests.e2e.schedule.scripts.multi_node
tests.e2e.nightly.multi_node.internal_dp.scripts    -> tests.e2e.schedule.scripts.multi_node.internal_dp
tests.e2e.nightly.multi_node.external_dp.scripts    -> tests.e2e.schedule.scripts.multi_node.external_dp
tests.e2e.nightly.single_node.models.scripts        -> tests.e2e.schedule.scripts.single_node
```

Case files vs. scripts: `tests/e2e/schedule/{model,accuracy,ops}/` hold the **case
files** registered in `schedule_config.yaml`; `tests/e2e/schedule/scripts/` holds the
**execution infrastructure**. The ops pytest conftest lives with the ops cases at
`tests/e2e/schedule/ops/conftest.py`. The 310p kernel test moved to its ops case home
`tests/e2e/schedule/ops/one_card/test_recurrent_gated_delta_rule_v310.py`.

---

## 4. Resource Directory Rules

Only English-form resource directories are supported.

Valid directories:

```text
one_card
two_card
four_card
eight_card

one_node
two_node
four_node
```

Invalid directories:

```text
1_card
2_card
4_card
8_card
1_node
2_node
4_node
```

The parser must not support numeric forms.  
If a numeric form is found, it should fail with a clear error.

---

## 5. schedule_config.yaml

Path:

```text
.github/workflows/scripts/schedule_config.yaml
```

It is the single registry for periodic CI cases.

Example:

```yaml
periodic_tests:
  - name: nightly-main
    cron: "45 15 * * *"
    files:
      - tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.yaml
      - tests/e2e/schedule/model/GLM/two_node/GLM5-W8A8-external_dp.yaml
      - tests/e2e/schedule/model/Qwen/four_card/Qwen3-A2.yaml
      - tests/e2e/schedule/accuracy/one_card/accuracy-group-a2.yaml
      - tests/e2e/schedule/ops/one_card/test_custom_op_a2.py

  - name: weekly-main
    cron: "45 15 * * 1"
    files:
      - tests/e2e/schedule/model/DeepSeek/four_node/DeepSeek-V3-weekly.yaml
      - tests/e2e/schedule/accuracy/four_card/qwen-accuracy-a2.yaml

  - name: manual
    cron: workflow_dispatch
    files:
      - tests/e2e/schedule/model/DeepSeek/one_node/DeepSeek-V3.yaml
      - tests/e2e/schedule/ops/one_card/test_custom_op_a2.py
```

Rules:

```text
1. cron uses UTC time.
2. name is only for readability and logging.
3. files should be plain paths.
4. Avoid adding execution details into schedule_config.yaml.
```

Do not put these fields into normal entries:

```text
chip
npu_num
runner
framework
route
multi_node_type
extra_components
```

These should be inferred from path, filename, and runner_label.json.

**Hardened (review finding #8):** the parser accepts a plain path **string** per
entry and nothing else. Dict-style entries (`- tests: ...`, `chip:`, `name:`) are
no longer supported. `extra_components` is fixed to `false` in this version; if a
case ever needs it, add it as an inferred property or a new router field rather
than a free-form override in `schedule_config.yaml`.

---

## 6. Unified PeriodicCase

Every entry from `schedule_config.yaml` should first be parsed into a unified data class or dictionary.

Recommended structure:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

Framework = Literal["model", "accuracy", "ops"]
Route = Literal["single_node", "multi_node", "accuracy", "ops"]
Chip = Literal["a2", "a3"]
ResourceType = Literal["card", "node"]
MultiNodeType = Literal["internal", "external"]

@dataclass(frozen=True)
class PeriodicCase:
    name: str
    path: str
    path_obj: Path

    framework: Framework
    route: Route

    chip: Chip
    resource_type: ResourceType
    resource_num: int
    resource_dir: str

    runner: str

    family: Optional[str] = None
    multi_node_type: Optional[MultiNodeType] = None

    config_path: Optional[str] = None        # single YAML (model single/multi-node)
    config_paths: Optional[list[str]] = None  # direct accuracy config YAMLs
    tests: Optional[str] = None

    size: Optional[str] = None
    replicas: Optional[str] = None

    raw: Optional[dict] = None
```

Note: `model_list` was removed in the direct-accuracy round (§1C). Accuracy cases now
carry `config_paths`; the accuracy YAMLs are real executable configs, not model-name
lists.

All later routing should operate on `PeriodicCase`, not raw strings.

---

## 7. Framework Detection

Framework must be inferred from the 4th path segment:

```text
tests/e2e/schedule/model/<family>/... -> model (family = 5th segment)
tests/e2e/schedule/accuracy/...       -> accuracy
tests/e2e/schedule/ops/...            -> ops
```

Any other 4th segment is an error. A bare `tests/e2e/schedule/<family>/` (without the
`model/` layer) is rejected, and the family segment must not be a resource-dir name.

Examples:

```text
tests/e2e/schedule/accuracy/one_card/xxx-a2.yaml
  -> framework = accuracy

tests/e2e/schedule/ops/one_card/test_xxx_a2.py
  -> framework = ops

tests/e2e/schedule/model/DeepSeek/one_node/xxx.yaml
  -> framework = model
  -> family = DeepSeek

tests/e2e/schedule/model/Kimi/one_node/Kimi-K2.5.yaml
  -> framework = model
  -> family = Kimi

tests/e2e/schedule/Kimi/one_node/Kimi-K2.5.yaml
  -> ERROR (missing model/ layer)
```

---

## 8. Resource Detection

Use the following mapping:

```python
RESOURCE_DIRS = {
    "one_card": ("card", 1),
    "two_card": ("card", 2),
    "four_card": ("card", 4),
    "eight_card": ("card", 8),
    "one_node": ("node", 1),
    "two_node": ("node", 2),
    "four_node": ("node", 4),
}
```

Rules:

```text
1. Each case path must contain exactly one valid resource directory.
2. If no resource directory is found, fail.
3. If more than one resource directory is found, fail.
4. Numeric forms such as 1_card or 2_node must fail.
```

---

## 9. Chip Detection

Chip should be inferred from path or filename.

Rules:

```text
path or filename contains 310 / 310p / v310 -> 310p
path or filename contains a2 / A2           -> a2
path or filename contains a3 / A3           -> a3
otherwise -> default to a3
```

Recommended regex behavior:

```text
Match 310 / 310p / v310 only as a separated token (checked first).
Match a2 / A2 only as a separated token.
Match a3 / A3 only as a separated token.
```

**Hardened:** 310p is a first-class chip (review requirement: "310p is supported like
a2/a3 through filename or directory token detection"). It is checked before a2/a3 and
matched by `(?<![A-Za-z0-9])v?310p?(?![A-Za-z0-9])`, so `_310`, `310p`, and `v310`
tokens all resolve to 310p while `310B` inside a model name does not.

Examples:

```text
Qwen3-A2.yaml                          -> a2
Qwen3_a2.yaml                          -> a2
DeepSeek-V3.yaml                       -> a3
DeepSeek-V3-A3.yaml                    -> a3
test_causal_conv1d_310.py              -> 310p
test_recurrent_gated_delta_rule_v310.py -> 310p
ops/310p/one_card/test_foo.py          -> 310p (directory token)
```

---

## 10. Route Rules

### 10.1 Model Framework

Model cases are routed as follows:

```text
model + one_card   -> single_node
model + two_card   -> single_node
model + four_card  -> single_node
model + eight_card -> single_node

model + one_node   -> single_node
model + two_node   -> multi_node
model + four_node  -> multi_node
```

### 10.2 Accuracy Framework

Accuracy cases are routed as follows:

```text
tests/e2e/schedule/accuracy/<num>_card/*.yaml -> accuracy
```

First version should only support card-based accuracy cases.

If this appears:

```text
tests/e2e/schedule/accuracy/one_node/
```

the parser should fail.

### 10.3 Ops Framework

Ops cases are routed as follows:

```text
tests/e2e/schedule/ops/<num>_card/*.py -> ops
tests/e2e/schedule/ops/<num>_card/     -> ops
tests/e2e/schedule/ops/one_node/*.py   -> ops
tests/e2e/schedule/ops/one_node/       -> ops
```

Ops should not be mixed into the model single-node matrix.  
Ops should have its own `ops_matrix` and its own job or reusable workflow.

### 10.4 Directory Entries & Expansion

`schedule_config.yaml` supports both **file** and **directory** entries; directory
entries are **expanded before routing** (an entry is a directory if it ends with `/`,
exists as a directory, or has no file extension).

```text
ops directory       -> group the contained test_*.py files by detected chip;
                       emit ONE ops case per chip. tests = space-separated file
                       list, runner resolved from (chip, resource). Recurses into
                       subdirs (e.g. ops/one_card/triton/).
accuracy directory  -> group the contained *.yaml configs by detected chip;
                       emit ONE accuracy case per chip. config_paths = the chip's
                       files, name = "<resource_dir>-<chip>" (§1C, §14.1).
model directory     -> expand per file (each *.yaml routes independently as its
                       own config_path), since each config drives one run.
```

Examples (per-chip grouping):

```text
tests/e2e/schedule/ops/one_card/  ->
  one-card-a3   (chip a3,   runner linux-aarch64-a3-1,   tests = many test_*.py)
  one-card-310p (chip 310p, runner linux-aarch64-310p-1, tests = the *_310 / *_v310 files)

tests/e2e/schedule/accuracy/one_card/a2/  ->
  one_card-a2   (chip a2, runner linux-aarch64-a2b3-1, config_paths = the a2 configs)
```

The reusable ops workflow runs `pytest -s ${tests}` (unquoted) so a space-separated
group list splits into separate pytest arguments. The accuracy workflow iterates
`config_paths` and runs `pytest --config <config_path>` per file. A directory with no
routable files fails with a clear error.

---

## 11. Multi-node Internal / External Rules

This rule only applies to:

```text
framework = model
route = multi_node
```

External DP detection:

```text
filename contains external_dp -> external
otherwise -> internal
```

Important:

```text
1. Match external_dp, not external.
2. Do not use external as the keyword.
3. Do not depend on an external_dp directory.
4. Only check the filename (stem).
```

**Hardened (review finding #5):** detection uses the file **stem** only —
`"external_dp" in Path(path).stem`. The old `two_node/external_dp/` directory has
been deleted and is never consulted. The migrated file is
`tests/e2e/schedule/model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml`.

Examples:

```text
GLM5-W8A8-external_dp.yaml -> external
GLM5-W8A8-external.yaml    -> internal
GLM5-W8A8.yaml             -> internal
```

---

## 12. Runner Selection

Runner should be selected using:

```text
chip + resource_type + resource_num
```

If `runner_label.json` currently maps by `chip + npu_num`, convert resource information into `npu_num`.

Recommended mapping:

```text
one_card   -> npu_num = 1
two_card   -> npu_num = 2
four_card  -> npu_num = 4
eight_card -> npu_num = 8

one_node + a3 -> npu_num = 16
one_node + a2 -> npu_num = 8

two_node / four_node -> LWS orchestration runner
```

Expected runner behavior:

```text
one_card + a3 -> linux-aarch64-a3-1
two_card + a3 -> linux-aarch64-a3-2
four_card + a3 -> linux-aarch64-a3-4
eight_card + a3 -> linux-aarch64-a3-8
one_node + a3 -> linux-aarch64-a3-16

one_card + a2 -> linux-aarch64-a2b3-1
two_card + a2 -> linux-aarch64-a2b3-2
four_card + a2 -> linux-aarch64-a2b3-4
one_node + a2 -> linux-aarch64-a2b3-8

one_card + 310p  -> linux-aarch64-310p-1
two_card + 310p  -> linux-aarch64-310p-2
four_card + 310p -> linux-aarch64-310p-4

two_node / four_node -> LWS runner, for example linux-aarch64-a3-0
```

Do not hardcode runner labels directly in workflow matrix entries.  
Resolve them in the parser from `runner_label.json`.

**Hardened (review finding #10):** the parser keys runner resolution on a complete
`(resource_type, chip, resource_num) -> npu_num` table, with no implicit fallbacks
for multi-node. Every supported combination is enumerated explicitly:

```python
_NPU_NUM = {
    ("card", "a3", 1): 1,  ("card", "a3", 2): 2,  ("card", "a3", 4): 4,  ("card", "a3", 8): 8,
    ("node", "a3", 1): 16, ("node", "a3", 2): 0,  ("node", "a3", 4): 0,
    ("card", "a2", 1): 1,  ("card", "a2", 2): 2,  ("card", "a2", 4): 4,  ("card", "a2", 8): 8,
    ("node", "a2", 1): 8,  ("node", "a2", 2): 0,  ("node", "a2", 4): 0,   # four_node + a2 explicit
}
```

`npu_num = 0` resolves to an LWS / CPU orchestration runner. `(a2, 0)` maps to
`linux-amd64-cpu-8-hk` via `_SPECIAL_RUNNERS`; `(a3, 0)` maps to `linux-aarch64-a3-0`
from `runner_label.json`. An unknown combination raises a clear error instead of
silently defaulting.

---

## 13. Router Architecture

Do not implement routing as one large chain of scattered if/else blocks.

Use the following structure:

```text
raw file entries
  -> PeriodicCase list
  -> framework routers
  -> framework-specific matrices
```

Recommended router interface:

```python
class BaseRouter:
    name: str
    output_name: str

    def match(self, case: PeriodicCase) -> bool:
        raise NotImplementedError

    def to_matrix_item(self, case: PeriodicCase) -> dict:
        raise NotImplementedError
```

Required routers:

```text
AccuracyRouter
OpsRouter
ModelSingleNodeRouter
ModelMultiNodeRouter
```

Router registry:

```python
ROUTERS = [
    AccuracyRouter(),
    OpsRouter(),
    ModelSingleNodeRouter(),
    ModelMultiNodeRouter(),
]
```

---

## 14. Router Output

### 14.1 AccuracyRouter

Match:

```python
case.framework == "accuracy"
```

Output example (direct accuracy config paths — §1C):

```json
{
  "name": "one_card-a2",
  "chip": "a2",
  "runner": "linux-aarch64-a2b3-1",
  "config_paths": [
    "tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B.yaml",
    "tests/e2e/schedule/accuracy/one_card/a2/Qwen3-8B-W8A8.yaml"
  ],
  "tests": ""
}
```

Accuracy parsing rules (per `_parse_to_case` / `_expand_directory`):

```text
single .yaml file -> config_paths = [path]   (validated as a dict config)
other extension   -> ValueError              (accuracy is YAML-only; see §1D)
directory entry   -> group the contained *.yaml by detected chip;
                     one case per chip, config_paths = the chip's files,
                     name = "<resource_dir>-<chip>" (e.g. one_card-a2)
```

(The `.py`-accuracy → `tests` route was removed in §1D — no accuracy `.py` cases
remain.)

`AccuracyRouter.to_matrix_item` emits:

```python
"config_paths": case.config_paths or ([case.config_path] if case.config_path else []),
```

No `model_list` is produced.

**Validation (`_validate_accuracy_config`):** accuracy YAMLs must be a dict (a real
executable config). A bare model-name list (the old group format) raises
`ValueError: "... must be a YAML dict. Old model-list group YAML is no longer
supported."` so stale group files cannot be silently mistaken for configs. Validation
is skipped only when the file is absent (the parser may run before a path materializes).

**Migration (resolves the former finding #11):** the real configs were **copied** from
`tests/e2e/models/configs/<model>.yaml` into
`tests/e2e/schedule/accuracy/<resource_dir>/<chip>/<model>.yaml`, and the old
`accuracy-group-*.yaml` files were deleted. The indirect path
(`group YAML -> model_list -> tests/e2e/models/configs/${model}.yaml`) is gone; the
execution workflow now runs `pytest --config <config_path>` directly. **§1D update:**
the `tests/e2e/models/` source tree was subsequently deleted and the accuracy harness
(`test_*_eval_correctness.py`, `conftest.py`, `report_template.md`) moved to
`tests/e2e/schedule/scripts/accuracy/`.

### 14.2 OpsRouter

Match:

```python
case.framework == "ops"
```

Output example:

```json
{
  "name": "test_custom_op_a2",
  "chip": "a2",
  "runner": "linux-aarch64-a2b3-1",
  "resource_type": "card",
  "resource_num": 1,
  "tests": "tests/e2e/schedule/ops/one_card/test_custom_op_a2.py"
}
```

### 14.3 ModelSingleNodeRouter

Match:

```python
case.framework == "model" and case.route == "single_node"
```

Output example:

```json
{
  "name": "Qwen3-235B-A22B-W8A8",
  "chip": "a3",
  "runner": "linux-aarch64-a3-16",
  "resource_type": "node",
  "resource_num": 1,
  "config_path": "tests/e2e/schedule/model/Qwen/one_node/Qwen3-235B-A22B-W8A8.yaml"
}
```

### 14.4 ModelMultiNodeRouter

Match:

```python
case.framework == "model" and case.route == "multi_node"
```

Output example:

```json
{
  "name": "GLM5-W8A8-external_dp",
  "chip": "a3",
  "runner": "linux-aarch64-a3-0",
  "resource_type": "node",
  "resource_num": 2,
  "size": "2",
  "replicas": "1",
  "multi_node_type": "external",
  "config_path": "tests/e2e/schedule/model/GLM/two_node/GLM5-W8A8-external_dp.yaml"
}
```

For multi-node:

```text
two_node  -> size = 2
four_node -> size = 4
```

---

## 15. parse_schedule_config.py Flow

The parser should follow this flow:

```text
1. Load schedule_config.yaml.
2. Load runner_label.json.
3. Select matching schedule group by cron or workflow_dispatch input.
4. Expand files.
5. Parse each file into PeriodicCase.
6. Apply test_filter.
7. Run framework routers.
8. Generate matrices.
9. Generate image_build_targets.
10. Generate selected_cases_summary.
11. Write outputs to GITHUB_OUTPUT.
```

Suggested CLI:

```bash
python3 .github/workflows/scripts/parse_schedule_config.py \
  --config .github/workflows/scripts/schedule_config.yaml \
  --runner-label .github/workflows/scripts/runner_label.json \
  --event-name "${{ github.event_name }}" \
  --cron "${{ github.event.schedule }}" \
  --schedule-name "${{ inputs.schedule_name }}" \
  --test-filter "${{ inputs.test_filter }}"
```

Expected outputs:

```text
single_node_matrix
multi_node_matrix
accuracy_matrix
ops_matrix
image_build_targets
selected_cases_summary
```

---

## 16. workflow_dispatch Filtering

`test_filter` should support:

```text
all
DeepSeek
Qwen
accuracy
ops
GLM5-W8A8-external_dp
GLM5-W8A8-external_dp.yaml
tests/e2e/schedule/model/GLM/two_node/GLM5-W8A8-external_dp.yaml
```

Recommended matching order:

```text
1. all
2. full path exact match
3. filename exact match
4. filename stem exact match
5. path segment match
6. substring match
```

---

## 17. schedule_periodic_test.yaml

Main workflow:

```text
.github/workflows/schedule_periodic_test.yaml
```

Triggers:

```yaml
on:
  schedule:
    - cron: "45 15 * * *"
    - cron: "45 15 * * 1"

  workflow_dispatch:
    inputs:
      vllm_ascend_branch:
        required: true
        default: main
      schedule_name:
        required: false
        default: manual
      test_filter:
        required: false
        default: all
```

Do not include:

```text
pull_request
nightly-test label
PR comment trigger
```

Jobs:

```text
parse-config
build-image
single-node-tests
multi-node-tests
accuracy-tests
ops-tests
summary
```

---

## 18. Branch Ref and Branch Tag

Do not normalize the branch before checkout.

Keep two values:

```text
branch_ref -> original branch name for checkout
branch_tag -> normalized branch name for image tag / artifact name
```

Example:

```text
branch_ref = releases/v0.20.2rc
branch_tag = releases-v0.20.2rc
```

Use:

```text
branch_ref for actions/checkout
branch_tag for image tag
```

**Hardened (review finding #6):** `build-image` passes `branch_ref` (not
`branch_tag`) into `_nightly_image_build.yaml`. That reusable workflow uses the input
as the `actions/checkout` `ref:` and **normalizes it internally** (`/` -> `-`) to form
the image tag `nightly-ci-<branch_tag>-<chip>`. The test jobs reference the same
`nightly-ci-<branch_tag>-<chip>` using `parse-config.outputs.branch_tag`, so tags
match while checkout still targets a real branch (e.g. `releases/v0.20.2rc`). Passing
`branch_tag` to checkout would break on any branch containing a `/`.

---

## 19. Image Build Rules

`image_build_targets` should be derived from selected cases.

Examples:

```text
selected cases contain only a2 -> build a2 image only
selected cases contain only a3 -> build a3 image only
selected cases contain both    -> build both
```

The workflow should not build unnecessary images for filtered manual runs.

---

## 20. Ops Workflow

Add a lightweight reusable workflow if needed:

```text
.github/workflows/_e2e_periodic_ops.yaml
```

Inputs:

```yaml
runner:
  required: true
  type: string

image:
  required: true
  type: string

tests:
  required: true
  type: string

name:
  required: true
  type: string

should_run:
  required: true
  type: boolean

vllm_ascend_branch:
  required: true
  type: string
```

Execution should run the requested pytest target:

```bash
pytest -s "${{ inputs.tests }}"
```

Ops should not be routed through the model single-node workflow.

**Hardened (review finding #9):** every `run:` step must be valid YAML — a single
line or a `|` block scalar, never a wrapped command split across bare lines:

```yaml
- name: Install vllm-ascend
  run: pip install -e . --no-build-isolation

- name: Run ops tests
  run: pytest -s "${{ inputs.tests }}"
```

The ops job runs inside the nightly image with the NPU device mounts and an
`actions/checkout@v6` of `inputs.vllm_ascend_branch`, mirroring the model e2e setup so
CANN / torch_npu / ascend-toolkit env are available. `should_run` gates the job at the
matrix level.

---

## 21. Validation Rules

The parser must fail early with clear messages for invalid entries.

Required validation:

```text
1. Every file must be under tests/e2e/schedule/.
2. tests/e2e/accuracy/ is invalid.
3. Numeric resource directories such as 1_card are invalid.
4. Each path must contain exactly one supported resource directory.
5. Model and accuracy entries must be YAML files.
6. Ops entries can be Python files or directories.
7. accuracy only supports card resources in the first version.
8. external_dp routing applies only to model multi-node cases.
9. Every selected case must match exactly one router.
10. Empty matrices should be allowed and should skip corresponding jobs.
```

---

## 22. Migration Plan

### Phase 1: Directory Migration

```text
1. Move accuracy configs from tests/e2e/accuracy/ to tests/e2e/schedule/accuracy/.
2. Move ops periodic tests to tests/e2e/schedule/ops/.
3. Move model families under the model/ layer:
   tests/e2e/schedule/<Family>/  ->  tests/e2e/schedule/model/<Family>/
   so configs live at tests/e2e/schedule/model/<family>/<resource_dir>/.
4. Rename external DP files so their filename contains external_dp.
```

Example:

```text
Before:
tests/e2e/schedule/model/GLM/two_node/external_dp/GLM5-external.yaml

After:
tests/e2e/schedule/model/GLM/two_node/GLM5-external_dp.yaml
```

### Phase 2: schedule_config.yaml Cleanup

```text
1. Use only plain path entries in files.
2. Remove chip, npu_num, runner, framework, route, multi_node_type, extra_components.
3. Keep only name, cron, files.
```

### Phase 3: Parser Refactoring

```text
1. Add PeriodicCase.
2. Add framework detection.
3. Add resource detection.
4. Add chip detection.
5. Add runner resolution.
6. Add route detection.
7. Add routers.
8. Output four matrices:
   - single_node_matrix
   - multi_node_matrix
   - accuracy_matrix
   - ops_matrix
```

### Phase 4: Workflow Refactoring

```text
1. Add ops_matrix output to parse-config job.
2. Add ops-tests job.
3. Keep single-node job consuming only single_node_matrix.
4. Keep multi-node job consuming only multi_node_matrix.
5. Keep accuracy job consuming only accuracy_matrix.
6. Separate branch_ref and branch_tag.
7. Print selected_cases_summary.
8. Skip empty matrices without failure.
```

### Phase 5: Tests

Add parser unit tests.

Suggested path:

```text
.github/workflows/scripts/tests/test_parse_schedule_config.py
```

Test cases:

```text
1. model one_node -> single_node
2. model four_card A2 -> single_node + a2
3. model two_node external_dp -> multi_node + external
4. model two_node external -> multi_node + internal
5. accuracy one_card -> accuracy
6. ops one_card -> ops
7. 1_card -> error
8. tests/e2e/accuracy -> error
9. file outside tests/e2e/schedule -> error
10. test_filter by family / filename / stem / full path
```

---

## 23. Acceptance Criteria

The refactor is complete when:

```text
1. schedule_periodic_test.yaml is the unified periodic CI entry.
2. schedule_config.yaml is the unified registry for periodic cases.
3. All selected entries are first converted into PeriodicCase.
4. accuracy, ops, model single-node, and model multi-node are grouped by routers.
5. one_card / two_card / four_card / eight_card / one_node / two_node / four_node are correctly recognized.
6. Numeric resource forms are rejected.
7. a2 / a3 are inferred from path or filename, defaulting to a3.
8. runner labels are resolved through runner_label.json.
9. tests/e2e/schedule/accuracy/ routes to accuracy.
10. tests/e2e/schedule/ops/ routes to ops.
11. model card and one_node cases route to single-node.
12. model two_node and four_node cases route to multi-node.
13. multi-node files containing external_dp route to external.
14. files containing only external do not route to external.
15. Empty matrices skip cleanly.
16. Workflow logs show selected cases, framework, route, chip, runner, and config path.
17. New frameworks can be added by adding a directory rule, a router, and a workflow job.
```

---

## 24. Final Agent Instruction

```text
Please continue modifying the nightly_refactor branch.

The goal is to implement a unified PeriodicCase + Framework Router architecture for periodic CI.

Do not treat this as a simple single_node_matrix / multi_node_matrix parser.
All entries from schedule_config.yaml must first become PeriodicCase objects or dictionaries.
Then each framework must filter the unified case list through its own router.

Required frameworks:
1. accuracy
2. ops
3. model single-node
4. model multi-node

All periodic test cases must live under tests/e2e/schedule/.

Directory rules (framework is always the 4th path segment):
- tests/e2e/schedule/model/<model_family>/<resource_dir>/*.yaml
- tests/e2e/schedule/accuracy/<resource_dir>/*.yaml
- tests/e2e/schedule/ops/<resource_dir>/*.py or directory

Supported resource_dir values:
- one_card
- two_card
- four_card
- eight_card
- one_node
- two_node
- four_node

Do not support numeric forms such as 1_card, 2_card, 1_node, or 2_node.

accuracy must be under:
tests/e2e/schedule/accuracy/

Do not use:
tests/e2e/accuracy/

External DP rule:
- Only applies to model multi-node cases.
- Filename contains external_dp -> external.
- Otherwise -> internal.
- Do not match external.
- Do not rely on an external_dp directory.

Chip rule:
- Path or filename contains a2/A2 -> a2.
- Path or filename contains a3/A3 -> a3.
- Default -> a3.

Route rule:
- model + one_card/two_card/four_card/eight_card -> single_node
- model + one_node -> single_node
- model + two_node/four_node -> multi_node
- accuracy + *_card -> accuracy
- ops + *_card or one_node -> ops

Runner rule:
- Resolve runner through runner_label.json.
- Do not hardcode runner labels in workflow matrices.

Workflow requirements:
1. schedule_periodic_test.yaml should keep only schedule and workflow_dispatch triggers.
2. parse-config outputs:
   - single_node_matrix
   - multi_node_matrix
   - accuracy_matrix
   - ops_matrix
   - image_build_targets
   - selected_cases_summary
3. Add or keep jobs:
   - build-image
   - single-node-tests
   - multi-node-tests
   - accuracy-tests
   - ops-tests
4. Empty matrices should skip cleanly.
5. Separate branch_ref and branch_tag:
   - branch_ref is used for checkout.
   - branch_tag is used for image tags and artifact names.

Migration requirements:
1. Move old tests/e2e/accuracy/ entries to tests/e2e/schedule/accuracy/.
2. Move periodic ops entries to tests/e2e/schedule/ops/.
3. Rename external DP files so the filename contains external_dp.
4. Clean schedule_config.yaml so files are plain paths.
5. Remove old parser logic that depends on an external_dp directory.
6. Remove support for tests/e2e/accuracy/.
7. Remove support for numeric resource directories.

Add parser unit tests covering:
1. model one_node -> single_node
2. model four_card A2 -> single_node + a2
3. model two_node external_dp -> multi_node + external
4. model two_node external -> multi_node + internal
5. accuracy one_card -> accuracy
6. ops one_card -> ops
7. 1_card -> error
8. tests/e2e/accuracy -> error
9. invalid path outside tests/e2e/schedule -> error
10. test_filter matching by family, filename, stem, and full path

The final design must allow future frameworks to be added by adding a new directory rule, a new router, and a new workflow job, without rewriting existing routing logic.
```

---

## 25. Implementation Status

The plan above is fully implemented on `nightly_refactor`. Manifest of delivered changes:

```text
.github/workflows/scripts/parse_schedule_config.py
  - PeriodicCase dataclass; _parse_to_case() builds one case per plain path entry
  - _detect_resource / _detect_framework / _detect_route / _detect_chip /
    _detect_multi_node_type (filename stem) / _resolve_runner
  - BaseRouter + AccuracyRouter, OpsRouter, ModelSingleNodeRouter,
    ModelMultiNodeRouter; ROUTERS registry; each case matches exactly one router
  - Always emits single_node_matrix, multi_node_matrix, accuracy_matrix,
    ops_matrix, image_build_targets, selected_cases_summary (empty -> [])
  - Strict validation: rejects numeric dirs, tests/e2e/accuracy/, non-schedule
    paths, accuracy node resources, missing/duplicate resource dirs
  - UTF-8 explicit I/O; --runner-label CLI flag

.github/workflows/scripts/schedule_config.yaml
  - Plain path strings only; ops/accuracy paths under tests/e2e/schedule/;
    external_dp encoded in filename

.github/workflows/schedule_periodic_test.yaml
  - parse-config emits ops_matrix + selected_cases_summary; prints summary
  - branch_ref (checkout) and branch_tag (image tag) separated;
    build-image receives branch_ref
  - ops-tests job consumes ops_matrix via _e2e_periodic_ops.yaml
  - empty matrices skip via '!= []' guards

  - 310p is a first-class chip (checked before a2/a3); directory entries are
    expanded before routing (ops + accuracy -> per-chip groups; model -> per-file)
  - accuracy: config_paths (not model_list); _validate_accuracy_config rejects
    old list-based group YAMLs; _load_model_list removed

.github/workflows/_e2e_periodic_ops.yaml
  - lightweight reusable ops workflow; valid run: steps; NPU mounts; checkout
  - runs `pytest -s ${tests}` unquoted so per-chip group file lists split into args

.github/workflows/_e2e_nightly_single_node_models.yaml
  - accuracy by direct config_paths: config_paths input (model_list removed),
    effective-config-paths step, per-config `pytest --config`, stem-based summary,
    config_paths-based concurrency + stable job name
  - §1D: test scripts read from tests/e2e/schedule/scripts/accuracy/; the unused
    `tests` input + "Run pytest accuracy test" (.py) step removed

.github/workflows/scripts/tests/test_parse_schedule_config.py
  - 75 tests, all green; covers chip edge cases (incl. 310p), directory expansion
    (ops + accuracy), direct accuracy config_paths, accuracy chip-grouping,
    old-group-YAML validation failure, router-registry exclusivity

Accuracy framework consolidation & cleanup (round 4 — see §1D)
  - test_qwen3_30b_acc.py -> single_node YAML
       model/Qwen/four_card/Qwen3-30B-A3B-W8A8-eagle3-mooncake.yaml (TP1 + TP4);
       single_node gained a `mooncake` sidecar (MooncakeLauncher + mooncake.json)
       and full-path CONFIG_YAML_PATH resolution
  - accuracy harness moved tests/e2e/models/{test_*_eval_correctness.py,conftest.py,
       report_template.md} -> tests/e2e/schedule/scripts/accuracy/
  - .py-accuracy route removed (parser + AccuracyRouter `tests` key + workflow step);
    accuracy is YAML-only
  - deleted tests/e2e/models/ (18 active configs verified identical to accuracy/
    copies; 4 unscheduled + accuracy.txt/json dropped), tests/e2e/weekly/, and the
    empty model/Qwen/one_card/
  - docs + model-adapter skill repointed to schedule/accuracy + schedule/scripts;
    stale tests/e2e/nightly/... multi_node/single_node doc refs repointed

Direct accuracy YAML integration (round 3)
  - real configs copied tests/e2e/models/configs/<model>.yaml
       -> tests/e2e/schedule/accuracy/<resource>/<chip>/<model>.yaml
  - old accuracy-group-*.yaml deleted; schedule_config registers the chip dirs
  - parser/router/workflow emit + consume config_paths; model_list path removed end-to-end

Script layout migration (round 2 — no common/)
  - tests/e2e/nightly/{multi_node,single_node,310p}/ execution scripts
       -> tests/e2e/schedule/scripts/{multi_node,multi_node/internal_dp,
          multi_node/external_dp,single_node}/  (scripts/ subdirs collapsed)
  - ops conftest -> tests/e2e/schedule/ops/conftest.py (with the ops cases)
  - test_recurrent_gated_delta_rule_v310.py -> tests/e2e/schedule/ops/one_card/
  - old tests/e2e/nightly/ tree deleted; all imports/paths/templates repointed;
    no scripts/common/ directories

Migration completed (round 1)
  - tests/e2e/accuracy/  -> tests/e2e/schedule/accuracy/ (old tree deleted)
  - model families moved under the model/ layer:
       tests/e2e/schedule/{DeepSeek,GLM,Hy,Kimi,MiniMax,Qwen}/
         -> tests/e2e/schedule/model/<Family>/
       framework is now always the 4th path segment (model/accuracy/ops)
  - GLM two_node/external_dp/GLM5_1-W8A8-EP-external.yaml
       -> model/GLM/two_node/GLM5_1-W8A8-EP-external_dp.yaml (old dir deleted)
  - accuracy group YAMLs rewritten as plain top-level lists

Resolved
  - Former finding #11 is resolved: accuracy configs are now copied directly into
    tests/e2e/schedule/accuracy/<resource>/<chip>/ and consumed via config_paths;
    the workflow no longer reads tests/e2e/models/configs/${model}.yaml by name.
    §1D: tests/e2e/models/ was deleted outright and the harness moved to
    tests/e2e/schedule/scripts/accuracy/.
  - Former open decision is resolved: tests/e2e/weekly/single_node/configs/ (5 YAMLs,
    superseded by schedule/model configs and unreferenced by code) was deleted in §1D.

Run the suite:
  pytest .github/workflows/scripts/tests/test_parse_schedule_config.py -v
```
