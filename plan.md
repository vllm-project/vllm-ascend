# Single-File Framework-Oriented Refactoring Plan for parse_schedule_config.py

## 1. Background

The current `parse_schedule_config.py` is responsible for:

1. Selecting the schedules that should run from `schedule_config.yaml`;
2. Reading test entries from `schedule.files`;
3. Inferring framework, resource type, chip type, and runner from paths;
4. Parsing paths into `PeriodicCase`;
5. Converting `PeriodicCase` objects into GitHub Actions matrices through routers;
6. Producing the following outputs:
   - `single_node_matrix`
   - `multi_node_matrix`
   - `accuracy_matrix`
   - `ops_matrix`
   - `image_build_targets`
   - `selected_cases_summary`

The current code already has the concept of routers, for example:

```python
class AccuracyRouter(BaseRouter):
    ...

class OpsRouter(BaseRouter):
    ...

class ModelSingleNodeRouter(BaseRouter):
    ...

class ModelMultiNodeRouter(BaseRouter):
    ...
```

However, these routers only convert already-parsed `PeriodicCase` objects into matrix items.

The real issue is that the common parser still knows too much about framework-specific rules, for example:

```python
def _detect_framework(path: str) -> str:
    ...
```

```python
def _detect_route(framework: str, resource_type: str, resource_num: int) -> str:
    ...
```

```python
def _expand_directory(dir_path: str, runner_map: dict[tuple[str, int], str]) -> list[PeriodicCase]:
    ...
```

These functions still hardcode concepts such as:

```text
model
accuracy
ops
single_node
multi_node
accuracy_matrix
ops_matrix
```

As a result, if we add a new framework in the future, such as:

```text
benchmark
serving
perf
external_dp_v2
accuracy_v2
```

we would need to modify the common parser, route detection, directory expansion, routers, and main output flow. This creates unnecessary coupling.

The goal of this refactoring is: **do not split the file, do not change `schedule_config.yaml`, and do not change workflow output fields. Only move framework-specific logic into dedicated Framework classes.**

---

## 2. Refactoring Goals

### 2.1 Core Goal

Change the current flow from:

```text
schedule_config.yaml
  -> raw paths
  -> common parser detects framework / route / directory expansion
  -> PeriodicCase
  -> Router converts cases into matrices
  -> GITHUB_OUTPUT
```

to:

```text
schedule_config.yaml
  -> raw paths
  -> Framework.match(path)
  -> Framework.expand(raw)
  -> Framework.group(cases)
  -> merge all framework outputs
  -> GITHUB_OUTPUT
```

In other words:

```text
The schedule layer only selects raw entries to run;
the framework layer identifies, expands, and groups its own cases;
main only handles orchestration, deduplication, filtering, merging, and output.
```

### 2.2 Design Principles

This refactoring should follow these principles:

1. Keep everything in a single file;
2. Use one class for each framework;
3. The main flow should not know the internal rules of model / accuracy / ops;
4. Each framework should be responsible for:
   - `match(path)`
   - `expand(raw)`
   - `group(cases)`
5. Keep output fields compatible;
6. Keep `schedule_config.yaml` compatible;
7. Keep `runner_label.json` compatible;
8. Keep the existing directory conventions and behavior compatible.

---

## 3. Non-Goals

This refactoring will not do the following:

1. It will not split the code into:
   - `frameworks/model.py`
   - `frameworks/accuracy.py`
   - `frameworks/ops.py`
2. It will not change the `schedule_config.yaml` format;
3. It will not explicitly add a framework field to the schedule YAML;
4. It will not move chip/resource metadata fully into YAML;
5. It will not change the output fields consumed by GitHub Actions workflows;
6. It will not modify test execution scripts;
7. It will not change the `tests` field from string to list;
8. It will not introduce a plugin registration mechanism;
9. It will not refactor `runner_label.json`.

These can be considered future improvements, but they should not be included in this round to avoid making the PR too large.

---

## 4. Overall Design

### 4.1 Keep Common Utilities

The following functions are common utilities and can remain as shared helpers:

```python
_load_runner_map()
_resolve_runner()
_detect_resource()
_detect_chip()
_derive_name()
_is_directory_entry()
_list_dir_files()
_group_by_chip()
_select_schedules()
_dedupe_cases()
_matches_filter()
```

These functions should only provide basic shared capabilities. They should not contain framework-specific business logic.

For example:

```python
_detect_resource()
```

should only detect resource directories such as:

```text
one_card
two_card
four_card
eight_card
one_node
two_node
four_node
```

```python
_detect_chip()
```

should only infer chip type from a path:

```text
a2
a3
310p
```

```python
_resolve_runner()
```

should only resolve a runner from:

```text
chip + resource_type + resource_num
```

### 4.2 Move Framework-Specific Logic Down

The following functions should be removed or no longer used by `main`:

```python
_detect_framework()
_detect_route()
_parse_to_case()
_expand_directory()
_parse_entry()
```

The replacement relationship should be:

```text
_detect_framework()  -> Framework.match(path)
_detect_route()      -> ModelFramework / AccuracyFramework / OpsFramework internal logic
_parse_to_case()     -> Framework._case_from_file()
_expand_directory()  -> Framework.expand(raw)
_parse_entry()       -> main finds framework and calls framework.expand(raw)
```

---

## 5. Core Data Structure

### 5.1 Keep PeriodicCase

Keep the existing `PeriodicCase` to minimize refactoring risk:

```python
@dataclass(frozen=True)
class PeriodicCase:
    name: str
    path: str
    framework: str
    route: str
    chip: str
    resource_type: str
    resource_num: int
    resource_dir: str
    runner: str
    multi_node_type: str | None = None
    config_path: str | None = None
    config_paths: list | None = None
    tests: str | None = None
    size: int | None = None
```

Different frameworks only use part of these fields, but keeping the existing structure reduces the impact on later output logic.

### 5.2 Add BaseFramework

Introduce `BaseFramework` to replace the core responsibility of the current `BaseRouter`.

Recommended interface:

```python
class BaseFramework:
    name: str
    output_names: tuple[str, ...]

    def __init__(self, runner_map: dict[tuple[str, int], str]):
        self.runner_map = runner_map

    def match(self, path: str) -> bool:
        raise NotImplementedError

    def expand(self, raw: Any) -> list[PeriodicCase]:
        raise NotImplementedError

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        raise NotImplementedError
```

Use:

```python
output_names: tuple[str, ...]
```

instead of:

```python
output_name: str
```

because `ModelFramework` produces two matrices:

```text
single_node_matrix
multi_node_matrix
```

If one framework were only allowed to output one matrix, we would have to split it again into:

```text
ModelSingleNodeFramework
ModelMultiNodeFramework
```

That would conflict with the goal of using one class to handle one framework.

---

## 6. Framework Design

## 6.1 ModelFramework

### Responsibilities

`ModelFramework` is responsible for:

1. Matching entries under `tests/e2e/schedule/model/`;
2. Expanding model directories;
3. Parsing each YAML file into a `PeriodicCase`;
4. Deciding whether the case belongs to single-node or multi-node;
5. Detecting whether `multi_node_type` is internal_dp or external_dp;
6. Producing:
   - `single_node_matrix`
   - `multi_node_matrix`

### Existing Rules to Preserve

```text
model + card resource       -> single_node_matrix
model + one_node            -> single_node_matrix
model + two_node/four_node  -> multi_node_matrix

filename stem contains external_dp -> external_dp
otherwise                         -> internal_dp
```

### Implementation Sketch

```python
class ModelFramework(BaseFramework):
    name = "model"
    output_names = ("single_node_matrix", "multi_node_matrix")

    def match(self, path: str) -> bool:
        norm = str(path).replace("\\", "/").rstrip("/")
        return norm.startswith("tests/e2e/schedule/model/")

    def expand(self, raw: Any) -> list[PeriodicCase]:
        path = str(raw).strip().replace("\\", "/").rstrip("/")

        if _is_directory_entry(raw, path):
            files = _list_dir_files(path, framework="model")
            if not files:
                raise ValueError(f"Directory entry {path!r} contains no model yaml files.")
            return [self._case_from_file(f) for f in files]

        return [self._case_from_file(path)]

    def _case_from_file(self, path: str) -> PeriodicCase:
        resource_dir, resource_type, resource_num = _detect_resource(path)
        chip = _detect_chip(path)
        runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)
        name = _derive_name(path)

        if resource_type == "card" or resource_num == 1:
            route = "single_node"
            multi_node_type = None
            size = None
        else:
            route = "multi_node"
            multi_node_type = self._detect_multi_node_type(path)
            size = resource_num

        return PeriodicCase(
            name=name,
            path=path,
            framework=self.name,
            route=route,
            chip=chip,
            resource_type=resource_type,
            resource_num=resource_num,
            resource_dir=resource_dir,
            runner=runner,
            config_path=path,
            multi_node_type=multi_node_type,
            size=size,
        )

    def _detect_multi_node_type(self, path: str) -> str:
        stem = Path(path.replace("\\", "/")).stem
        return "external_dp" if "external_dp" in stem else "internal_dp"

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        single_node = []
        multi_node = []

        for case in cases:
            if case.route == "single_node":
                single_node.append({
                    "name": case.name,
                    "chip": case.chip,
                    "runner": case.runner,
                    "config_path": case.config_path or "",
                    "tests": "",
                    "extra_components": False,
                })
            elif case.route == "multi_node":
                multi_node.append({
                    "name": case.name,
                    "chip": case.chip,
                    "runner": case.runner,
                    "config_path": case.config_path or "",
                    "multi_node_type": case.multi_node_type or "internal_dp",
                    "extra_components": False,
                    "size": case.size or case.resource_num,
                })
            else:
                raise ValueError(f"Unknown model route: {case.route}")

        multi_node.sort(key=lambda e: -e.get("size", 0))

        return {
            "single_node_matrix": single_node,
            "multi_node_matrix": multi_node,
        }
```

---

## 6.2 AccuracyFramework

### Responsibilities

`AccuracyFramework` is responsible for:

1. Matching entries under `tests/e2e/schedule/accuracy/`;
2. Supporting only card resources;
3. Requiring file entries to be yaml/yml files;
4. Expanding directory entries recursively into yaml/yml files;
5. Grouping directory entries by chip;
6. Producing:
   - `accuracy_matrix`

### Existing Rules to Preserve

```text
accuracy + card resource -> accuracy_matrix
accuracy + node resource -> error

file entry      -> config_paths = [path]
directory entry -> group by chip, one matrix item per chip
```

### Implementation Sketch

```python
class AccuracyFramework(BaseFramework):
    name = "accuracy"
    output_names = ("accuracy_matrix",)

    def match(self, path: str) -> bool:
        norm = str(path).replace("\\", "/").rstrip("/")
        return norm.startswith("tests/e2e/schedule/accuracy/")

    def expand(self, raw: Any) -> list[PeriodicCase]:
        path = str(raw).strip().replace("\\", "/").rstrip("/")

        if _is_directory_entry(raw, path):
            files = _list_dir_files(path, framework="accuracy")
            if not files:
                raise ValueError(f"Directory entry {path!r} contains no accuracy yaml files.")
            return self._cases_from_directory(path, files)

        if not path.endswith((".yaml", ".yml")):
            raise ValueError(f"Accuracy entries must be YAML configs: {path}")

        return [self._case_from_files(path, [path])]

    def _case_from_files(self, path: str, files: list[str]) -> PeriodicCase:
        resource_dir, resource_type, resource_num = _detect_resource(path)
        if resource_type != "card":
            raise ValueError("Accuracy framework only supports card resources.")

        chip = _detect_chip(path)
        runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)

        return PeriodicCase(
            name=_derive_name(path),
            path=path,
            framework=self.name,
            route="accuracy",
            chip=chip,
            resource_type=resource_type,
            resource_num=resource_num,
            resource_dir=resource_dir,
            runner=runner,
            config_paths=files,
        )

    def _cases_from_directory(self, dir_path: str, files: list[str]) -> list[PeriodicCase]:
        resource_dir, resource_type, resource_num = _detect_resource(dir_path)
        if resource_type != "card":
            raise ValueError("Accuracy framework only supports card resources.")

        groups = _group_by_chip(files)
        cases = []

        for chip in sorted(groups):
            group_files = sorted(groups[chip])
            runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)

            cases.append(PeriodicCase(
                name=f"{resource_dir}-{chip}",
                path=dir_path,
                framework=self.name,
                route="accuracy",
                chip=chip,
                resource_type=resource_type,
                resource_num=resource_num,
                resource_dir=resource_dir,
                runner=runner,
                config_paths=group_files,
            ))

        return cases

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        return {
            "accuracy_matrix": [
                {
                    "name": case.name,
                    "chip": case.chip,
                    "runner": case.runner,
                    "config_paths": case.config_paths or [],
                }
                for case in cases
            ]
        }
```

---

## 6.3 OpsFramework

### Responsibilities

`OpsFramework` is responsible for:

1. Matching entries under `tests/e2e/schedule/ops/`;
2. Supporting Python file entries;
3. Supporting directory entries;
4. Recursively expanding directory entries into `test_*.py`;
5. Grouping directory entries by chip;
6. Producing:
   - `ops_matrix`

### Existing Rules to Preserve

```text
ops + any resource -> ops_matrix

file entry      -> tests = path
directory entry -> group by chip, tests = " ".join(group_files)
```

### Implementation Sketch

```python
class OpsFramework(BaseFramework):
    name = "ops"
    output_names = ("ops_matrix",)

    def match(self, path: str) -> bool:
        norm = str(path).replace("\\", "/").rstrip("/")
        return norm.startswith("tests/e2e/schedule/ops/")

    def expand(self, raw: Any) -> list[PeriodicCase]:
        path = str(raw).strip().replace("\\", "/").rstrip("/")

        if _is_directory_entry(raw, path):
            files = _list_dir_files(path, framework="ops")
            if not files:
                raise ValueError(f"Directory entry {path!r} contains no ops test files.")
            return self._cases_from_directory(path, files)

        return [self._case_from_file(path)]

    def _case_from_file(self, path: str) -> PeriodicCase:
        resource_dir, resource_type, resource_num = _detect_resource(path)
        chip = _detect_chip(path)
        runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)

        return PeriodicCase(
            name=_derive_name(path),
            path=path,
            framework=self.name,
            route="ops",
            chip=chip,
            resource_type=resource_type,
            resource_num=resource_num,
            resource_dir=resource_dir,
            runner=runner,
            tests=path,
        )

    def _cases_from_directory(self, dir_path: str, files: list[str]) -> list[PeriodicCase]:
        resource_dir, resource_type, resource_num = _detect_resource(dir_path)
        groups = _group_by_chip(files)
        base_name = _derive_name(dir_path)
        cases = []

        for chip in sorted(groups):
            group_files = sorted(groups[chip])
            runner = _resolve_runner(chip, resource_type, resource_num, self.runner_map)

            cases.append(PeriodicCase(
                name=f"{base_name}-{chip}",
                path=dir_path,
                framework=self.name,
                route="ops",
                chip=chip,
                resource_type=resource_type,
                resource_num=resource_num,
                resource_dir=resource_dir,
                runner=runner,
                tests=" ".join(group_files),
            ))

        return cases

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        return {
            "ops_matrix": [
                {
                    "name": case.name,
                    "chip": case.chip,
                    "runner": case.runner,
                    "tests": case.tests or "",
                }
                for case in cases
            ]
        }
```

---

## 7. Framework Registration

Add `_build_frameworks()`:

```python
def _build_frameworks(runner_map: dict[tuple[str, int], str]) -> list[BaseFramework]:
    return [
        ModelFramework(runner_map),
        AccuracyFramework(runner_map),
        OpsFramework(runner_map),
    ]
```

When adding a new framework later, only this registration function needs to be updated:

```python
def _build_frameworks(runner_map: dict[tuple[str, int], str]) -> list[BaseFramework]:
    return [
        ModelFramework(runner_map),
        AccuracyFramework(runner_map),
        OpsFramework(runner_map),
        BenchmarkFramework(runner_map),
    ]
```

The main flow should not need new model / accuracy / ops if-else branches.

---

## 8. Framework Matching

Add `_find_framework()`:

```python
def _find_framework(path: str, frameworks: list[BaseFramework]) -> BaseFramework:
    matched = [fw for fw in frameworks if fw.match(path)]

    if not matched:
        raise ValueError(f"No framework matched path {path!r}.")

    if len(matched) > 1:
        names = [fw.name for fw in matched]
        raise ValueError(f"Multiple frameworks matched path {path!r}: {names}")

    return matched[0]
```

This guarantees:

1. Unknown frameworks fail fast;
2. Multiple matches fail fast;
3. `main` does not need to know framework names.

---

## 9. main Flow Refactoring

### 9.1 Current main Problem

The current main flow is roughly:

```text
select schedules
  -> _parse_entry(raw, runner_map)
  -> _parse_to_case / _expand_directory
  -> dedupe
  -> filter
  -> ROUTERS route to matrix
  -> output
```

The problem is that:

```text
_parse_entry
_parse_to_case
_expand_directory
ROUTERS
```

still couple the common parser with framework-specific logic.

### 9.2 New main Flow

Recommended new flow:

```python
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to schedule_config.yaml")
    parser.add_argument("--runner-label", help="Path to runner_label.json (default: same dir as config)")
    parser.add_argument("--event-name", default="workflow_dispatch")
    parser.add_argument("--cron", default="")
    parser.add_argument("--schedule-name", default="")
    parser.add_argument("--test-filter", default="all")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    runner_label_path = Path(args.runner_label) if args.runner_label else Path(args.config).parent / "runner_label.json"
    runner_map = _load_runner_map(runner_label_path)
    frameworks = _build_frameworks(runner_map)

    schedules = _select_schedules(config, args.event_name, args.cron, args.schedule_name)
    if not schedules:
        print(
            f"No schedules matched event={args.event_name!r} cron={args.cron!r} "
            f"schedule_name={args.schedule_name!r}",
            file=sys.stderr,
        )

    all_cases: list[PeriodicCase] = []
    errors: list[str] = []

    for schedule in schedules:
        for raw in schedule.get("files", []):
            path = str(raw).strip().replace("\\", "/").rstrip("/")
            try:
                framework = _find_framework(path, frameworks)
                all_cases.extend(framework.expand(raw))
            except Exception as exc:
                errors.append(f"  {raw!r}: {exc}")

    if errors:
        print("Errors parsing schedule entries:", file=sys.stderr)
        for e in errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    all_cases, duplicate_cases = _dedupe_cases(all_cases)

    test_filter = args.test_filter.strip()
    if test_filter:
        all_cases = [c for c in all_cases if _matches_filter(c, test_filter)]

    cases_by_framework: dict[str, list[PeriodicCase]] = {
        fw.name: [] for fw in frameworks
    }

    for case in all_cases:
        cases_by_framework[case.framework].append(case)

    outputs: dict[str, list[dict]] = {}
    for fw in frameworks:
        for output_name in fw.output_names:
            outputs[output_name] = []

    for fw in frameworks:
        grouped = fw.group(cases_by_framework[fw.name])
        for output_name, items in grouped.items():
            if output_name not in outputs:
                raise ValueError(f"Framework {fw.name!r} returned unknown output {output_name!r}.")
            outputs[output_name].extend(items)

    image_targets = sorted({c.chip for c in all_cases})

    summary = _build_summary(all_cases, outputs, duplicate_cases)
    _write_outputs(outputs, image_targets, summary)
```

With this structure, `main` only orchestrates the process and no longer contains branches such as:

```python
if framework == "model":
    ...

if framework == "accuracy":
    ...

if framework == "ops":
    ...
```

---

## 10. Output Logic Encapsulation

To keep `main` clean, move summary generation and GitHub output writing into two helper functions.

### 10.1 _build_summary()

```python
def _build_summary(
    all_cases: list[PeriodicCase],
    outputs: dict[str, list[dict]],
    duplicate_cases: dict[str, list[PeriodicCase]],
) -> str:
    summary_lines = ["=== Selected test cases ==="]

    for c in all_cases:
        loc = c.config_path or c.tests or c.path
        summary_lines.append(
            f"  [{c.framework:8s}] [{c.route:11s}] [{c.chip}] "
            f"[{c.runner:30s}] {c.name} ({loc})"
        )

    summary_lines.append(
        f"\nTotals: "
        f"{len(outputs.get('single_node_matrix', []))} single-node, "
        f"{len(outputs.get('multi_node_matrix', []))} multi-node, "
        f"{len(outputs.get('accuracy_matrix', []))} accuracy, "
        f"{len(outputs.get('ops_matrix', []))} ops"
    )

    if duplicate_cases:
        summary_lines.append("\nWARNING: duplicate test case names detected; kept the first occurrence:")
        for name in sorted(duplicate_cases):
            cases = duplicate_cases[name]
            summary_lines.append(f"  {name}:")
            summary_lines.append(f"    kept: {cases[0].path}")
            for case in cases[1:]:
                summary_lines.append(f"    duplicate: {case.path}")

    return "\n".join(summary_lines)
```

### 10.2 _write_outputs()

```python
def _write_outputs(
    outputs: dict[str, list[dict]],
    image_targets: list[str],
    summary: str,
) -> None:
    print(summary, file=sys.stderr)

    output_path = os.environ.get("GITHUB_OUTPUT", "")

    lines = [
        f"single_node_matrix={json.dumps(outputs.get('single_node_matrix', []))}",
        f"multi_node_matrix={json.dumps(outputs.get('multi_node_matrix', []))}",
        f"accuracy_matrix={json.dumps(outputs.get('accuracy_matrix', []))}",
        f"ops_matrix={json.dumps(outputs.get('ops_matrix', []))}",
        f"image_build_targets={json.dumps(image_targets)}",
        f"selected_cases_summary={json.dumps(summary)}",
    ]

    if output_path:
        with open(output_path, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    else:
        print("\n=== Outputs ===")
        for output_name in (
            "single_node_matrix",
            "multi_node_matrix",
            "accuracy_matrix",
            "ops_matrix",
        ):
            items = outputs.get(output_name, [])
            print(f"\n{output_name} ({len(items)} entries):")
            print(json.dumps(items, indent=2))
        print(f"\nimage_build_targets: {image_targets}")
```

The output fields must remain compatible:

```text
single_node_matrix
multi_node_matrix
accuracy_matrix
ops_matrix
image_build_targets
selected_cases_summary
```

---

## 11. Behavior Compatibility Requirements

The refactored implementation must preserve the following behavior.

### 11.1 model file entry

Input:

```text
tests/e2e/schedule/model/one_card/foo.yaml
```

Output target:

```text
single_node_matrix
```

### 11.2 model directory entry

Input:

```text
tests/e2e/schedule/model/one_card/
```

Behavior:

```text
Recursively find *.yaml / *.yml
Each yaml file becomes one PeriodicCase
All cases go to single_node_matrix
```

### 11.3 model multi-node

Input:

```text
tests/e2e/schedule/model/two_node/foo.yaml
tests/e2e/schedule/model/four_node/foo_external_dp.yaml
```

Behavior:

```text
two_node/four_node -> multi_node_matrix
filename contains external_dp -> multi_node_type = external_dp
otherwise -> internal_dp
```

### 11.4 accuracy file entry

Input:

```text
tests/e2e/schedule/accuracy/one_card/foo.yaml
```

Output:

```json
{
  "name": "foo",
  "chip": "a3",
  "runner": "...",
  "config_paths": ["tests/e2e/schedule/accuracy/one_card/foo.yaml"]
}
```

### 11.5 accuracy directory entry

Input:

```text
tests/e2e/schedule/accuracy/one_card/
```

Behavior:

```text
Recursively find *.yaml / *.yml
Group by chip
One matrix item per chip
```

### 11.6 ops file entry

Input:

```text
tests/e2e/schedule/ops/one_card/test_foo.py
```

Output:

```json
{
  "name": "test_foo",
  "chip": "a3",
  "runner": "...",
  "tests": "tests/e2e/schedule/ops/one_card/test_foo.py"
}
```

### 11.7 ops directory entry

Input:

```text
tests/e2e/schedule/ops/one_card/
```

Behavior:

```text
Recursively find test_*.py
Group by chip
One matrix item per chip
tests = " ".join(group_files)
```

---

## 12. Error Handling Strategy

### 12.1 No framework matched

If a path does not start with one of the following prefixes:

```text
tests/e2e/schedule/model/
tests/e2e/schedule/accuracy/
tests/e2e/schedule/ops/
```

fail fast with:

```text
No framework matched path '...'
```

### 12.2 Multiple frameworks matched

This should not normally happen, but it should still be guarded:

```text
Multiple frameworks matched path '...': ['xxx', 'yyy']
```

### 12.3 Missing resource directory

If a path does not contain one of:

```text
one_card
two_card
four_card
eight_card
one_node
two_node
four_node
```

keep the existing error behavior:

```text
No resource directory in path ...
```

### 12.4 Multiple resource directories

Keep the existing error behavior:

```text
Multiple resource directories in path ...
```

### 12.5 accuracy uses node resource

Keep failing with:

```text
Accuracy framework only supports card resources.
```

### 12.6 Directory entry contains no routable files

Keep failing with:

```text
Directory entry '...' contains no routable files.
```

---

## 13. Unit Test Plan

Update or add tests to ensure the refactoring does not change behavior.

### 13.1 Framework match tests

Test targets:

```text
ModelFramework.match()
AccuracyFramework.match()
OpsFramework.match()
_find_framework()
```

Coverage:

1. model path only matches ModelFramework;
2. accuracy path only matches AccuracyFramework;
3. ops path only matches OpsFramework;
4. unknown path raises an error;
5. multiple framework matches raise an error, which can be tested with a fake framework.

### 13.2 ModelFramework expand tests

Coverage:

1. `one_card yaml -> single_node`;
2. `two_card yaml -> single_node`;
3. `one_node yaml -> single_node`;
4. `two_node yaml -> multi_node`;
5. `four_node yaml -> multi_node`;
6. `external_dp filename -> multi_node_type = external_dp`;
7. `normal multi-node filename -> internal_dp`;
8. `model directory entry -> each yaml becomes one case`.

### 13.3 AccuracyFramework expand tests

Coverage:

1. `accuracy yaml file -> single-element config_paths`;
2. `accuracy directory -> group by chip`;
3. `accuracy node resource -> error`;
4. `non-yaml file -> error`;
5. `empty directory -> error`.

### 13.4 OpsFramework expand tests

Coverage:

1. `ops py file -> tests = path`;
2. `ops directory -> recursively find test_*.py`;
3. `ops directory -> group by chip`;
4. `empty directory -> error`.

### 13.5 main output compatibility test

Create a schedule config containing:

```yaml
periodic_tests:
  - name: manual
    files:
      - tests/e2e/schedule/model/one_card/foo.yaml
      - tests/e2e/schedule/model/two_node/bar_external_dp.yaml
      - tests/e2e/schedule/accuracy/one_card/acc.yaml
      - tests/e2e/schedule/ops/one_card/test_ops.py
```

Verify that the output contains:

```text
single_node_matrix
multi_node_matrix
accuracy_matrix
ops_matrix
image_build_targets
selected_cases_summary
```

Also verify that each matrix item remains compatible with the previous behavior.

### 13.6 test_filter tests

Preserve the existing semantics:

1. Exact full path match;
2. Exact filename match;
3. Exact stem match;
4. Path segment match;
5. Name/path substring match;
6. `all` matches everything.

---

## 14. Recommended Implementation Steps

### Step 1: Introduce BaseFramework

Add:

```python
class BaseFramework:
    ...
```

Do not remove the old routers immediately.

### Step 2: Implement ModelFramework

Migrate model logic, including:

```text
single_node
multi_node
internal_dp
external_dp
model directory expansion
```

Add or update corresponding unit tests.

### Step 3: Implement AccuracyFramework

Migrate accuracy logic, including:

```text
yaml file entry
directory entry
chip grouping
card-only validation
```

Add or update corresponding unit tests.

### Step 4: Implement OpsFramework

Migrate ops logic, including:

```text
py file entry
directory entry
test_*.py discovery
chip grouping
```

Add or update corresponding unit tests.

### Step 5: Replace the main flow

Change main from:

```text
_parse_entry -> ROUTERS
```

to:

```text
_find_framework -> framework.expand -> framework.group
```

### Step 6: Clean up old logic

Remove or stop using:

```python
_detect_framework()
_detect_route()
_parse_to_case()
_expand_directory()
_parse_entry()
BaseRouter
AccuracyRouter
OpsRouter
ModelSingleNodeRouter
ModelMultiNodeRouter
ROUTERS
```

If the PR becomes too large, it is acceptable to stop using the old logic in the first round and remove it in a later cleanup PR.

### Step 7: Local validation

Run at least:

```bash
python .github/workflows/scripts/parse_schedule_config.py \
  --config .github/workflows/scripts/schedule_config.yaml \
  --runner-label .github/workflows/scripts/runner_label.json \
  --event-name workflow_dispatch \
  --schedule-name manual \
  --test-filter all
```

Compare the JSON output before and after the refactoring.

---

## 15. Review Focus

When submitting the PR, highlight the following points:

1. This PR does not change `schedule_config.yaml`;
2. This PR does not change GitHub Actions output fields;
3. This PR does not change existing model / accuracy / ops routing rules;
4. This PR only moves framework-specific parsing, routing, and grouping logic into dedicated classes;
5. Adding a new framework later only requires adding a new class and registering it;
6. `main` no longer needs to know the internal rules of model / accuracy / ops.

---

## 16. Future Extension Example

If we add a benchmark framework later, with paths such as:

```text
tests/e2e/schedule/benchmark/one_card/*.yaml
```

we only need to add:

```python
class BenchmarkFramework(BaseFramework):
    name = "benchmark"
    output_names = ("benchmark_matrix",)

    def match(self, path: str) -> bool:
        return path.replace("\\", "/").rstrip("/").startswith(
            "tests/e2e/schedule/benchmark/"
        )

    def expand(self, raw: Any) -> list[PeriodicCase]:
        ...

    def group(self, cases: list[PeriodicCase]) -> dict[str, list[dict]]:
        ...
```

Then register it:

```python
def _build_frameworks(runner_map: dict[tuple[str, int], str]) -> list[BaseFramework]:
    return [
        ModelFramework(runner_map),
        AccuracyFramework(runner_map),
        OpsFramework(runner_map),
        BenchmarkFramework(runner_map),
    ]
```

No changes should be needed in `main`.

---

## 17. Expected Benefits

After this refactoring, the file structure will look like:

```text
parse_schedule_config.py
  - common helpers
  - PeriodicCase
  - BaseFramework
  - ModelFramework
  - AccuracyFramework
  - OpsFramework
  - schedule selection
  - dedupe/filter
  - output writer
  - main
```

Benefits:

1. Still a single file, so review cost remains low;
2. Framework logic is centralized and easier to understand;
3. The main flow becomes simpler and no longer contains framework-specific if-else branches;
4. Adding a new framework does not require modifying the whole pipeline;
5. Workflow output compatibility is preserved, reducing risk;
6. Existing path conventions are preserved, reducing migration cost;
7. Future frameworks such as benchmark, serving, perf, external_dp_v2, and accuracy_v2 can be added naturally.

---

## 18. Acceptance Criteria

After the refactoring, the implementation should satisfy:

1. `parse_schedule_config.py` is still a single file;
2. The following classes exist:
   - `BaseFramework`
   - `ModelFramework`
   - `AccuracyFramework`
   - `OpsFramework`
3. `main` no longer contains model / accuracy / ops business-specific branching logic;
4. `main` only handles:
   - reading config;
   - selecting schedules;
   - finding the matching framework;
   - calling `framework.expand`;
   - deduplication;
   - filtering;
   - calling `framework.group`;
   - merging outputs;
   - writing `GITHUB_OUTPUT`;
5. Output fields remain unchanged;
6. Existing `schedule_config.yaml` does not need to be modified;
7. Existing workflows do not need to be modified;
8. Existing model / accuracy / ops behavior remains unchanged;
9. Unit tests cover framework matching, expansion, grouping, and main output compatibility.
