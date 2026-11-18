# Bisect dependency recipe

The bisect environment is prepared from the target commit's build chain.
`BuildManager` uses the build analyzer and planner to prepare each trial;
legacy version-adapter installation commands are not a second switching path.

## Build inputs and analysis

The analyzer selects the root Dockerfile and matching nightly Dockerfile by
SoC, architecture and Ubuntu/openEuler variant. For A3 Ubuntu, the entries are
`Dockerfile.a3` and `.github/workflows/dockerfiles/Dockerfile.nightly.a3`.
Workflow YAML is not an analysis entry point.

Dockerfile instructions and Bash syntax are parsed before semantic analysis.
The analysis tracks variable scope, symbolic environment references, working
directories, installation order and resource dependencies. Local dependency
files, including `requirements.txt` and `requirements-nightly.txt`, are
expanded at their consuming install commands. An install with `--no-deps`
does not implicitly install the project's requirements.

`setup.py` and `pyproject.toml` are project build inputs. A change can require
replaying the target project installation; the analyzer does not execute
arbitrary Python to discover every possible dynamic dependency. Unknown
external inputs and unsupported semantics remain explicit diagnostics.

## Planning and execution

When the root base image is unchanged, the planner compares semantic operations
and final package states, then includes the dependency steps needed to replay
the changes in target build order. Package names are normalized, so changing
hyphens to underscores alone does not require reinstalling a package.

Changing the checked-out project commit also requests the target vLLM Ascend
installation action, including for native-source changes. This action is
merged with an already planned project install rather than appended twice.
Before project installation, stale build artifacts such as `build`,
`csrc/build` and egg metadata are removed. Source directories are retained.

Installation commands and their shell context come from the target recipe.
Persistent Dockerfile `ENV` and temporary `RUN` exports have different scope:
a temporary export is not automatically passed to model tests. Symbolic
references such as `$PATH` are expanded during execution. The model environment
is not given an additional default pip index by `BuildManager`.

The executor checks out the target commit and executes the generated actions.
Planner `blockers`, including a plan whose status is `blocked`, are printed as
warnings; that status alone does not prevent actions from running. A failed
optional action is also a warning, and actions requiring its unavailable
outputs can be omitted. A failed required action still aborts preparation and
is reported as an environment error rather than a model failure. Continuing
after warnings does not prove that every dependency was successfully switched.

The plan's `preflight` and `postflight` command lists are review information;
`BuildManager` does not automatically run these lists as package verification.
Execution-specific image safety checks still apply.

## Base-image changes

Image history is fetched on demand when the root `FROM` reference changes,
not during ordinary analysis of every commit. This applies to all image names;
there is no CANN-name filter and no two-image filesystem-diff selection path. The target image's history and
configuration provide the replay instructions; image contents provide the
sources for replayed copies. The old CANN recipe is not compared to select
which target installation steps should run.

The plan first replays the target image's final-stage operations, including its
Python environment, then replays the full target vLLM Ascend build chain in
order. This path does not use incremental package differences to omit unchanged
installations: those packages may be absent from the replacement Python.
Build metadata instructions are not executed as shell commands, and the project
checkout is retained instead of being replaced by a build-context copy.

Replay is not guaranteed for arbitrary images. Unrecoverable history,
unsupported instructions and execution safety failures remain visible in the
logs. Broad temporary-directory cleanup is guarded to avoid deleting the
running bisect tool and its files. Migration logs show copy progress and CANN
version-file contents; printing these versions is not a device validation test.

## Tool environment, mock mode and logs

On POSIX, the CLI prepares a separate `tool_venv` beside the default runs
directory. `tools/bisect/build_analyzer/requirements.txt` declares its parser
packages and `regex`; a requirements hash triggers installation when this file
changes. Tool bootstrap uses an existing `PIP_INDEX_URL`, or a Huawei Cloud
fallback. This bootstrap policy is separate from model dependency installation.
`PATH` is left unchanged so trial commands still use the model environment.

`--mock-tests` performs real checkout and environment preparation but replaces
pytest and model serving with synthetic verdicts. It can download packages and
rebuild native extensions; it is not a dry run. The CLI currently rejects this
mode for multi-node scenes.

Per-trial logs are written under
`<work-dir>/<scene>__<config-yaml>/logs/round<N>_<commit>.log`.
Mock runs add `__mock` to the case directory name. State and `report.json` are
stored alongside `logs`. CANN target-image plans are saved as
`<work-dir>/target-image-plan-node<N>.json`. Build subprocess output and pytest
output are streamed to the console and trial log.

## Initial environment preparation

By default, the tool assumes the starting HEAD is already installed. With
`--force-initial-build` or `--no-assume-built-head`, no installed baseline is
assumed: the first actual trial replays the target base image, including Python
and CANN, followed by the full target build chain. No old-image comparison is
needed. Later trials use the established baseline and normal transition rules.

Every nonempty transition plan, including ordinary dependency installations,
writes the pending marker before checkout and execution. Successful preparation
clears it; interruption or required-action failure leaves it in place. A restart
with a pending marker refuses to trust HEAD. Restarting with
`--force-initial-build` explicitly permits a full target initialization without
using the old baseline. The marker remains until preparation succeeds; a new
failure blocks subsequent trials in that process. `--no-assume-built-head`
alone does not authorize recovery from a pending transition.
Optional failures retain their existing warning-and-continue behavior.
