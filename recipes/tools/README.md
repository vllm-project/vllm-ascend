# Ascend recipe configuration checks

These CPU-only checks run after the official renderer's recipe validation.
The official reusable workflow retains ownership, schema and API validation; this
tool checks effective Ascend installation and serving settings. It never
executes recipe shell commands, pulls serving images, or loads model weights.

## CI flow

The original `validate` job keeps calling
`vllm-project/recipes/.github/workflows/validate-recipes.yml@main` with its
existing inputs. A separate `Ascend configuration` CPU job has `needs: validate`
and calls the local `.github/workflows/validate-ascend-recipes.yml` reusable
workflow only after the official check succeeds. The entry workflow contains
only orchestration; checkout, setup, lint/tests, checks and upload live in the
local reusable workflow. Changes to either workflow trigger the recipe check.
No upstream changes or copies
of the official workflow steps are needed in that original job.

The additional job checks out the PR merge revision, target revision and
`vllm-project/recipes@main`, installs its own renderer dependencies, then runs
JavaScript syntax checks, ESLint, tests and Ascend configuration checks. Jobs
do not share a filesystem: the downstream job prepares its own source cache.
Within that job, candidate and baseline use the same renderer and source pins.
Both jobs independently resolve renderer main, so the report records the exact
downstream SHA rather than claiming it must match the earlier job's SHA.
Checkouts do not persist credentials, contents permission is read-only, and no
NPU runner is requested.

The checker invokes the renderer's `sync-sources.mjs` and strict
`assemble-content.mjs` in temporary channels. Both candidate and baseline use
the same non-Ascend source pins. Ambient `RECIPES_*` overrides are removed from
child processes; source configuration is generated from the supplied renderer.
Cleanup removes only these temporary channels, leaving `latest`, `validate`,
the renderer's source configuration and public API output untouched.

## Run locally

Use Node.js 22 or later and a clean renderer checkout with its dependencies:

```bash
git clone https://github.com/vllm-project/recipes.git /tmp/recipes-renderer
cd /tmp/recipes-renderer
pnpm install --frozen-lockfile --ignore-scripts
cd /path/to/vllm-ascend
node recipes/tools/validate-ascend.mjs \
  --renderer /tmp/recipes-renderer \
  --report /tmp/ascend-recipes/report.json
```

No manually prepared core YAML snapshot is required. The renderer fetches its
locked recipe sources. `--recipes PATH` selects another Ascend recipe directory.
Filters `--model org/model`, `--hardware ID`, `--variant KEY` and `--strategy KEY`
can narrow a local investigation. No matching candidate configurations is a
failure, not a successful empty scan. Symlink entries in inspected source trees
are explicitly rejected rather than silently omitted.

By default all errors block. To reproduce the PR's no-new-errors mode:

```bash
node recipes/tools/validate-ascend.mjs \
  --renderer /tmp/recipes-renderer \
  --baseline-recipes /path/to/target-revision/recipes \
  --report /tmp/ascend-recipes/report.json
```

The baseline is checked with the current checker and the same renderer/source
pins; it is not a committed findings allowlist. Findings are compared by model,
hardware, variant, strategy, role, rule, severity, message and rule-specific
evidence. New/changed errors block; unchanged errors stay visible as existing
debt; removed errors are listed as resolved. Source/zero-coverage invariants
always block. A missing or failed baseline cannot silently become a clean one.
An empty valid baseline is allowed when adding the source's first model.

Exit codes are `0` for no blocking findings, `1` for blocking findings, and `2`
for input/tool failures. In no-new-errors mode, `0` does **not** mean all
configurations are valid. `--warnings-as-errors` also blocks new warnings, or
all warnings when no baseline is supplied.

## Checks

| Rule | Effective configuration checked |
| --- | --- |
| ASC001 | Empty/default GPU image, inherited NVIDIA/CUDA image map, or known NVIDIA image path. Custom NPU images need no particular registry or tag suffix. |
| ASC002–003 | Visible pip tab using generic/CUDA defaults or explicitly selecting known CUDA wheel indexes. Adding dependencies does not replace the main pip installation command. |
| ASC004–005 | Both installation methods disabled, or custom Docker setup directly pulling a known NVIDIA image. |
| ASC010–014 | Missing PD expectations, connector/router mismatch, malformed KV JSON, or invalid producer/consumer role. NIXL is not banned by name. |
| ASC020–023 | CUDA-only PD pinning; conflicting, duplicated, malformed or out-of-range device IDs; symbolic IDs reported as warnings. |
| ASC024–025 | Single-node TP×DP×PP exceeding visible capacity, invalid parallel sizes, or symbolic sizes reported as warnings. TP/DP/PP aliases follow last-value precedence. |
| ASC030 | Known FLASHINFER attention or MoE backend settings retained in NPU output. |

Do not interpret this as a CUDA/NCCL word blacklist. Names such as
`--gpu-memory-utilization`, `cudagraph_mode` and `NCCL_SOCKET_IFNAME` are allowed.
The device-range check assumes profile-local continuous numbering; explicit
host/container remapping needs separate review.

PD expectations may live at `recipes/validation/pd-expectations.json`, with
each snapshot reading its own version. `--pd-expectations FILE` is an explicit
local override. Entries have this shape:

```json
{
  "example/model": {
    "ascend_910c": {
      "ascend_variant": {
        "connector": "MooncakeConnectorV1",
        "router": "vllm-router"
      }
    }
  }
}
```

The example shows the file structure, not a qualified deployment pairing.
Expectations must describe the intended tutorial deployment; they are not
runtime pass records. Do not change them solely to match an unintended default.

## Compatibility and limits

The renderer does not export a shared installation resolver yet. The adapter
guards the relevant JSX installation-selection blocks with fingerprints.
Unrelated renderer commits are accepted; a change to those blocks fails with
an adapter-review diagnostic. Review the UI behavior and tests before updating
fingerprints. Other command/assembly APIs are exercised by integration tests.

Coverage is Huawei overlays, baseline preset, features disabled, Python
frontend requested, default PD connector/router, and minimum multi-node topology.
Inherited environment is preserved: `VLLM_USE_RUST_FRONTEND=1` may still enable
Rust despite the Python request. Reports disclose this limitation; frontend
compatibility is not verified. Full
recipes under `models/` are explicitly rejected. Tuned presets, feature
combinations, other frontends, all PD pool modes and all topology sizes are
outside this change. Multi-node/PD per-node capacity is not verified.

Custom install scripts and GPU-only package detection are not fully interpreted.
The checker does not establish registry availability, package ABI compatibility,
weight presence, NPU inference correctness or performance. Missing hardware
opt-outs can expose extra page selections; those combinations are checked too.

The official add-only migration rules remain in effect. If a core recipe still
contains an imported Ascend field, conflicting downstream changes are rejected
until the corresponding upstream split is completed.

## Tests and reports

```bash
node --test recipes/tools/test-ascend-checks.mjs
node recipes/tools/test-ascend-integration.mjs --renderer /tmp/recipes-renderer
/tmp/recipes-renderer/node_modules/.bin/eslint \
  --no-config-lookup --config recipes/tools/eslint.config.mjs recipes/tools/*.mjs
```

Integration tests use self-contained local fixtures and the real official
assembler/generator, without remote source fetches. `--override source=PATH`
is available for local non-Ascend source checkouts; it cannot replace the
candidate Ascend source and is not used in CI.

JSON and Markdown reports contain renderer/source provenance, input hashes,
generated worker argv/env, findings and skipped combinations. The Actions
summary shows at most 40 findings; the artifact contains the full report.
Input/tool failures also write a failure report when `--report` is supplied.
The workflow does not create or update branch-protection requirements. The
existing reusable check is preserved; requiring the new Ascend check is a
separate repository setting.
