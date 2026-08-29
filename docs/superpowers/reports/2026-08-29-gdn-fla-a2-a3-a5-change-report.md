# GDN FLA A2/A3/A5 change report

## Baseline

- Branch: `a2-a3-gdn-core-fwd-debug`
- Base commit: `69d84b236`
- Date: 2026-08-29

## Confirmed contract

`gdn_core_fwd_phase6` uses the same FLA Python API, ACLNN entry, and operator
source for A2, A3, and A5. Each hardware family still requires a wheel and OPP
built for its own target:

- A2: `ascend910b`
- A3: `ascend910_93`
- A5: `ascend950`

The FLA repository contains formal Phase 6 A2 evidence and supports all three
build targets. A3 device acceptance remains pending until the tests in this
report run on A3 hardware.

## Production diff

1. Added `get_fla_gdn_soc()` and `is_fla_gdn_supported()` as the semantic
   hardware capability boundary. A2, A3, and A5 map to their FLA build targets;
   310P remains unsupported.
2. Replaced the Qwen GDN `is_950()` gate and hard-coded `soc="ascend950"` with
   the capability result.
3. Generalized adapter, dispatcher, cache, metadata aliases, local variables,
   comments, and log prefixes from A5 to FLA terminology.
4. Moved the primary implementation from `vllm_ascend.ops.gdn_a5` to
   `vllm_ascend.ops.gdn_fla`. The old module and class names remain compatibility
   aliases.
5. Preserved existing Stage 1 exclusions: speculative decode, graph capture,
   PCP world size greater than one, and non-BF16 activations.
6. Preserved `auto`, strict `fla_npu`, and `native` backend semantics. The fused
   prefill symbol remains `fla_npu.ops.ascendc.gdn_core_fwd_phase6`.

## Test diff

- Added device capability tests for A2, A3, A5, and 310P.
- Added a parameterized Qwen adapter routing test for `ascend910b`,
  `ascend910_93`, and `ascend950`.
- Renamed unit and operator smoke coverage from `test_gdn_a5.py` to
  `test_gdn_fla.py` and changed fixtures to use the detected SoC.
- Enabled the Qwen3.5 and Qwen3.6 eager smoke cases on every supported FLA GDN
  hardware family instead of skipping everything except A5.

## Documentation diff

- Added the authoritative cross-SoC, fused-core design at
  `docs/superpowers/specs/2026-08-29-qwen35-qwen36-gdn-fla-design.md` and linked
  it through the developer design index.
- Marked the original A5 design and implementation plan as historical Stage 1
  artifacts. The former design body is now a stable redirect so obsolete
  A5-only and six-operator decisions cannot be mistaken for current behavior.
- Reworked the developer-guide page into a concise entry point that distinguishes
  resolution fallback from the current Phase 6 live-probe limitation.
- Updated the environment-variable description to state A2/A3/A5 behavior.

## Local verification

Completed on the development workstation:

- `git diff --check`: no whitespace errors.
- Long-line and trailing-whitespace scan over every changed/new file: production
  and test files are clean.
- Repository search: no A5-only gate remains in the Qwen GDN routing, adapter,
  current tests, environment help, or current design document.

The workstation has no `pytest`, `ruff`, or Python runtime, and repository
instructions prohibit invoking `python` or `python3`. Therefore no Python test
or syntax-pass claim is made here.

## Required device verification

On each A2/A3/A5 environment, install the matching FLA wheel and run:

```bash
cd /home/z00886386/vllm-ascend

pytest -q tests/ut/device/test_device_config.py
pytest -q tests/ut/ops/test_gdn_fla.py
pytest -s -q tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py
```

Then run the Qwen3.5 and Qwen3.6 eager smoke tests. Confirm that selection logs
contain the expected SoC and `gdn_core_fwd_phase6`, and that unexpected fallback
logs identify the failing operator and stage.

## Remaining acceptance boundary

This change makes A3 eligible and testable; it does not turn build support into
an A3 acceptance claim. A3 operator smoke, model output comparison, and TP model
tests must pass on physical A3 hardware before enabling it in a release matrix.
