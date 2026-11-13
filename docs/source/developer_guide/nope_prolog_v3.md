# NoPE PROLOG_V3 integration

NoPE SFA layers, including GLM-5.3-Flash, can opt in to fused preprocessing
after a Dr=0-capable operator is installed and callable through
`torch_npu.npu_mla_prolog_v3`. The same-named function in an older installation
does not establish Dr=0 support.

The current contract requires A2/A3 (DAV_2201), W8A8 dynamic MLA projections,
INT8 per-tile C8 KV storage, D=256, KV rank 512, Q rank 1536 or 2048, tile size
128, and PCP/DCP sizes of one. An indexer's query projection may use the
matching W8A8 dynamic scheme or an unquantized FLOAT linear (normally BF16).
The packed cache width is 528 bytes per slot.
MXFP8 MLA weights and other hardware families remain outside this opt-in path.

W8A8 checkpoints can keep `indexer.wq_b` in floating point. This does not
require requantizing the checkpoint or a separate C8 checkpoint. For this
consumer, the adapter multiplies Prolog's INT8 `q_c` by its FP32 per-token
scale, then casts to the hidden-state dtype before the existing indexer linear.
Quantized indexers keep the prepared-input path without this conversion.
`query_quant_mode=0` controls the main attention query, not the low-rank `q_c`.
Dequantization preserves the returned quantized values, not the original
pre-quantization query; compare end-to-end accuracy and performance with native
preprocessing on the target NPU, including graph replay and MTP.

## Install the Dr=0 operator

Build a revision of `ops-transformer` containing the Dr=0 changes and select
`--ops=mla_prolog_v3`. Retain the appropriate SoC, vendor name, and other build
options for that revision. `torch_npu.npu_mla_prolog_v3` calls the V3 operator;
the build targets are distinct:

| Source directory | CANN operator | Build selection |
| --- | --- | --- |
| `attention/mla_prolog_v3` | `MlaPrologV3` | `--ops=mla_prolog_v3` |
| `attention/mla_prolog` | `MlaProlog` | `--ops=mla_prolog` |

Building only `mla_prolog` does not replace `MlaPrologV3`, even though the
implementations share code. Check that the resulting package contains the
updated V3 operator. An installation that still reports
`Parameter Dr of MlaPrologV3 has incorrect value 0. It should be 64.` has not
established that the Dr=0-capable V3 implementation is active.

### Default vendor installation

In the reported Dr=0 deployment, installing the V3 package into the active
CANN installation's `opp/vendors/mlaprolog_dr0_transformer` directory and
giving it priority in `opp/vendors/config.ini` made the updated operator
effective. Install the built package using its default destination and check
the actual destination against the CANN environment used by the serving
process. For an environment whose other installed vendor is `batch_invariant`,
the priority entry is:

```ini
load_priority=mlaprolog_dr0_transformer,batch_invariant
```

The vendor name must match the installed directory. Preserve the other
installed vendors when adjusting their priority; the example is not a
replacement for every environment's existing list.

### Installation into a separate directory

Do not assume that `--install-path` plus an environment variable proves that
the custom implementation replaces the same-named built-in operator. One
reported Dr=0 deployment continued to execute the old V3 implementation with
this setup and succeeded with the default vendor installation above. That
observation does not establish that separate-directory installation is
universally unsupported.

CANN's [operator deployment guide](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_10_0069.html)
supports separate-directory installation. Source the installed vendor's
`bin/set_env.bash` in the same environment that launches Python and the workers:

```bash
source /path/to/custom-opp/vendors/mlaprolog_dr0_transformer/bin/set_env.bash
```

Use the path produced by the installer. `ASCEND_CUSTOM_OPP_PATH` must contain
the concrete vendor directory, for example
`/path/to/custom-opp/vendors/mlaprolog_dr0_transformer`, rather than merely
`/path/to/custom-opp`. The environment script also establishes package order.
If V3 still rejects Dr=0, check the package target, version, vendor path and
the libraries/configuration actually loaded; setting the variable alone is
not evidence of successful replacement. Start a fresh process after changing
the installed package or environment and rerun the smoke check below through
`torch_npu.npu_mla_prolog_v3` before enabling the framework route.

## Validate and enable

Run the standalone smoke check inside the target container, after activating
the custom operator environment. It uses synthetic weights and private cache
storage, through the same Python entry point as the framework:

```bash
python tools/nope_prolog/smoke_nope_prolog_npu.py --device 0
```

The standalone script enables `torch.npu.config.allow_internal_format = True`
after selecting the device and before creating weights. Without this setting,
`npu_format_cast(..., 29)` can downgrade to ND, causing an INT8/ND versus
FRACTAL_NZ OpDef mismatch. Each generated INT8 weight is checked with
`torch_npu.get_npu_format` before calling Prolog; a format mismatch fails at
weight preparation with a specific diagnostic. The script does not inherit
the model runner's internal-format configuration.

The check covers single-token and four-token inputs, zero-width RoPE output,
and preservation of unmapped cache slots when a `-1` padding index is present.
It does not establish model accuracy, performance, or graph capture support.

Merge these options into the existing server command and restart the server
so the decision is made before weight conversion:

```bash
--kv-cache-dtype int8 --additional-config '{"enable_nope_prolog": true}'
```

Preserve other keys already present in `--additional-config`. Start with
`--enforce-eager`, compare against native preprocessing, then validate FULL
capture/replay, different valid row counts within one bucket, and MTP.

Successful selection logs `SFA NoPE preprocessing: PROLOG_V3`. A static
configuration mismatch logs `PROLOG_V3 rejected` and selects native before
weight conversion. Selection alone does not prove that a request executed the
operator; confirm actual calls with a real request and profiling.
FLOAT indexers additionally log `SFA NoPE indexer: dequantize Prolog INT8 q_c
for floating-point wq_b.` at initialization.

## CPU contract tests

The normal unit tests are in `tests/ut/attention/test_nope_prolog_preparation.py`.
For development without vLLM or torch_npu installed, run:

```bash
python tools/nope_prolog/run_cpu_contract_tests.py .
```

This source-isolated runner requires PyTorch and pytest. It executes production
function bodies and the hardware capability matrix, while replacing surrounding
infrastructure and NPU operations. Its results are not full-module, hardware,
graph, or end-to-end validation.
