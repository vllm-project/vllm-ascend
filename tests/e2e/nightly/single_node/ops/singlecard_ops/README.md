# Single-card operator tests

Tests are organized by the implementation under test and the CI architecture
group:

```text
singlecard_ops/
  ascendc/
    a2a3/
    a5/
  triton/
    a2a3/
    a5/
```

`ascendc` contains locally maintained Ascend C/native bindings and their
associated integration regressions. `triton` contains local Triton kernels
and the wrappers whose state, layout or dispatch behavior these tests protect.
Calling CANN or FLA as a reference or an auxiliary step does not make a local
integration test an external-operator test. Standalone tests of external CANN
or FLA implementations belong with their implementation repositories.

The two existing `test_causal_conv1d_310.py` and
`test_recurrent_gated_delta_rule_310.py` files remain at this level and route
to 310P. There are no backend-specific `310p` directories.

## Architecture routing

The authoritative routing rules are in
[test_config.yaml](../../../../../../.github/workflows/scripts/test_config.yaml).

- `a2a3` tests normally run on both A2 and A3.
- `a5` tests run only on A5.
- The A3-only `attn_res_fwd`, `gmm_dequant_situ_quant_msd`,
  `copy_sfa_offload_metadata`, `lim_mtp_index_reuse` and `sparse_kv_offload`
  tests have specific A3 routes within `ascendc/a2a3`.
- Generic Triton tests are classified under the current A2/A3 coverage group.
  This placement does not establish whether an untested A5 path is supported.
- Mixed tests with existing A5 coverage have an A5 variant. FP8 RoPE and the
  A5 DCP batching cases are excluded from their A2/A3 variants. The known
  A2 SISO shape skips remain in the A2/A3 RoPE test.

Adding a test requires choosing its backend and architecture directory.
Preserve `__init__.py` files so pytest can distinguish identically named
tests in different architecture packages. Use operator names for new test
files; split independently tested native operator interfaces into separate
files. Several files covering different contracts of the same operator may
remain separate.

The CI launcher exposes one assigned NPU before importing accelerator
packages. Runner allocation can contain additional cards; it does not change
the single-device test contract. CI selects and routes individual test files,
so local runs should select files supported by the current device rather
than collect every architecture directory together.
