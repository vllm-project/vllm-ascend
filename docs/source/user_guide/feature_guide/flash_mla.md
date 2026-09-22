# A5 FlashMLA

`VLLM_ASCEND_ENABLE_FLASH_MLA=1` selects FlashMLA for dense MLA layers. The
switch defaults to `0`; GQA and recurrent-state cache layouts retain their
existing paths.

The backend supports BF16 activations, one KV head, a 512-dimensional latent
cache and a 64-dimensional positional component on Ascend A5. Persistent cache
is BF16 or calibrated FP8 E4M3 with a BF16 positional component. PCP must be 1.
DCP head exchange uses the existing process group. Kernel pages contain 128
tokens, including when the manager uses larger pages.

## Execution

The metadata builder prepares stable query boundaries, used lengths, slot
mappings and native schedules from device lengths. MRv1/MRv2 refresh these
buffers before the model consumes them. Padding rows have no cache writes and
zero output. Dense MLA draft capture uses the same metadata contract.

C8 decode quantizes the latent query, preserves the BF16 positional component,
and passes calibrated KV and dynamic query scales to the native reader.
Split history/current execution keeps current tokens in BF16 and includes them
in the LSE merge exactly once. Full prefill expands bounded history chunks.
Mixed batches retain a separately scheduled decode prefix.

Each C8 page owns two dense planes: `128 * 512` FP8 bytes followed by
`128 * 64` BF16 elements. Cache views and block zeroing preserve page pitch,
manager padding and neighboring layers.

If the optional MLAPO package exports `npu_mla_prolog_v3` and
`npu_mla_prolog_dcp_c8`, supported NoPE shapes use fused preprocessing.
Otherwise ordinary projection and quantization implement the same C8 contract.
Kimi TP8/DCP8/PCP1 uses the upstream replicated-query projection before loading
weights.

## Build and validation

The A5 build includes `flash_mla_with_kvcache`, its AICPU metadata producer,
`gather_mla_prefill` and `flash_mla_bf16_prepare`. Expanded prefill requires
matching FlashAttn bindings with 192-dimensional Q/K and 128-dimensional V.
Build from the target device's detected SOC as usual.

CPU tests cover production orchestration with emulated native operators,
metadata ownership, cache views and zeroing boundaries. They do not establish
native numerical accuracy. On A5 with the native extension installed, run:

```bash
pytest tests/e2e/nightly/single_node/ops/test_flash_mla_contract.py
pytest tests/e2e/nightly/single_node/ops/test_flash_mla_c8_module.py
pytest tests/e2e/nightly/single_node/ops/test_gather_mla_prefill.py
pytest tests/e2e/nightly/single_node/ops/test_flash_mla_bf16_prepare.py
```
