# A5 FlashAttn for Kimi K3

This source snapshot adds the non-absorbed MLA prefill combination QK192/V128
to the official `flash_attn` and `flash_attn_metadata` operators. The existing
absorbed `flash_mla_with_kvcache` decode operator is separate and unchanged.

The A5 build retains the K3 call shapes:

- TND Q/K/V: QK192/V128 for MLA prefill, D64/V64 for GQA current chunks.
- TND Q and PA_BNBD KV: D64/V64 GQA draft attention, including page strides.
- Causal current chunks and noncausal history chunks.

These template selections do not claim general support for every layout and
dimension accepted by the upstream operator. Shared helpers are reused from
`a5_mla_common`.

Build through the normal vLLM-Ascend A5 build. The metadata native ABI and
official torch bindings must be installed together: `head_dim_v` is optional
and defaults to the QK dimension for existing GQA callers. The build packages
the selected CANN Python package, changes only its FlashAttn wrapper/source,
and precompiles that extension. Other CANN operator modules are preserved.
The GQA backend calls FlashAttn for A5 BF16 D64 eager PrefillNoCache
batches with no sliding window, sinks, PCP/DCP, or batch-invariant mode.
Other paths retain their existing dispatch, including graph capture.
No service command or new runtime switch is required.

The expanded MLA adapter in `vllm_ascend/attention/flash_attn.py` exposes
BF16 QK192/V128 and C8 preparation/attention calls. C8 uses per-row query
scales and per-head K/V scales. Its metadata tile-selection bound is at
least 65 while runtime sequence lengths stay exact. This PR does not
activate non-absorbed MLA prefill in the MLA backend or change its cache.
The C8 kernel and preparation operators build in the A5 custom-op list.

CPU contract tests cover scheduling, shapes, C8 runtime lengths and the
GQA backend dispatch. The single-card C8 tests require an A5 environment;
they cover preparation, fake quantization and native attention accuracy.
