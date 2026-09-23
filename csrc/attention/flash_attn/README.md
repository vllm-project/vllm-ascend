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
No service command or new runtime switch is required.

MLA keeps the compressed persistent KV layout and expands history in bounded
chunks in the existing attention backend. The kernel page size remains 128.
