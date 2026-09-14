# A5 packed MLA operators

FlashMlaWithKvcache and its device scheduling metadata producer use
ops-transformer commit `8d5d69c35e6517c064158b6e0c40267f66856533`.
These sources retain their CANN license in `LICENSE` and compile through the
normal A5 operator list in `csrc/build_aclnn.sh`. The local changes preserve
physical query offsets, live query counts, and the packed cache page stride.

The selected CANN 9.2.0 B035 package already supplies FlashAttn,
FlashAttnMetadata, and ScatterPaKvCache. Their framework bindings use those
installed APIs. FlashMlaWithKvcache and its metadata API are absent from both
the inspected B035 and B060 packages, so these two native operators remain
part of the vllm-ascend build, together with CausalConv1dV2.

Build in the selected CANN environment with `MAX_JOBS=256` or higher.
The bindings use the existing `torch_binding.cpp` and `EXEC_NPU_CMD` path;
no separate operator package or runtime source patching is required.

`VLLM_ASCEND_ENABLE_FLASH_MLA=1` enables the opt-in route in the existing
attention backends. Initial support is BF16 dense MLA, unquantized KV,
SD convolution states, and PCP=1. DCP uses local paged history plus replicated
current-token pages for causal attention, and local full-cache attention for
non-causal DSpark drafting. Service validation follows the build.

P/D disaggregation is supported with `MooncakeConnectorV1`. Producer and
consumer must use the same Flash-MLA flag, TP/DCP sizes, interleave size, cache
dtype, and kernel block size. Only the persistent paged cache is transferred;
the replicated current-token cache is per-step scratch space. The combined
DSpark path currently covers MLA draft backends (including causal block
drafting); GQA draft backends are outside this Flash MLA DCP path.
