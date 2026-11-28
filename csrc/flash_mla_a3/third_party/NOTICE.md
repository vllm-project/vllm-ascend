# FlashMLA A3 third-party sources

The A3 kernel adapts `src/shared_lib` from [TyphoonMLA in CANN ops-transformer](https://gitcode.com/cann/ops-transformer/tree/master/experimental/attention/typhoon_mla).
The adaptation in `native/catlass` supports 576-channel paged inputs, a right-aligned causal mask for four query tokens,
nonuniform split merging, device-side scheduling inputs, and direct writes in contiguous NTD layout.
The original source notices are retained, and `native/catlass/LICENSE` contains the CANN Open Software License Agreement Version 2.0.

`catlass-1.0.0-headers.tar.gz` contains the `include` directory and original `LICENSE` from
[CATLASS v1.0.0](https://gitee.com/ascend/catlass/releases/tag/v1.0.0)
(commit `ef8be58db449a7271cf260c69fb0a6e02093946a`) for offline builds.
The overrides in `native/catlass/src/catlass` adapt memory access and pipelining for A3. The archive SHA-256 is
`2b388ba9c00519ec7801435c9f87400e5f135dfba7ff715762d60f9d8f9d32cf`.
