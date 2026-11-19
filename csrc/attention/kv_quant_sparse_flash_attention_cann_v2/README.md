# KvQuantSparseFlashAttentionCannV2

此算子移植自 CANN `ops-transformer` 的
[`v9.2.1`（`8fbf1ffea0cf`）](https://gitcode.com/cann/ops-transformer/tree/v9.2.1/attention/kv_quant_sparse_flash_attention_v2)。
V2 已包含在正式标签 [`v9.2.0`](https://gitcode.com/cann/ops-transformer/tree/v9.2.0/attention/kv_quant_sparse_flash_attention_v2)。
导入内容包含 V2 的 Ascend 950 入口、shape 推导、共享 arch35 kernel、
tiling 和依赖头文件。
`op_kernel/arch35/upstream` 中的头文件来自同一标签的 `attention/common` 和
`attention/sparse_flash_attention`。移植源码遵循随目录附带的 [CANN 许可证](LICENSE)。

算子包使用独立的 `KvQuantSparseFlashAttentionCannV2` 名称，在 Ascend 950 构建中启用。
从源码构建时使用 A5 镜像或对应的 CANN 工具链，并启用 `COMPILE_CUSTOM_KERNELS=1`。

SFA 调用采用 TND query、PA_BSND KV、64 维 RoPE 和 128 元素量化 tile。
FP8 E4M3FN 与 INT8 均使用 656 字节的 packed KV 行。Torch 绑定支持可选的
FP32 `sinks` 和 `return_softmax_lse`，开启后返回 `softmax_max` 和 `softmax_sum`。
常规 RoPE64 SFA 接入移植版。DCP、RoPE0 和 Kimi K3.1 使用现有
[`KvQuantSparseFlashAttentionVllm`](../kv_quant_sparse_flash_attention_vllm/README.md)。

在已构建并加载该算子的 A5 环境运行回归用例。

```bash
python -m pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_kv_quant_sparse_flash_attention_vllm.py -k test_a5_c8_lse
```
