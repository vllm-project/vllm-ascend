# Kimi KDA fused prefill 的 V2 迁移

## 算子与启用条件

参考 `27.0_kda` 分支的 `_run_prefill_fused_kda`，实际导出为
`ops.flash_kda.flash_kda` 和 `ops.flash_kda_metadata.flash_kda_metadata`。
两者在 worker 构造 attention 时加载；BF16、bounded gate 且两个导出均可用时，
prefill 自动走 FlashKDA。缺少依赖、其他 dtype、无 bounded gate 或传入已激活 beta
时使用现有 AscendC 路径。此选择独立于 `enable_mega_kda`，未新增环境变量。

算子 ABI：Q/K/V/raw gate 为 `[T,H,D]`，raw beta 为 `[T,H]`，请求边界为设备端
int32，初始及最终 SSM 状态为 FP32 `[B,H,V,K]`。Q/K normalization、gate 激活、
beta sigmoid 在算子内部完成。混合 decode/verify 批次仍为 recurrent 路径单独计算
sigmoid，避免重复激活 prefill beta。

## cache 生命周期

1. 沿用已有 connector wait，等待本层 cache 可用。
2. 沿用 `causal_conv1d_fn` 更新原 Conv cache。
3. 复用 V2 绑定 cache 后初始化的 state-copy plan，按请求索引 gather SSM；
   `has_initial_state=False` 的行在临时初始状态中清零，continued prefill 保留旧状态。
4. 直接复用 metadata 的设备端请求边界；空片段同步压缩边界、索引和初始状态标志。
5. 调用 FlashKDA，校验输出及最终状态，再 scatter 到原 SSM cache。
6. 沿用已有 connector save；后续 MegaKDA 直接读取这些原始 cache 页。

不重绑 runner cache，不修改物理分配、stride、storage offset 或页所有权，
不转置 VK 状态。scatter 仅更新选中的有效页；其他请求与页间保护字节保持原样。
算子执行异常直接上抛，不在同一批次重试；此前 Conv cache 可能已经更新。
prefill 沿用 `_forward` 的 eager capture boundary，不新增 full-graph prefill 承诺。

## 验证命令

CPU 契约回归（无需导入 NPU/vLLM runtime）：

```bash
python -m pytest --noconftest -q \
  tests/ut/ops/test_kimi_prefill_fused_kda.py \
  tests/ut/ops/test_kimi_mega_kda.py \
  tests/ut/worker/test_mega_kda_v2.py
```

Ascend 上安装对应 FlashKDA 与当前 MegaKDA 包后，运行真实算子验收：

```bash
pytest -sv tests/e2e/nightly/single_node/ops/test_kimi_prefill_fused_kda.py
```

该测试对比 AscendC 的输出及完整 Conv/SSM cache，覆盖连续与 page-strided 布局、
chunk 尾部、空请求、新请求清零、continued prefill、请求重排及 prefill 后首次
MegaKDA verify，并记录含 gather/scatter 的 prefill 延迟。依赖缺失会失败。

服务沿用现有 Kimi 模型与部署参数，通过已有的 `VLLM_USE_V2_MODEL_RUNNER=1` 选择 V2；
需要 MegaKDA decode/verify 时保留 `--additional-config '{"enable_mega_kda":true}'`。
在真实权重服务中检查文本请求、chunked prefill、混合调度与 MTP，再进行性能 A/B。

## 本机验证范围

本机为 Windows CPU 环境，完成 CPU 调用契约与 MegaKDA/V2 回归、Python lint 和语法检查。
NPU 数值、吞吐、ACLGraph、TP/SP、PD 传输及真实权重服务尚未验证；
合成算子测试也不能替代真实权重验收。
