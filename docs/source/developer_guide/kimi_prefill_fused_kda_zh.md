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

## metadata 跨层复用

参考 [recipe PR #29](https://gitcode.com/haittbakerhall/cann-recipes-infer_kimik3/pull/29)，
本仓的 FlashKDA prefill 默认使用 metadata 副流，并在同一次 forward、同一个
attention group 的兼容层间共享 metadata，无需额外配置开关。

metadata 算子只从 Q/V/initial_state 读取描述信息，从 `cu_seqlens` 读取请求边界；
首层生成的调度信息可以供张量规格相同的后续层使用。每层仍传入自己的 Q/K/V 和状态，
仍执行必要的 cache gather/scatter。共享上下文挂在本次调度新建的 GDN prefill metadata 上，
包括空请求压缩后的 int32 边界；不同 forward、attention group、模型/草稿阶段不共享调度内容。
跨层规格或 metadata 接口不一致时明确报错。

环境中的 `ops.flash_kda_metadata` 需要额外导出 `metadata_capacity_upper_bound`，缺失时启动报错。
builder 初始化时，按 `max_num_seqs` 和 `max_num_batched_tokens` 在 metadata 副流预分配
int32 缓冲区。每次 forward 的首层在副流生成临时 metadata，拷入固定缓冲区后释放临时结果；
后续层只读复用。流事件保证边界准备完成后生成、生成完成后消费，以及前一轮消费者结束后
才覆盖缓冲区；不进行设备全局同步，不在推理中扩容。容量不足或算子返回的 dtype/device/大小
不符合容量接口约定时立即报错。

在单个 attention group 包含 10 个兼容 KDA 层、每 step 只有一次模型 forward 的情况下，
迁移前预期 metadata kernel 10 次、FlashKDA kernel 10 次；当前默认预期分别为 1 次和 10 次，
另有一次 metadata 设备内拷贝。多个 group、microbatch 或多次模型 forward 需分别计数，
不能把 scheduler step 与单次模型 forward 混为一谈。真实收益需在目标 NPU 上 profiling 验证。

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
  tests/ut/ops/test_kimi_prefill_metadata.py \
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

本次 PR #29 的 metadata 迁移已通过 lint、格式、语法和使用生产代码的 mock 生命周期检查，
覆盖 10 层复用、下一轮刷新、流依赖、布局不匹配、消费异常及容量检查。
新增 PyTorch CPU 回归 `tests/ut/ops/test_kimi_prefill_metadata.py`，以及 NPU 跨层/跨轮
精度对照与性能记录。尝试在隔离 Python 3.12 环境运行 CPU 回归时，PyTorch 在导入
`c10.dll` 时发生 Windows `WinError 1114`，测试收集失败，尚未执行测试代码。
Windows 当前未提供 Bash/pre-commit，无法执行完整的 `bash format.sh ci`。
