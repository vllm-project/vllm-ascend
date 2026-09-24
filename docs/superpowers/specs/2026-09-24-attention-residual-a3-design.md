# Attention Residual A3 适配设计

## 目标

将 A5 attention residual 算子族适配到 A3（arch22），保持与 A3 原有小算子组合一致的语义和精度，并使融合算子在代表性 shape 上不劣于等价小算子组合。

## 范围

覆盖 `attn_res_fwd`、`attn_res_fwd_with_add`、`attn_res_fwd_fused` 和 `attn_res_fwd_prefill`。A5 保留 arch35 实现；A3 只使用 arch22 kernel 和 A3 对应的 host 注册。

## 语义合同

- `raw_prefix`、`materialized`、bank 写回和输入保护必须 bitwise 对齐 A3 小算子参考。
- `prefix + addend` 必须保留 BF16 舍入边界。
- 最终 output RMSNorm 使用 `rtol=1e-2, atol=1e-2`。
- 所有输出不得包含 NaN 或 Inf。
- 支持 valid=0、1、2、4、8、63、64，以及连续、slice、strided 和 transpose 输入。

## 实现设计

### 1. A3 向量 helper

`SubLastDimRow1NoBrc` 和 `MulLastDimRow1NoBrc` 使用 A3 可接受的批量向量路径：

1. 对每个完整 64-FP32 元素块执行一次 repeat 操作；
2. 对不足 64 的尾部执行一次对齐 tail 操作；
3. `curColNum == 64` 直接走完整块快速路径；
4. 仅在批量块和 tail 之间保留必要的 `PIPE_V` barrier；
5. 不使用每 8 元素一次的循环作为最终实现。

所有 UB 地址、repeat stride、tail 起始地址和临时 buffer 都必须满足 A3 32B 对齐约束。

### 2. 融合路径

复用 arch22 reload/resident 的数据搬运和 FP32 中间计算，保持 host tiling ABI 不变。with_add、fused、prefill 分别保留独立 kernel entry；prefill cache 只有在 A3 性能实测收益为正时启用。

### 3. 性能基线

性能基线不是修复前的 `opp_final`，而是 A3 上等价的小算子组合：add、RMSNorm、score、softmax、weighted sum、可选 output RMSNorm 和 bank/materialized 写回。每个 case 使用相同输入、warmup、重复次数和 NPU 设备计时。

## 验收标准

1. A3 编译、安装和算子注册成功；A5 arch35 构建不受影响。
2. 官方单算子测试和直接 reference 测试通过，覆盖上述语义合同。
3. 所有代表性 shape 的融合算子 P50/P90 不高于等价小算子组合；若某条 prefill 路径不满足，则 A3 不启用该路径。
4. 记录构建、精度、性能和环境信息，生成可复现报告。
