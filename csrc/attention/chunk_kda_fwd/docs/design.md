# ChunkKdaFwd 设计

## 目标

1. 顶层接口对齐非 CP 的 FLA `chunk_kda_fwd`。
2. 不新增公开算子原型；A5 快路径复用既有 `ChunkKdaFwd` 原型和外层 kernel 入口。
3. A2/A3/A5 使用同一数学定义；A5 保留 regbase 双发射特化。
4. 输入 layout 与输出 layout 解耦。
5. FwdH 同时服务 KDA 与 GDN，并支持可选 scalar gate、key-wise gate 和 `state_v_first`。

## L2 调度

```text
raw g -> ChunkKdaFwd[
    gate cumsum -> Prepare/Post-WU -> FwdH -> Finalize
] -> attn_out
```

`aclnnChunkKdaFwd` 做公开 layout 的连续化和必要视图转换。A5 的 BF16、chunk=64、K=V=128
dense 对齐快路径保持单次物理 `ChunkKdaFwd` L0；A5 其他多 chunk 场景将同一个私有 L0 按
Gate/Prepare、Post-WU、FwdH、Finalize 四个阶段依次提交，使阶段间通过物理 launch 边界重置事件状态。
A2/A3 和单 chunk 场景仍使用单次物理 L0。阶段选择仅使用私有 `stage` 属性，不增加公开属性、
接口字段或独立算子原型。

## 阶段职责

### KdaGateCumsum

将 raw/已激活 gate 转为 FP32 chunk-local log2 累计值：

```text
gk = cumsum(gate) / ln(2)
```

该算子同时保留独立 L2 接口供 GDN2 调用，输入和输出固定为 BNSD/NTD。

### Prepare

只读取 `q/k/v/gk/beta` 及变长元数据，产生：

```text
Aqk, Akk, qg, qg_scaled, w_seed, u_seed
```

矩阵计算和三角求逆使用 FP32 累积；公开中间量在写回时转为 q dtype。

### Post-WU

只读取 `k/gk/w_seed/Akk/u_seed`，产生：

```text
w, u, kg, v_new_seed
```

`Akk` 的 head 循环按 `H_v` 执行，GQA 映射只在读取 q/k head 时换算，避免按 `H_k` 重复或漏算。

### FwdH state propagation

读取 `kg/w/u/gk` 和可选 `initial_state`，计算 chunk 间递推：

```text
v_new = u - w @ h_prev
h_next = exp2(gk_last) * h_prev + kg^T @ v_new
```

arch35 路径复用与 `ChunkGatedDeltaRuleFwdH` 相同的数学实现；其他场景在 `ChunkKdaFwd` 内嵌
共享 FwdH 实现。独立 GDN L0 原型继续保留给其他调用方，key-wise `gk` 固定使用 `exp2`。

### Finalize

只读取 `qg_scaled/Aqk/v_new/h`，计算：

```text
attn_out = qg_scaled @ h + Aqk @ v_new
```

kernel 内直接按 BSND/TND 写出 `attn_out`。供反向使用的中间量保持 BNSD/NTD。

## 状态布局

内部递推统一使用 `[...,K,V]`。`state_v_first=true` 时，L2 在进入 FwdH 前转置 initial state。
内部 `hCompute` 始终保持 head-major 供 Finalize 消费；公开 `hOut` 在 L2 导出边界转为
sequence-major，并按 `state_v_first` 决定末两维顺序。`final_state` 按序列排列，与 FLA 顶层
输出一致。

## 重计算策略

L2 不理解 autograd 重计算策略。`final_state/gk/w/u/qg/kg/v_new/h` 是相互独立的
`OPTIONAL_OUTPUT`；非空指针表示导出，空指针表示不公开该结果。单 launch 路径为隐藏输出传递
固定 ABI 占位，并由 tiling 在 kernel workspace 中承接实际中间结果。A5 四段 launch 路径将
阶段间依赖的 `gk/w/u/qg/kg/v_new/h/final_state` 和私有 `qg_scaled/u_seed` 物化为 executor 内部张量，
使后续 launch 不依赖前一 launch 的 kernel workspace。公开输出存在时直接作为内部目标使用。

Python/legacy 包装层对齐 fla-org `chunk_kda_fwd` 提交
`0f0f0c97af39343855b43bbbaddcedfda5cb9d77`：

- `Aqk/Akk` 始终返回。
- `disable_recompute=false` 时不保留 `w/u/qg/kg/v_new`。
- `disable_recompute=true` 或 `return_intermediate_states=true` 时保留公开 `hOut`。
- `use_gate_in_kernel=false` 或 `disable_recompute=true` 时保留 `gk`。
- `final_state` 只在 `output_final_state=true` 时创建公开输出。

内部 `hCompute` 与公开 `hOut` 是两个生命周期：`hCompute` 是 FwdH 到 Finalize 的必需
head-major 阶段结果；`hOut` 为空时，单 launch 路径由 kernel workspace 承接，四段路径由 executor
内部张量承接。`hOut` 非空时，L2 提供 head-major 临时输出并在导出边界转为 sequence-major。
该规则对齐非 CP 的低层 12 返回值接口；
第 12 项 `initial_state` 由 Python 层原对象透传。

## 模板化方案与 tiling key

`ChunkKdaFwd` 只有一个外层 `op_kernel/chunk_kda_fwd.cpp` 入口和一个私有 L0 类型。A5 实现位于
`op_kernel/arch35/*.h`，host 侧 A5 模板选择位于 `op_host/arch35/*.h`。Prepare、Post-WU、
Finalize 的内部实现头与统一 kernel 入口同属 `chunk_kda_fwd/op_kernel/`，不存在对应的独立 L0
原型或 `.cpp` 入口。A5 四段路径只是用不同私有 `stage` 属性连续调用该入口。

- `tiling key=1`：非 chunk=64、K=V=128 场景的通用模板族。
- `tiling key=2`：chunk=64、K=V=128 模板族，包括 dense、tail 和 varlen。

两个 key 是同一 L0 的编译期场景变体，不是平台编号、独立算子或独立接口。A2/A3/A5
均生成两个 key；同一个 key 内再由编译架构选择根目录通用实现或 `arch35/` 实现。host 的
`SetTilingKey` 只检查 chunk、K、V，不检查 SoC。

在 arch35 上，key2 的 dense 对齐场景使用单 launch 和 arch35 FwdH；融合 score 写回在跳过
共享 PostWU 时会额外物化以块尾 gate 为参考的最终 `kg`，供 FwdH 和可选公开输出共同使用。
A5 多 chunk 的 tail/varlen 以及 key1 泛化场景使用四段 launch。其他架构在同一 key 下使用其
对应单 launch 实现。tiling key 和私有 `stage` 均不改变公开算子原型、输出契约或数学定义。

## 性能设计

- Prepare 的右矩阵在 L1 驻留，避免 K/K^T 重复搬运和重复转置。
- AIC 使用 L1/L0 双缓冲组织 MTE2、MTE1、Cube、Fixpipe。
- AIV 使用输入 staging ping-pong，使下一 tile MTE2 与当前 tile VEC 重叠。
- A5 VEC 路径使用 regbase 双发射；数值主计算仍保持 FP32。
- inter-sub-chunk 合并使用独立 workspace 区域，避免阻塞主 tile 流水。

性能结论只使用 `msopprof`。目标回归 case 定义在 `tests/op_cases/chunk_kda_fwd.json`。

## 验证矩阵

- 平台：A2/A3/A5。
- dtype：FP16/BF16。
- layout：BSND/BNSD/TND/NTD。
- gate：raw/已激活、safe true/false。
- Shape：K=128，V=128/256，chunk=64/128，dense/varlen/tail/GQA。
- 属性：final state、重计算策略、`state_v_first`。
