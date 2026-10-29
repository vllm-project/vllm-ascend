# AttnResFwd

## 功能说明

`AttnResFwd` 是 Kimi K3 的注意力残差融合算子。单个 Torch 入口
`torch.ops._C_ascend.attn_res_fwd` 将可选的 BF16 加法、残差 bank 混合、
可选的输出 RMSNorm，以及 bank 写回放在一次调用中。当前算子配置面向
Ascend 950（A5）推理；A3 运行行为尚未验证。

令 `p` 为 `prefix_sum`（有 `addend` 时先做 BF16 加法），`v_i` 为
`block_residual` 中前 `num_valid_blocks` 行，并令最后一行 `v_N=p`。
启用 `mix` 时，每行先按 `norm_eps` 做 RMSNorm，再与 `norm_weight`、
`proj_weight` 相乘并沿隐藏维求和，所得分数经 softmax 后对原始 `v_i`
加权求和并舍入到 BF16。`mix=False` 时直接使用 `p`。传入
`output_norm_weight` 时，对混合结果再做一次 RMSNorm。

## 接口与参数

```python
output, raw_prefix, materialized = torch.ops._C_ascend.attn_res_fwd(
    prefix_sum, addend, block_residual, proj_weight, norm_weight,
    norm_eps, num_valid_blocks, output_norm_weight=None,
    output_norm_eps=1e-5, block_write_idx=-1,
    return_materialized=False, mix=True, optimize_prefill=False,
)
```

| 参数 | 形状与约束 | 作用 |
| --- | --- | --- |
| `prefix_sum` | `[T,H]`，BF16，连续 | 当前层的残差前缀。 |
| `addend` | `None` 或 `[T,H]`，BF16，连续 | 可选的前层输出；加法结果以 BF16 供后续计算。 |
| `block_residual` | `[T,S,H]`，BF16；每行连续，token 间可有 stride | 残差 bank；仅前 `num_valid_blocks` 行参与混合。 |
| `proj_weight`、`norm_weight` | 分别为 `[1,H]`、`[H]`，BF16，连续 | 混合分数的投影和 RMSNorm 权重。 |
| `norm_eps`、`num_valid_blocks` | `norm_eps > 0`；`0 ≤ num_valid_blocks ≤ S` | 混合归一化参数和有效 bank 行数。 |
| `output_norm_weight`、`output_norm_eps` | `None` 或 `[H]`，BF16，连续；传入权重时 `output_norm_eps > 0` | 可选的输出 RMSNorm。 |
| `block_write_idx` | `-1` 或 `[num_valid_blocks,S)` 中的行号 | 将 `raw_prefix` 写入尚未参与本次混合的 bank 行；`-1` 表示不写回。 |
| `return_materialized` | `bool` | 为真时单独保留归一化前的混合结果。 |
| `mix` | `bool` | 为假时跳过 bank 混合，用于前层或只做加法的路径。 |
| `optimize_prefill` | `bool` | 请求 prefill 策略；满足输出归一化、每核多 token 和 UB 容量条件时复用缓存权重。 |

所有 Tensor 必须位于同一设备。`output`、`raw_prefix` 和
`materialized` 均为 `[T,H]`、BF16。无 `addend` 时 `raw_prefix`
复用输入 `prefix_sum`；`return_materialized=False` 时
`materialized` 复用 `output`。写回只修改指定的 bank 行，不修改
`prefix_sum` 或 `addend`。`T=0` 可走空输出路径。

## 实现与验证

Host tiling 根据 H、有效行数和 UB 容量选择 RESIDENT 或 RELOAD；
`optimize_prefill` 仅在满足缓存条件时切换对应的 PREFILL tiling key，
不增加公开算子入口。模型中的层内和末层调用均使用同一个 Torch 入口。

单卡精度用例位于
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_attn_res_fwd.py`，
使用独立 CPU 参考值覆盖 Kimi K3 的 H=7168、decode/prefill、
可选加法、输出归一化和写回。该用例不代表整网 TP/EP/DSpark 吞吐验收。
