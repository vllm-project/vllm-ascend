# QuantSparseLightningIndexerMetadata（QSLI metadata，AICPU）

## 功能

`VllmAscendQuantSparseLightningIndexerMetadata` 是 QLI v2 主算子 **candidate 模式（mode=2）**
配套的独立 AICPU 调度算子。它在设备侧计算每个 AI Core 应处理的
`(batch, gS1 块, 候选 tile)` 区间，供主算子的候选驱动路径使用。

## 与 QLI metadata 的关系（重要）

- **两份 metadata 分开实现、不可混用**：QSLI metadata 的输出是 QSLI 私有语义的
  `int32[1024]`（前 36 条 LI、第 36 条起 LD 记录，布局与 QLI 相同但**字段语义不同**）；
- 本算子的工作量按 `max(L) × 8` token、**候选 tile = 256 token** 计算；
  QLI metadata 按可见 S2 长度、S2 tile（910B 为 2048 token）计算；
- 主算子在接收到 `candidate_block_length` 时必须搭配本算子生成的 metadata，
  不能使用 QLI metadata 替代。

## 输入 / 输出

| 参数 | 类型 | 说明 |
|---|---|---|
| `candidate_block_length` | INT32 `[T,1]` 或 `[T]`，**必选** | 每行有效候选块数（DSL 语义，一候选块 = 8 token） |
| `cu_seqlens_q/k`、`seqused_q/k`、`cmp_residual_k` | INT32，可选 | 序列长度信息，语义同 QLI |
| 属性 | | `num_heads_q/k`、`head_dim`、`topk`、`quant_mode`、`layout_q/k`、`mask_mode`、`cmp_ratio` 等 |
| `metadata` | INT32 `[1024]` | 输出，QSLI 私有调度信息 |

## V1 范围

- **不做 FD/S2 分片**（`supportFd_ = false`）：按整行（gS1 块）调度、主算子直出；
  原因是主算子"候选 × FD 归并"路径尚未通过正确性验证（候选分片归并 bug）。
- 工作量按 `max(L)×8`，分核粒度是一个 gS1 块（mBaseSize = 4 行）。

## 已知问题（2026-11-03）

主算子的候选驱动路径在 **1~2 token 的 gS1 块**（M=32/64）上分数不正确
（4-token 块正常）。本算子本身只负责调度，不受影响；但端到端使用前需先修复主算子
（见 `scripts/prof_output/qli_v2_caseopt/RESULT_candidate_driven.md` §9）。

## 构建

```bash
cd csrc
source <CANN>/set_env.sh
bash build.sh --pkg --soc=ascend910b --vendor_name=<name> \
  --ops=quant_lightning_indexer_v2,quant_lightning_indexer_v2_metadata,quant_sparse_lightning_indexer_metadata -j32
```
