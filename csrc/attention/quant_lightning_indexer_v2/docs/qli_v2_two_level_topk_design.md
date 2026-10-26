<!-- markdownlint-disable MD040 MD055 MD056 -->

# QuantLightningIndexerV2 两级 TopK（候选块选择）方案设计 — 仅 arch22 / 910b

> **Provenance**: 本方案基于 `ops-transformer` 仓库 `ds41` 分支 commit `32d64c27f`（提交MegaMoeWave模板）的代码调研。
> 参考语义来源: 用户提供的 `select_candidate_blocks` PyTorch 参考实现（口头描述，2026-09-06）。
> 适用范围: `attention/quant_lightning_indexer_v2`（主算子）+ `attention/quant_lightning_indexer_v2_metadata`（配套算子），**仅 910b (arch22) 路径**，arch35 (950) 不在本次范围。

---

## 1. 需求分析

### 1.1 模型侧语义（来自参考实现）

```python
# Level One: 每个query选 topk_blocks 个最高分的块（block_size 个位置为一块）
def select_candidate_blocks(logits, compress_lens, topk_blocks=2048, block_size=8):
    width = logits.size(-1)  # S2（压缩后KV长度）
    scores = F.pad(logits, (0, -width % block_size), -inf)  # 尾块补 -inf
    scores = scores.unflatten(-1, (-1, block_size)).amax(-1)  # 块内取max → [.., num_blocks]
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size  # 最后一个未满块
    scores[last] = +inf  # pin：最新token所在块无条件保留
    top = scores.topk(min(topk_blocks, num_blocks))
    keep = scatter(top.indices, top.values > -inf)  # 丢弃全 -inf（不可达）块
    return keep.repeat_interleave(block_size, -1)[..., :width]  # 位置级 bool mask


# 调用侧（跨层共享 shared_attn.candidates）:
if self.is_candidate_source:  # source 层：产出候选块
    shared_attn.candidates = select_candidate_blocks(index_score, compress_lens, 2048, 8)
elif self.uses_candidates:  # consumer 层：候选块外 score 置 -inf 后再走原 topk
    index_score = index_score.masked_fill(~shared_attn.candidates, -torch.inf)
```

关键点提炼：

1. **块定义**: 位置 p 属于块 `floor(p / 8)`，块数 `num_blocks = ceil(S2 / 8)`，S2 为**压缩后**的 key 有效长度。
2. **块得分**: 块内 8 个位置的 score 取 **amax**；不可达位置在 logits 中已是 -inf，不抬升块得分（与 pad 语义一致）。
3. **pin 规则（compress_lens 形态决定，两种）**：
   - **decode / mid-chunk prefill（模型 else 分支，op mask_mode=0）**：`compress_lens = end_pos // ratio` 为标量（= actS2Size），pin 块号 `(actS2Size - 1) // block_size`，batch 级（该 batch 所有行 pin 同一块）；
   - **prefill 首块（模型 start_pos==0 分支，op mask_mode=3）**：`compress_lens = (arange(1, seqlen+1) // ratio)` 为行级 `[S1, 1]`，行 i 的 pin 块号 `(rowValidLen(i) - 1) // block_size`，其中 `rowValidLen(i) = (actS2SizeOrig - actS1Size + i + 1) // cmpRatio`（与 op mask_mode=3 的行级有效长度公式逐行一致：首块时 actS2SizeOrig=actS1Size=seqlen → `(i+1)//ratio`，与模型代码吻合）→ **每行 pin 各自的最后一块**；
   - 保证最新 token 所在块必被选中的语义在两种形态下均成立。
4. **有效槽**: 可达块数不足 `topk_blocks` 时，未被选中的槽位无效（值 `-inf` 被丢弃）。
5. **Level-2 动态 topk（模型 gather 侧，无需 op 改动）**：
   ```python
   topk = min(self.index_topk, end_pos // ratio)  # 动态 topk = min(sparse_count, actS2Size)
   idxs = index_score.topk(topk, sorted=False).indices.sort(dim=-1)  # 升序
   return torch.where(idxs < compress_lens, idxs + offset, -1)
   ```
   - `topk` 属性为每次调用传入的动态值（[1,2048]），模型传 min 值即可；
   - `where(idxs < compress_lens, ·, -1)` 与 kernel 现有语义**等价**：不可达槽位输出 -1（`InitSortOutBuf` 填 (-inf,-1)，有效长度外沉底）；"topk=sparse_count + 滤 -1" 与 "topk=min 直接输出" 有效索引集合相同；
   - `idxs + offset` 由模型侧完成（910b 不支持 output_idx_offset，无需支持）；
   - **结论：无 kernel/host 改动，仅测试覆盖**（§6.3/§6.5 增加 topk<2048 动态值与 min 等价性用例）。
6. **source 层自身仍做正常 topk**（if/elif 互斥，source 不做 mask 但照常输出 sparse_indices）。
7. **候选块跨层共享**，同一 forward 内 KV 状态不变 → 块划分一致；**跨 step 复用无效**（语义约束）。

### 1.2 现有算子流水线摘要（arch22, 910b）

| 阶段 | 执行单元 | 内容 | 数据位置 |
|---|---|---|---|
| S1 | AIC `ComputeMm1` | Q(256=s1×g, 128) @ K^T(128, s2≤2048) → L0C | L1/L0 |
| S2 | AIC `FixpSToL1` | DEQF16(scale 0.001) + **ReLU** → half [s1g, s2] | L1 `sL1_` |
| S3 | AIC `ComputeWs` | w(s1g) @ ReLU(QK)(s1g, s2) → g 维求和 | L0 |
| S4 | AIC `FixpResToGm` | nz2nd → **score[s1, s2]** float | GM `mm1ResGm`（loop%2 双buffer） |
| S5 | AIV `ProcessVec0` | (w × qScale) 广播 → 供 S3 使用 | GM `vec0OutGm` |
| S6 | AIV `ProcessVec1` | score × kScale → SortAll(降序) → MergeSort 进 `globalTopkUb_`(2048 value/idx pair/s1行) → 行末 ExtractIndex 直出 `sparse_indices` | UB/GM |

> **LD（跨核归约）在 910b 不激活**：`supportFd_` 默认 false 且仅 ASCEND950 分支置 true（metadata aicpu.cpp:329-338, aicpu.h:285）→ `AssignByBlock`（S2 块级跨核切分）被跳过 → 单个 (b, s1) 行的 S2 永不跨核 → kernel `isNeedLD` 恒 false，`ProcessLD` 的 QLD_V2 元数据全 0。**本方案不考虑 LD 路径**（kernel 中的 LD 代码保留不动，仅不激活）。

结论：**`index_score`（参考实现中的 logits）= S4 输出 × kScale（S6 第一步之后）**，按 (s1行, s2 tile ≤2048) 流式产生，且每行 topk 结果在本核内独立完成。两级 TopK 的插入点在 S6（ProcessVec1）内部。

### 1.3 输入输出定义（新增部分）

**主算子 QuantLightningIndexerV2 新增（命名与模式编号按用户 2026-09-06 决策）**：

| 项目 | 名称 | 方向 | 类型/Shape | 约束 |
|---|---|---|---|---|
| 输入 | `candidate_topk_index` | 可选输入 | INT32, `[B, S1, N2, 2048]`（BSND） | `candidate_mode=2` 时必传；值域 `[0, numBlocks)` 或 `-1`（无效槽） |
| 输出 | `candidate_topk_index` | 可选输出 | INT32, 同上 | `candidate_mode=1` 时输出；无序，槽位语义：块号或 -1 |
| 输出 | `sparse_indices`（即 topk_index） | 既有输出 | 不变 | mode=1: 全量 score 的 topk；mode=2: **从 candidate_topk_index 展开的 ≤16384 个候选位置中再挑 topk 个**；mode=3: 现网行为 |
| 属性 | `candidate_mode` | 可选属性 | INT, 默认 **3** | **1**=source（is_candidate_source，输出候选块索引 + 照常输出 topk_index）；**2**=consumer（use_candidate，输入候选块索引，在候选内选 topk）；**3**=关闭（现网行为，默认值保证既有调用零影响） |
| 属性 | `candidate_topk_blocks` | 可选属性 | INT, 默认 2048 | **(0, 2048] 内 64 的倍数**（2026-09-07 放宽，原为仅 2048；上限受 BASE_TOPK=2048 的 sort/merge 结构约束；`numBlocks > topk_blocks` 时 pin 最新块必须入选，见 §9/P2）|
| 属性 | `candidate_block_size` | 可选属性 | INT, 默认 8 | 范围 [1, 64] 且为 2 的幂（向量 reduce 实现约束）；**source 与 consumer 必须一致** |

**设计决策 — 输出索引而非 bool（用户决策"不再输出 bool，直接输出 topk index"）**：

- `candidate_topk_index` 为**块级索引**（2048 个 int32/行，值=块号），展开后等价于 2048×8=16384 个位置级索引；相比位置级索引（16384 个 int32=64KB/行）**GM 内存压缩 8×**，相比 bool mask `[B,S1,S2]`（prefill 大序列可达 512MB）更是量级差异；
- mode=2 消费时按 `position ∈ [8×blk, 8×blk+8)` 展开，天然还原块内连续 8 位置；
- 既有输出 `sparse_indices` 名称不变（aclnn 兼容），文档语境中的 topk_index 即指它。

**int 块索引 vs bool mask 的权衡记录（2026-09-06，结论：本场景无实质不方便）**：

| 维度 | bool mask | int 块索引（本方案） |
|---|---|---|
| 算子外消费 | 直接可用 | 需先展开（届时在 torch 封装层提供 expand 工具即可，不必改算子接口） |
| 接口约定 | 无 | -1=无效槽、槽位无序（topk 序）、块号×8=起始位置，需文档化（README 约束项） |
| mode=2 kernel 实现 | 读 tile mask 段 + Select，简单 | membership（compare-OR 分批）+ 展开，复杂一档（S4a 已设计） |
| 调试/比对直观性 | 高 | 低（dump 为块号列表；比对已集合化 §6.2） |
| GM 内存/带宽 | S2 字节/行（最坏 512MB） | 8KB/行，**省 8×**（决定性理由） |
语义完备性说明：块内有效性由原 score 兜底（mask_mode=3 下行内不可达位置 score 本为 -inf），mode=2 的 masked score 与 bool 语义严格等价。

**metadata 算子：不改接口**（论证见 §3.2）。

### 1.4 数学语义（新增部分的形式化）

对每个 batch b、query 行 s1（n2=1, 910b 上 kHeadNum=1）：

```
score(b, s1, :)  ∈ R^{S2}            # 现有 S4×kScale 结果，不可达位置语义为 -inf
numBlocks(b)     = ceil(actS2Size(b) / block_size)
blkScore(b, s1, j) = max_{8j ≤ p < min(8j+8, S2)} score(b, s1, p)     # 尾块不足8位置按 -inf pad

# pin 块号（行级，随 mask_mode 取行级/批次级有效长度）:
rowValidLen(b, i) = (actS2SizeOrig(b) - actS1Size(b) + i + 1) / cmpRatio   # mask_mode=3（行级）
rowValidLen(b, i) = actS2Size(b)                                          # mask_mode=0（batch 级，各行相同）
lastBlk(b, i)     = (rowValidLen(b, i) - 1) / block_size                 # 整除
blkScore(b, i, lastBlk(b, i)) = +inf                                     # pin：每行 pin 自己的最后一块

mode=1 (source): candidateTopkIndex(b, i, :) = topK_blocks 个 blkScore 最大的块号
        （并列值 tie-break 不保证与 PyTorch 一致，见 §6.2；blkScore=-inf 的槽输出 -1）
        sparse_indices(b, i, :) = topK(score)                            # 与现有逻辑完全一致（不 mask）
mode=2 (consumer): score'(b, i, p) = score(b, i, p)      若 floor(p/8) ∈ candidateTopkIndex(b, i, :)
                                          -inf          否则
        # 等价视角：sparse_indices 是"从 candidate_topk_index 展开的 ≤16384 个候选位置中再挑 topk 个"
        sparse_indices(b, i, :) = topK(score')                          # 复用现有 sort/merge 管线
mode=3 (off):     现网行为，candidate 功能整体关闭（默认值，向后兼容）
```

dtype/精度约束：score 全程 fp32（现有路径不变）；块化 amax 与 -inf/+inf 处理均在 fp32 域（`NEG_INF=0xFF800000` 已有，新增 `POS_INF=0x7F800000`）。

### 1.5 验收标准

**功能验收**（mode=3 回归为正确性门禁，非性能）：

- candidate_mode ∈ {1, 2, 3}；mode=3 输出与现网 bit 级一致（回归，默认值保证既有调用零影响）。
- **layout_q：BSND（本轮已实现）+ TND（2026-09-07 需求追加，见 §11）**；layout_k = PA_BBND（BSND 时）或 TND（TND 时，cu_seqlens_k 变长拼接）。**key 0 轴非连续**（PA_BBND 第 0 维 stride > 块逻辑大小）为追加需求，参照 arch35 机制（§11.2）。
- mask_mode ∈ {0, 3}；**cmp_ratio ∈ {1, 2}**（用户指定重点场景，见 §6.6）。
- **q_head_num 仅测 32**（用户指定：所有 candidate 用例统一 g=32，不再覆盖 64；A1/A2 改造后 g=64 逻辑路径不变，由既有 910b 用例回归兜底）。
- **topk 动态值**：topk ∈ [1, 2048] 任意值，含 topk = min(index_topk, actS2Size) 语义（等价性验证，无代码改动）。
- S2 使 numBlocks < / = / > topk_blocks 三类；actS2Size 非 block_size 整除（尾块）。
- decode（S1=1）与 prefill（S1>1）。
- 910b 约束继承：quant_mode=2（INT8 + fp16 scale/weight）、无 return_value/output_idx_offset。

**精度验收**：

- `candidate_topk_index`：与参考实现的**选中块集合**一致（无序集合比较 + -1 槽位数一致）；允许 tie 场景按 §6.2 策略仲裁。
- mode=2 `sparse_indices`：与 "参考实现 mask 后走现有 golden topk" 一致（复用现有 compare 框架与 tie 容差策略）。
- mode=1 `sparse_indices`：与现网 golden 一致（不应受块级计算影响——同一 Vec1 内两套独立 sort buffer）。

> 本次交付不含性能验收（用户明确性能暂不关注），但设计保留向量化的块化/mask 实现路径（§3.3/§3.5），避免后续性能优化时返工。

**稳定性验收**：多 seed（≥3）、边界 shape（S2%8≠0、actS2Size=1、numBlocks=1、topk_blocks>numBlocks）、B>1 变长 batch。

---

## 2. 算法拆解（在现有 6 阶段上的增量）

| # | 阶段 | 现状 | 本次变更 | 执行单元 |
|---|---|---|---|---|
| 0 | 输入解析 | layout/actual seqlen/metadata 分核 | consumer(mode=2): 解析 `candidate_topk_index` GM 指针；新增 TilingData 字段下发 | Scalar |
| 1 | 预处理 | Vec0: w×qScale 广播 | 不变 | Vector |
| 2 | MatMul1 (QK) | AIC | 不变 | Cube |
| 3 | score 后处理 | Fixp DEQF16+ReLU → Ws g-sum → GM | 不变 | Cube |
| 4 | Vec1 score 生成 | mm1Res×kScale | **mode=2: 追加候选块 mask 生成 + Select 置 -inf**（插在 Mul(kScale) 之后、SortAll 之前） | Vector |
| 5 | 归约/块化 | —（无） | **mode=1: 块化 amax(8:1) + pin 尾块置 +inf**（与位置级 SortAll 并行，规模 1/8） | Vector |
| 6 | 选择/TopK + 输出 | SortAll+MergeSort → globalTopkUb_(2048) → 行末 ExtractIndex 直出 | **mode=1: 镜像维护 globalBlockTopkUb_(2048)，行末 ExtractIndex 直出 candidate_topk_index GM**；生命周期/重置点与位置级完全同步 | Vector |
| — | 无效清理 | CleanInvalidOutput 填 -1（sparse_indices） | **mode=1: 所有填 -1 的路径同步对 candidate_topk_index 填 -1**（DealActSeqLenIsZero / BSND 无效 S1——BSND 范围内共 2 处，见 §3.3 检查清单；TND padding 路径本轮不适用） | Vector |

阶段信息表（新增阶段细目）：

**S4a（mode=2, consumer mask）**：

- 输入：`candidate_topk_index` GM 行 `[topk_blocks]` int32（8KB）、当前 tile score UB `[tileLen]` fp32
- 输出：masked score UB（原地）
- 计算单元：Vector（compare-OR 分批，见 §3.5）
- 同步：无跨核（每行候选集独立，行内自洽）

**S5a（mode=1, 块化）**：

- 输入：score UB `[tileLen]`（kScale 相乘后）
- 输出：`blkScore UB [tileBlkNum=tileLen/8]` fp32 + `blkIdx UB` int32（基址 = tileS2Base/8）
- pin：当前行 pin 块 `lastBlk = (rowValidLen - 1) / block_size`（行级，`rowValidLen` 即 Vec1 已有的 `cuRealAcSeq`，mask_mode=0 时各行相同退化为 batch 级）；tile 包含该块时置 +inf；tile 内超出行有效长度（mask_mode=3）的块值为 -inf（pad 语义）
- 计算单元：Vector（8:1 strided ReduceMax + 尾块 -inf 对齐）

**S6a（mode=1, 块级 sort/merge + 直出）**：镜像 S6 结构，`SortAll(blkNumAligned)` + `MergeSort(globalBlockTopkUb_[row], topk_blocks, ...)` + 行末 `ExtractIndex` → `candidate_topk_index GM`；`AlignS2` 复用（块数对齐到 32/128/512 粒度）。

---

## 3. Host Tiling 设计（arch22）

### 3.1 维度建模

| 维度 | 含义 | 现值/来源 |
|---|---|---|
| B / N2 / G | batch / kv头(=1) / query头组 | **G = 32（本轮唯一测试规格）** |
| S1 | query 长度 | 不变 |
| S2 | 压缩后 key 长度（actS2Size） | 不变 |
| **numBlocks** | `ceil(actS2Size / block_size)` | **新增，随 batch 变长** |
| **tileBlkNum** | 单 tile 内块数 = `s2BaseSize / block_size = 256` | **新增，常量（block_size=8 时）** |
| topk | 位置级 topk | 动态 [1, 2048]，含 min(index_topk, actS2Size) 语义（无代码改动） |
| **topkBlocks** | 块级 topk | **固定 2048**（BASE_TOPK 结构复用；属性保留，host 校验限定，未来扩展只放宽校验） |

> **q_head_num=32 的支持方式（参照 v1 `quant_lightning_indexer` 实现，全量关键词搜索适配点）**：
>
> - **v1 与 v2 的关键差异**：v1 arch22 kernel 固定 `S1_BASE_SIZE=4`、推导 `mBaseSize = s1BaseSize × gSize`（g=32 → mBase=128）；v2 现状反之——固定 `M_BASE_SIZE=256`、推导 `s1BaseSize = 256/gSize`（g=32 → s1BaseSize=8，按行 UB 翻倍）。
> - **一致性论证**：v2 的 metadata aicpu 本就是 v1 风格（`s1BaseSize_=4` 默认、`mBaseSize_ = s1BaseSize_ × groupSize_`，aicpu.h:307 / aicpu.cpp:247）——g=64 时 4×64=256 与 v2 kernel 固定值恰好重合，**g=32 时若 kernel 不改，分核区间（128 基准）与 kernel 循环（256 基准）失配**。因此本修改同时是 g=32 正确性的必要条件。
> - **收益**：s1BaseSize 恒为 4 → sortOutBuf_/candidate globalBlockTopkUb_ 等 UB 与 workspace 全部不变，§3.3 的 g=32 UB 紧张场景（原 R3）不存在。
> - **验证范围（2026-09-07 用户确认）**：candidate 测试统一 **q_head_num=32**（g=64 由既有回归兜底，不在本轮范围）；R3 风险随之收窄为"推导变更正确性"而非 UB 容量。
>
> **适配点全量清单（v1↔v2 归一化 diff + 关键词扫描结论）**：
>
> | # | 位置 | v1 实现 | v2 现状 | 本次动作 |
> |---|---|---|---|---|
> | A1 | kernel `InitTilingData`（v2 kernel_arch22.h:191-193） | `s1BaseSize=4` 固定；`mBaseSize = s1BaseSize × gSize`（v1:171-175） | `mBaseSize=256` 固定；`s1BaseSize = 256/gSize` | **改**：对齐 v1 推导 |
> | A2 | host `GetGSize` 910b 分支（v2 tiling.cpp:857-860） | `gSize > 64` 才拒绝（v1:700-705，即 32 合法） | `gSize != 64` 拒绝 | **改**：放宽允许 32（复用 v2 tiling.h:87 既有常量 `G_SIZE_LIMIT_32_950`） |
> | A3 | cube L0/L1 切分（`ComputeMm1`） | 按 S1 维切：`s1L0LoopCnt = CeilDiv(actM/g, s1Base/2)`，L0 子块 `gSize×s1Base/2`（随 g 缩放，g=32 → 64 行/次） | 固定粒度：`CeilDiv(actM, S1G_BASIC_BLOCK_L0=128)`，L0 子块恒 128 行 | **不改**：g=32 时 actMBaseSize=128 → 单次 L0 循环，mExtension=CeilAlign(128,16)=128 ≤ L1 容量，v2 现有逻辑自动适配（两种切分等价可行，保持 v2 风格改动最小） |
> | A4 | cube `ComputeWs/LoadSToL0b/LoadWeightToL0a/FixpResToGm` | 全部 `gSize` 参数化（`k=gSize`、`s1gOffset += gSize`、`repeatTimes=CeilDiv(gSize,16)`） | 同样已 gSize 参数化 | **不改**（diff 确认一致） |
> | A5 | Vec0 `cuProcEleNum`（v2 service_vector:277） | `CeilAlign(cuS1ProcNum × gSize, 32)`（v1:289，UB 对齐防御） | 无对齐（g=64 时恒为 32 倍数而省略） | **建议同步 v1 的 CeilAlign**：g=32 时 4×32/3×32/2×32/1×32=128/96/64/32 恰好均为 32 倍数（数学上不改动也安全），但对齐写法防御未来 g 非 2 的幂 |
> | A6 | host workspace（v2 DoTiling arch22 分支） | `QliCalcWorkspaceSize` 用保守上界常量（mBase=512×s2Base=512，g 无关） | `M_BASE_SIZE(256) × S2_BASE_SIZE(2048)` 上界 | **不改**：256 是 g∈{32,64} 的 mBaseSize 上界（4×64），g=32 的 128 被覆盖 |
> | A7 | kernel Init workspace 布局（v2 kernel:463-478） | — | `mm1Res: 2×s1BaseSize×s2BaseSize`（s1BaseSize 恒 4 不变）；`weightMemSize: 16×mBaseSize×2`（mBaseSize=128 → 减半，变小安全） | **不改**：布局全部由 constInfo 推导，A1 改后自适应 |
> | A8 | metadata aicpu | — | 已是 v1 风格（`mBaseSize_ = 4×groupSize_`） | **不改**（g=32 与改后 kernel 基准一致） |
> | A9 | `GetS2BaseBlockNumOnMask/GetTotalBaseBlockNum/CalcGS1LoopParams` 等 | `s1BaseSize/gSize/mBaseSize` 参数化 | 同 | **不改**（A1 后自动正确） |
> | A10 | UT（tests/ut/op_host/arch22） | v1 有 arch22 tiling 单测 | v2 有同款 | **补**：g=32 的 tiling/infershape 单测 case |

### 3.2 多核切分 —— 不变性论证

- 910b 上 metadata 仅做 batch 级（`AssignByBatch`）与整行级（`AssignByRow`）切分，`AssignByBlock`（S2 块级）因 `supportFd_=false` 跳过 → 每个 (b, s1) 行整体落在一个核上，topk 天然单核完成，无需跨核归约。
- 块级 topk 与位置级 topk 在**同一 Vec1 调用**内、同一 (bN2, gS1, s2) 任务块上执行，共享现有 metadata 分核（QLI_V2_* AIC 段）。
- 负载增量：Vec1 每 tile 增加 `O(tileLen/8)` 的块化 + 块级 sort/merge（≈位置级 sort 开销的 1/8）→ 仅放大每块耗时，**不改变任务块数量与划分粒度** → metadata 的分核算法、输出布局、协议（1024 int32, AIC 36×8 + AIV 72×8）均不变，QLD_V2 段在 910b 本就全 0。

### 3.3 UB Buffer 规划（AIV，910b UB **实测 192KB**）

采用 §3.1 的 v1 对齐方案后，s1BaseSize 恒为 4（g∈{32,64} 通用），UB 不随 gSize 变化：

现有（`InitBuffers`）：inQueue 32KB + outQueue 8KB + indexBuf 8KB + tmpBuf 64KB + sortOutBuf 32KB = 144KB。

**mode=1 增量**：

| buffer | 元素数 | 字节 | 生命周期 | 复用 |
|---|---|---|---|---|
| globalBlockTopkUb_ | CeilDiv(s1BaseSize,2) × topkBlocks × 2 (fp32 pair) | 32KB | 整个 gS1 基本块 | 新增 TBuf，重置点与 globalTopkUb_ 同步 |
| 块化临时（实装布局，均在 tmpBuf 64KB 内） | blkScore@tmp[6144] / blkIdx@tmp[6144+blockNumPad] / isPad+pin 链@tmp[14336–15872] / blkSortTmp@tmp[11776] | ≤5KB | 单 tile | 复用 tmpBuf；**blkSortTmp 需容纳 mrgDst+mrgSrc ≤4608 floats，11776+4608=16384 恰为 64KB 末尾（曾置 12288 越界 2KB 触发 aicore）** |

合计 176KB < 192KB ✓（**实测 UB 为 192KB，非 256KB**；编译期从 PlatformInfo ubSize 校验，**禁止**硬编码常量）。

**mode=2 增量**（实装）：

| buffer | 元素数 | 字节 | 说明 |
|---|---|---|---|
| candBuf_（TBuf） | CeilDiv(s1BaseSize,2) 行 × topkBlocks × 2 (fp32 pair) | 32KB（R6 按行分区） | 排序后的候选 [values|idx] 对，每行独立（同核行间 tile0 重排序覆盖是 R6 根因） |
| candConstBuf_（TBuf） | topkBlocks (fp32) | 8KB | -1e30 常量（候选外罚分） |
| 掩码临时（复用 tmpBuf） | candInt@12288 / candSortTmp@4096 / blkIdxF·acc·diff@4352–4864 / posDist@12288 / isOutI32@14336 | ~20KB | **isOutI32 必须避开 [4096,6144)**（该区被 ProcessVec1 的 pen/idxPen 复用，曾重叠致掩码失效） |

**输出清理检查清单（mode=1 填 -1 的路径，BSND 范围内共 2 条）**：

1. `DealActSeqLenIsZero`（actS1Size=0 或 actS2Size=0，BSND 分支）
2. BSND 无效 S1 尾部（qSeqSize > actS1Size）

（causal 下 actS1Size > actS2SizeOrig 的行在 mask_mode=3 时按行有效长度自然处理为块 -inf，非独立清理路径；TND padding 路径本轮不适用，见 §5。）

### 3.4 Workspace 规划（GM）—— 无增量

现有布局（arch22，per AIC 核）不变：

```
[0]                    mm1ResGm      : 2 × s1BaseSize × s2BaseSize × 4B
[+off1]                vec0OutGm     : 16 × mBaseSize × 2 × 2B
[+off2]                vec1ResGm(LD) : s1BaseSize × 2 × 2 × BASE_TOPK × 4B   ← V1_DECODE 区（910b 不激活，保留）
```

**结论：candidate 功能不新增任何 workspace。** 理由：块级 topk 在单核 Vec1 内完成并直出（§3.2），无跨核中间结果落盘；mode=2 的候选集每 tile 从 GM 直接读入 UB。tiling.cpp 的 workspaceSize 计算不变（仍需随本方案回归确认无隐性依赖）。

### 3.5 分支策略（host 集中，kernel 单点判断）

| 分支 | 条件 | 行为 |
|---|---|---|
| mode=3 | candidate_mode=3 | 现有路径原样（默认值，向后兼容；模板内 if 包裹新增段） |
| mode=1 | candidate_mode=1 | S5a/S6a 全开（source） |
| mode=2 | candidate_mode=2 | S4a mask 生效（consumer） |

mode=2 的 membership 判定（候选列表 → tile 内 256 块 bool，向量化无标量循环）：

- `blkMask[i] = OR_j (candList[j] == tileBlkBase + i)`，compare-OR 分批：每批 32 候选 × 256 块（32KB fp32 view），共 topkBlocks/32 = 64 批/行；随后 `repeat_interleave(8)` 展开 + `Select` 置 -inf。

分支条件全部由 host tiling 写入 TilingData 字段（含 `candidateMode`、`candidateTopkBlocks`、`candidateBlockSize`），kernel 内不重复推导。

> 注：`numBlocks ≤ topkBlocks` 时 mask 恒为全选（等价于不 mask），实现上**不做专门 fast-path 分支**（性能暂不关注，减少分支与测试组合）；该等价性仍作为正确性用例覆盖（§6.3）。

---

## 4. 契约设计

### 4.1 TilingData（QLIV2TilingData 尾部追加，4B 对齐）

```cpp
TILING_DATA_FIELD_DEF(uint32_t, candidateMode)        // 1=source / 2=consumer / 3=off(默认)
TILING_DATA_FIELD_DEF(uint32_t, candidateTopkBlocks)  // ≤2048
TILING_DATA_FIELD_DEF(uint32_t, candidateBlockSize)   // 默认8，2的幂
```

host 写入 ↔ kernel 消费对照表随本方案落入 docs（维护接口式追踪）。字段分组注释：`// ---- candidate (two-level topk) ----`。

### 4.2 TilingKey —— 不变

理由（遵循"只编码影响模板实例的维度"）：candidate 路径是纯 Vector 数据通路，不改变 MatMul 形状/dtype/layout/模板实例，仅 kernel 内运行时分支；编码进 key 会造成 ×3 模板实例化与编译时间膨胀，无收益。

### 4.3 算子原型 / Infershape / aclnn / torch

- `quant_lightning_indexer_v2_def.cpp`：新增 optional input `candidate_topk_index`(INT32, ND)、optional output `candidate_topk_index`(INT32, ND)、三个可选属性；仅 `ascend910b` 配置声明（`aicore_config`），950 配置不动。
- `quant_lightning_indexer_v2_infershape.cpp`：`candidate_mode=1` 时输出 shape = sparse_indices 前缀 + 末维 `candidate_topk_blocks`（**仅 BSND 分支**，q 只考虑 BSND）；否则末维 0。InferDataType 补 SetOutputDataType(2, DT_INT32)。
- aclnn 两段式接口（`aclnnQuantLightningIndexerV2`）：GetWorkspaceSize/执行签名追加 optional `candidateBlocks` 输入/输出指针与 3 个属性，向后兼容（指针可空）。
- torch schema（**已定方案 b——新增 overload，旧接口不动**）：
  ```python
  # 新增变体（source/consumer 统一入口，返回固定三元组）
  quant_lightning_indexer.candidate(
      query, key, weights, q_descale, k_descale, topk, quant_mode, *,
      Tensor? candidate_topk_index=None,        # mode=2 时必传
      int candidate_mode=3,                 # 1=source / 2=consumer / 3=off(默认)
      int candidate_topk_blocks=2048,
      int candidate_block_size=8,
      ...其余参数同旧接口...) -> (Tensor sparse_indices, Tensor sparse_values, Tensor candidate_topk_index)
  # 非 source 模式第三元返回 shape (0,) 占位，元信息（block_size 等）随返回对象附带的 sidecar 属性传递
  ```
    - 旧 `quant_lightning_indexer` schema 二元组完全不变，现网调用零影响；
    - 模型侧 source/consumer 层统一走 `.candidate` 入口，按 candidate_mode 区分行为；
    - block_size 一致性断言（R4）在 python 封装层实现：shared 对象携带 `candidate_block_size` 元信息，consumer 调用时校验。

### 4.4 metadata 算子（`quant_lightning_indexer_v2_metadata`）

**结论：不改**。分核算法/输出协议/属性均不变（论证见 §3.2；910b 无 FD/LD 路径，QLD_V2 段维持全 0）。配套约束：

- 主算子 mode≠0 时 metadata 照常传入；
- 文档（README）补充：candidate 相关属性不参与分核，因负载模型未变。

---

## 5. 与既有机制的交互确认

### 5.0 mask 计算结论：不修改、不新增 mask mode

现有 mask_mode 与模型代码分支**逐行数学等价**，无需任何 mask 侧改动。

**关键澄清**：模型 if/else 是**推理时间步**的分支，不是一次调用内的分支——每次算子调用仍只有一种 mask_mode：

```
prefill 首块 (start_pos==0):  调用 QLI 1 次 (S1=seqlen)  → mask_mode=3, 行级 pin
decode step  (start_pos>0):   每步调用 QLI 1 次 (S1=1)   → mask_mode=0, batch 级 pin
两级 TopK 时间线:
  prefill 首块: QLI(mode=1, mask_mode=3) → 产出 candidates 存 shared_attn
  decode step:  QLI(mode=1/2, mask_mode=0) → source 更新候选 / consumer 消费候选
```

| 模型分支 | 模型行为 | op 对应 | 等价性论证 |
|---|---|---|---|
| `start_pos == 0`（prefill 首块） | `compress_lens[i] = (i+1)//ratio`（`[S1,1]` 行级），`p ≥ compress_lens[i]` 置 -inf（模型侧 masked_fill_；score 下沉算子后由 mask_mode=3 承担） | mask_mode=3 | op 行级有效长度 `(actS2SizeOrig - actS1Size + i + 1)/cmpRatio` 在首块时 `actS2SizeOrig = actS1Size = seqlen` → 退化为 `(i+1)//ratio`，与模型逐行恒等（golden `create_mask`、kernel `cuRealAcSeq`、模型三方公式一致） |
| `else`（decode） | 标量 `compress_lens = end_pos//ratio`，不 mask | mask_mode=0 | op 不 mask，天然匹配；pin 用 batch 级 `(actS2Size-1)/block_size` |

新增的行级 pin（S5a）**复用同一个 `cuRealAcSeq`**，不构成新 mask——pin 与 mask 同源，不会出现 pin 块落在被 mask 区域的矛盾。

**开放问题 O3**：`else` 分支为标量 compress_lens，意味着 mid-chunk prefill（start_pos≠0 且 S1>1）时模型**不做行内因果 mask**。若该调用形态实际不存在，测试矩阵中 "mid-chunk prefill×mode0" 用例删除；若存在，op 照实不 mask（复现模型行为），同样无需改动。待用户确认。

> ⚠ **O3 状态（标红保留）**：**此问题未关闭，不允许随本需求悄悄消化。** 当前决策为"暂不测试"——mid-chunk prefill×mode0 用例从本轮测试范围剔除（用例定义保留在矩阵，注释标明暂不执行），但设计上模型语义为"prefill 不做行内 mask"，**与常规认知相反**。在模型侧确认该调用形态（chunked prefill / 混布场景是否走 else 分支）之前：
>
> 1. 禁止有人"顺手"在算子里给 mode0+S1>1 补行级 mask（那是语义变更，不是 bug fix）；
> 2. 精度测试若碰到 mode0+S1>1 的数据，比对结论一律以"模型不 mask"的 golden 为准；
> 3. 本问题最终去向二选一：确认形态不存在 → 删除用例；确认存在 → 启用用例并补 golden 说明。

| 机制 | 交互 | 结论 |
|---|---|---|
| LD / ProcessDecode | 910b 不激活（`supportFd_` 仅 950 置 true，aicpu.cpp:329-338） | **不适用，方案不含 LD 路径**；kernel 现有 LD 代码与 V1_DECODE workspace 保留不动 |
| mask_mode=0（decode / mid-chunk prefill） | 无行 mask，全行有效长度 = actS2Size；pin 为 batch 级 `(actS2Size-1)/block_size`（对应模型 else 分支标量 `compress_lens = end_pos//ratio`） | 兼容 |
| mask_mode=3（prefill 首块） | 行级有效长度 `rowValidLen(i) = (actS2SizeOrig-actS1Size+i+1)/cmpRatio`（块化时行尾块按该行有效长度 pad -inf，等价模型 `masked_fill_`）；**pin 为行级** `(rowValidLen(i)-1)/block_size`（对应模型 `[S1,1]` 行级 compress_lens） | 兼容，S5a 内处理；行级 pin 与行级 mask 公式同源（Vec1 `cuRealAcSeq`），天然一致 |
| cmp_ratio 压缩 | 块划分基于压缩后 actS2Size（与 Python 的 S2=logits.size(-1) 一致） | 兼容 |
| Vec1 双缓冲预取（`LI_QUANT_PRELOAD_TASK_CACHE_SIZE=2`） | 块级 sort 结果随 runInfo 双缓冲流转，重置点与位置级 `globalTopkUb_` 完全同步（`info.s2Idx==0` 时重置、行末直出后重置） | 设计强约束：两套 buffer 的重置/输出点成对出现，UT 加不变量断言 |
| TND padding（sequsedQ < cuSeqlensQ） | padding 行的 candidate_topk_index 需填 -1（同 sparse_indices 无效值路径） | **不适用**（仅 TND 布局存在该路径，本轮 q 只考虑 BSND）；若未来放开 TND 需同步补上 |
| Sort32/MrgSort 降序假设 | 假设：降序（top-k 语义自洽，-inf 沉底/初始填充行为验证） | 实现期以 API 文档+单测确认（验证点 V1） |

---

## 6. 验证计划

### 6.1 Golden 与 provenance（强制）

- **参考实现独立重建**：golden 不得只搬 Python 片段——用 numpy 按 §1.4 数学定义逐行重建 `select_candidate_blocks`（pad→amax→pin→topk→scatter→expand），与用户 PyTorch 实现交叉验证后再作为 ground truth。
- **sidecar 强制字段**（`xxx.ref.json`，遵循全局 AGENTS 规则）：分支 `ds41` + commit `32d64c27f`、candidate 参数（mode/topk_blocks/block_size/mask_mode/cmp_ratio/layout）、随机种子、生成脚本路径、机器（910b 节点）、时间。
- **使用前核验**：比对结果与预期矛盾时，第一嫌疑人是 golden 本身（先独立复算，再怀疑 kernel）。

### 6.2 精度比对策略（2026-09-07 起：对齐官方 result_compare_method 规则）

`sparse_indices` 与 `candidate_topk_index` 统一采用与 `tests/pytest/result_compare_method.py::check_result` 相同的两级规则（harness `cmp_indices`，行粒度）：

1. **多重集合门**：整行排序后完全相等（值、`-1`、重复均敏感，**顺序不敏感**——MrgSort 的 tie-break 与 PyTorch 稳定排序不保证一致，实测 r2_m1_prefill_m3 存在精确平分对（score 位型相同）的顺序交换，属合法差异）→ 该行 PASS；
2. **边界容忍回退**（门未过时）：直接复用官方 `compare_topk_valid`——gold 有效前缀集合比较，差异元素按**边界值**（gold 前缀最后一个元素的分数）相对误差 ≤ thres(0.001) 容忍；
3. **-1 槽数硬校验**（对官方规则的收紧，仅一处）：行级有效计数不一致 → 直接 FAIL。官方仅按 gold 的 valid_len 切片、会放过 npu 多填/少填 -1 槽的情况；candidate 的 -1 槽是硬契约（numBlocks < topk_blocks 时必须填 -1），故收紧。

自检（selftest_cmp.py）覆盖五分支：同集不同序 PASS / 边界 0.1% 容忍 PASS / 大差异 FAIL / 重复 FAIL / -1 计数不匹配 FAIL。
当前状态：15 用例全部行过多重集合门（boundary 0 触发），即输出达到"排序后完全一致"。

- 构造 tie 密集用例（常值 score）单独验证 tie 不影响**多重集合**正确性；
- 大 shape（16K/128K/1M）harness 为设备侧独立脚本，CPU 全量 golden 不可行，沿用抽样行集合比较 + 有效性/确定性检查（§10）。

### 6.3 测试矩阵（每格至少 1 用例，mode=3 抽样回归）

| 维度 | 取值 |
|---|---|
| candidate_mode | 0 / 1 / 2 |
| cmp_ratio | **1 / 2**（用户指定两类场景） |
| layout_q × layout_k | BSND×PA（已实现）+ **TND×TND**（§11，cu_seqlens_q/k 变长拼接，q=TND/k=PA 组合不支持——PA 依赖 block_table 与 TND 前缀语义冲突） |
| key 0 轴 stride | **紧凑（stride == 块大小，已实现）+ 非连续（stride > 块大小，§11，key 与 k_scale 均需）** |
| q_head_num | **仅 32**（用户指定；全部 candidate 用例统一 g=32，A1 推导下 mBase=128，重点验分核对齐） |
| topk（动态） | 2048 / **min(index_topk, actS2Size) 场景**（topk<2048，验证 -1 padding 与 min 等价性） |
| 阶段 × mask_mode（按模型分支联动） | decode(S1=1)×mode0；prefill 首块(S1>1)×mode3（行级 pin）；~~mid-chunk prefill×mode0~~（**暂不测试，问题保留见 R5/§5.0 O3，问题解决后补测**）；**mask0 × 大 shape 全遍历**（2026-09-08 补：{16K,128K,1M}×{BSND,TND}×{m1,m2}×{r1,r2}=24 用例；注意 host 契约 mask_mode=0 时 cmp_residual_k 必须不传，ratio=2 的 act_k=2×s2 无 residual 仍为非整除场景） |
| 大 shape × candidate_mode | m1 全遍历 12（§6.6）+ **m2 补齐**（16K/128K-TND/128K-r2/1M 共 5，含 R6 修复后转正的 big128k_m2） |
| 大 shape × key 非连续 | **{16K,128K,1M}×{m1,m2} 全 6 用例**（stride0=2×紧凑， 生产 pool 翻倍 [15872,128,1,128]） |
| numBlocks vs topkBlocks | <, =, > |
| S2 对齐 | actS2Size%8=0 / ≠0（含 actS2Size=1） |
| B | 1 / 2 / 4（变长 batch；2026-09-08 补 B=2 四件套 + B=4 三件套大 shape：m1/m2/mask0 × BSND/TND，变长 seqused_k，B=4 含非对齐尾 tile 98432） |
| topk_blocks | 2048 全量 + **pin 用例 64**（host 已放宽为 (0,2048] 内 64 的倍数；`numBlocks > topk_blocks` 时 pin 最新块必须入选，单/多 tile 各 1 用例） |
| 大规模全遍历（生产规格） | **q_seq {16K, 128K, 1M} × layout {BSND, TND} × cmp_ratio {1, 2} = 12 用例直接并入 pytest 矩阵**（2026-09-08 用户要求；q_seq=s2 全 prefill，mask_mode=3，mode=1，ratio2 带 cmp_residual_k=1）；key pool **[7936,128,1,128]**、block_table **[1,1055]**（置换非恒等映射；1M 场景 pool 默认 8192+4）。**CPU 全量 golden 不可行 → 抽样行官方规则比对 + 全行有效性 + pin（强制保留最新 token 所在块）检查**（§6.6）；独立大 shape 脚本降级为深检工具 |

重点组合：mode1+尾块不对齐+causal、**mode1+prefill 首块（验证行级 pin：不同行 pin 不同块）**、mode2+causal、`numBlocks ≤ topkBlocks` 时 mode=2 与 mode=3 的等价性（mask 全选）、**全部用例统一 g=32（mBase=128 分核对齐，A1 推导的本命规格）**、**topk=min(2048, actS2Size) 与 topk=2048+滤-1 的等价性**、mode1+多核（B>1 触发行间切分，验证各核独立直出正确）、cmp_ratio=2 下块划分与 pin（actS2Size 为压缩后长度，cmp_residual_k 参与原始长度还原）。

### 6.4 分阶段验证（中间结果可 dump）

开发期在 `candidate_topk_index` 输出之外保留 debug 开关：dump 块化得分（S5a 后）与块级 merge 中间结果（S6a 后），用于阶段边界定位（kernel 错 vs golden 错）。合入前移除或进 debug 分支。

### 6.5 测试脚本设计（基于 test_quant_lightning_indexer_v2_single.py 修改）

新增 `tests/pytest/test_quant_lightning_indexer_v2_candidate.py`，结构复制自 `test_quant_lightning_indexer_v2_single.py`（保留 SAVE_PT_DIR/RESULT_PATH 环境变量、run_mode eager/graph 分支、QliV2ResultWriter 落盘机制），做以下修改：

**(1) 参数扩展**：`param_names` 尾部追加

```python
("candidate_mode",)  # 1=source / 2=consumer / 3=off(默认)
("candidate_topk_blocks",)  # 默认 2048
("candidate_block_size",)  # 默认 8
```

`test_data` 元组同步扩展；`QliV2ResultWriter.case_name/row` 的列随之扩展（sidecar 记录 candidate 参数，满足 §6.1 provenance 要求）。

**(2) paramset 用例**（新文件内定义或扩展 `test_quant_lightning_indexer_v2_paramset.py`）：

- 基线模板沿用 910b int8 形态（参照 `quant_li_default_a3`：quant_mode=2、qk_dtype=int8、dequant=float16、**q_head_num=32**、layout_key=PA_BBND），去掉 910b 不支持的 return_value/output_idx_offset。
- **设备分支新增 `Ascend910B`**（当前服务器为 910B3；现有 paramset 仅有 `Ascend910_93`/`Ascend950` 分支，910B 会 NameError）。
- 用例集（cmp_ratio × mode 全组合 + 边界）：

| 用例名 | cmp_ratio | mode | 覆盖点 |
|---|---|---|---|
| cand_r1_mode3_decode（回归） | 1 | 3 | decode 基线（mask_mode=0，标量 compress_lens；现网行为 bit 级一致） |
| cand_r1_mode1_decode | 1 | 1 | decode source 基础（batch 级 pin） |
| cand_r1_mode1_prefill_m3 | 1 | 1 | **prefill 首块 + mask_mode=3：行级 pin（不同行 pin 不同块）**+ 行级 mask |
| cand_r1_mode1_prefill_m0 | 1 | 1 | ~~mid-chunk prefill + mask_mode=0~~ **暂不执行**（问题 O3 未决保留，见 §5.0/R5；用例定义保留在矩阵中，待确认后启用） |
| cand_r1_mode1_tail | 1 | 1 | actS2Size%8≠0 + causal + 变长 B |
| cand_r1_mode2_self | 1 | 2 | 自洽候选（见 (3)a） |
| cand_r1_mode2_rand | 1 | 2 | 随机候选子集（见 (3)b）+ numBlocks<topkBlocks |
| cand_r1_mode2_few | 1 | 2 | numBlocks>topkBlocks（S2>16K） |
| cand_r2_mode1_decode | 2 | 1 | cmp_ratio=2 + cmp_residual_k 传入 |
| cand_r2_mode1_prefill_m3 | 2 | 1 | cmp_ratio=2 prefill 首块（行级 pin 含 residual 还原） |
| cand_r2_mode2_self | 2 | 2 | cmp_ratio=2 消费 |
| cand_r1_mode1_g32 | 1 | 1 | **q_head_num=32（本轮统一规格；mBase=128，验证 v1 式推导下分核/循环/输出全链路）**。其余用例同规格执行，不再单列 64 回归 |
| cand_r1_mode1_topk_min | 1 | 1/2/3 | **topk=min(2048, actS2Size)<2048：验证 -1 padding 与 min 等价性（与 topk=2048+滤-1 对比）** |

**(3) mode=2 的 `candidate_topk_index` 输入生成（golden 侧）**：

- (a) **自洽候选**：对同一 score 调用参考 `select_candidate_blocks` 的输出作为输入（等价于"source 层与 consumer 层权重相同的退化情形"，可校验 mode=2 结果 ⊆ mode=0 结果且含 pin 块）；
- (b) **随机子集**：从 `[0, numBlocks)` 随机采样 `min(topk_blocks, numBlocks)` 个块（覆盖任意候选集，含 numBlocks<topkBlocks 时的全选等价性）；两者都以 int32 tensor（`-1` padding 到 topk_blocks）随测试数据下发。

**(4) golden 扩展**（`quant_lightning_indexer_v2_golden.py`）：

- 新增 `select_candidate_blocks_ref(score, compress_lens, topk_blocks, block_size)`：按 §1.4 数学定义实现（pad→amax→pin→topk→-1 槽），**同时提供 numpy 逐行实现与 torch 实现交叉验证**（§6.1 要求的独立重建）。`compress_lens` 支持标量（decode/mask_mode=0 → batch 级 pin）与 `[S1,1]` 行级（prefill 首块/mask_mode=3 → 行级 pin）两种形态，行级值按模型公式 `(actS2SizeOrig - actS1Size + i + 1) // cmpRatio` 生成。
- `GeneralizedQLIV2` 扩展：mode=1 时在 `cal_atten_per_batch_int8` 的 `reduce_sum`（kScale 相乘 + causal mask 之后、sort 之前，fp32 域）上计算候选块 golden 并按 out 布局（BSND）返回；mode=2 时在同一位置对 `reduce_sum` 做 `masked_fill` 后走原 sort/topk。
- int8 路径 score 保持 fp32（无 bf16 cast），与 kernel 块化计算域一致，保证块 golden 确定性。

**(5) compare 扩展**（`result_compare_method.py`）：

- 新增 `check_result_candidate`：`candidate_topk_index` 按行集合比较——`set(有效块号)` 相等 + `-1` 槽数一致即 PASS（tie 容差见 §6.2）；
- `sparse_indices` 比对复用现有 `check_result`；mode=2 的 sparse golden 由 (4) 的 masked 路径产生。

**(6) 执行**：沿用现有 `test_run.sh`/pytest.ini 机制，单独一条 pytest 选择器（`test_quant_lightning_indexer_v2_candidate.py`），不混入现网 CI 集。

---

### 6.6 大 shape 用例的抽样比对机制（harness，2026-09-07）

16K/128K/1M 用例并入 pytest 后，CPU 全量 golden（参考真值）不可行（1M 行 × 1M 位置 × 32 头 ≈ 数十 Tflop），采用：

1. **抽样行官方规则比对**：每 batch 固定行 + 等分行 + 固定 seed 随机行（1M 约 16 行、16K/128K 约 32 行），逐行计算 golden（逐行 matmul：q 行向量 [G,D] × k 段 [S2,D]，秒级/行）并按 §6.2 两级规则比对；抽样覆盖 vl（valid length，行级有效长度）极小（首行）/中间/极大（末行）；
2. **全行有效性检查**（numpy 向量化）：sparse_indices（稀疏索引输出）∈ [-1, S2)、candidate_topk_index（候选块索引输出）∈ [-1, numBlocks)、mode=2 输出第三元为空；
3. **pin（强制保留最新 token 所在块）检查**：numBlocks > candidate_topk_blocks 的抽样行必须含最新块；
4. **双跑确定性**：大 shape 用例二次运行抽样行逐元素一致；
5. **force_rows 确定性锚点（R10，2026-09-08）**：用例可声明 `force_rows` 行号列表并入抽样集——随机抽样可能漏掉特定 vl（行级有效长度）窗口行（R10 的 stale 窗口：尾 tile cuS2Len∈(64,96]，即 mask_mode=3 下行号 i 满足 (vl mod 2048)∈(64,96]，每 batch 仅约 32 行）；big128k_b2_varlen 锚 [64,73,95]×2 batch，big128k_b4_varlen 锚 12 行覆盖 4 batch。

### 7. 风险与开放问题

| # | 风险/问题 | 影响 | 缓解 |
|---|---|---|---|
| R1 | tie-break 不一致 | 误报精度问题 | §6.2 集合比较 |
| R2 | Sort32/MrgSort 降序假设不成立 | 整体语义反转 | 验证点 V1：实现前单测确认 API 排序方向 |
| R3 | g=32 改动触碰 `mBaseSize/s1BaseSize` 推导，影响分核与循环边界（kernel 与 metadata 基准必须一致） | g=32 分核错乱/越界 | 对齐 v1 推导（§3.1）+ `cand_r1_mode1_g32` 用例 + g=64 全量回归（推导变更影响既有路径，mode=0 也需回归） |
| R4 | source/consumer 的 block_size 不一致（跨算子约定） | 语义错乱 | 文档强约束 + torch 层断言（`quant_lightning_indexer.candidate` python 封装内校验，§4.3） |
| ~~R6（2026-09-07 实测，**2026-09-08 已解**）~~ | mode=2 S2≥128K prefill candBuf 被中途污染 | **根因：同核多行共享 candBuf**——每 AIV 处理 CeilDiv(s1BaseSize,2)=2 行，s2 内层循环按 gS1 块整体推进，后一行（row+2）的 tile0 重排序覆盖前一行候选，前一行 tile1..63 全部读错（三点快照 SNAP/MID/REF 定位：MID==SNAP 证明主路径无辜，漂移精确在 tile 切换）；s1=1（每核单行）不触发，故此前所有小 shape 用例漏检 | **修复：candBuf 按行分区**（innerS1Idx × candBlocks × 2 对索引，2 行 32KB，mode=2 UB 184KB≤192KB）；新增 r1_m2_prefill（s1=8 mode=2）作为 R6 小 shape 门禁 + big128k_m2 转正；47 用例全绿 |
| R7（§11） | TND 下 candidateOutOffset 与 cuS1Idx 双重前缀（两者均含 cu_seqlens_q 前缀则行号翻倍） | 偏移结构与主输出 indiceOutOffset 完全同构（前缀在 offset、行号 batch 内），理论无双重；tnd_m1_decode 用例显式验证 GM 行对位 |
| R8（§11） | keyStride0 改造影响现网紧凑场景（PA 现网假设 stride==块大小） | keyStride0==0 或 ==紧凑值时走原公式兜底，现网行为 bit 级不变；pa_gap_m3_regress 回归 |
| R9（§11.6） | **output_idx_offset 在 arch22 为死参数**（入口收指针未绑定未消费，host 校验完整但合法值被静默忽略）；调用方传非零偏移时 sparse_indices 不含偏移 → 上层绝对位置还原错误 | A15 对齐 arch35 使能；与 candidate 的契约：offset 仅作用于 sparse_indices，candidate_topk_index 保持相对块号（source 输出取加 offset 前），避免 mode=2 掩蔽整行错位 |
| ~~R10（2026-09-08 实测，**当日已解**）~~ | mode=1 大 shape（总块数>candidate_topk_blocks）vl（行级有效长度）尾 tile 块分数 stale：`brmRepeat = blkLen/64` 整除截断，blkLen=96（AlignS2 在 (64,128] 段唯一非 64 倍数输出）时只归约 [0,64)，块 8..11 残留上一 tile/行分数 | 实测 big128k_b2_varlen b1 行 73：块 14344 拿 stale 3.5568（真值 1.1753）虚高挤掉第 2048 名边界块 2760（3.0918）；**小 shape 总块数≤2048 时候选集合=全部块，stale 分数不改变集合故漏检**；B=1 大 shape 抽样未踩中窗口行（尾 tile cuS2Len∈(64,96] 即行号 i∈[64,95] 每 batch 仅 32 行） | **修复：brmRepeat 改 `CeilDiv(blkLen, 64)`**（多归约的 [96,128) stale 只落 pad 块槽位，被 -inf 位型链位精确覆盖，无害）；harness 新增 force_rows 确定性锚点（§6.6），B=2 锚 6 行、B=4 锚 12 行覆盖各 batch 窗口 |
| **R5（标红，O3 未决）** | **mid-chunk prefill（start_pos≠0 且 S1>1）形态下模型不做行内因果 mask，与常规认知相反**；该调用形态是否存在未确认 | 若误当作 bug "修复"（擅自加行级 mask）将引入语义变更；测试碰 mode0+S1>1 数据可能误报精度问题 | **暂不测试该用例，问题保留**（§5.0 O3 标红段）；三不准：不准顺手加 mask / 不准以"常规认知"为 golden / 处理前必须先与模型侧确认调用形态 |

已决策记录：

- ~~O1~~ **已定**：torch 接口采用方案 b——新增 overload `quant_lightning_indexer.candidate(...) -> (Tensor, Tensor, Tensor)`，旧 schema 二元组不动（§4.3）。
- ~~O2~~ **已定**：`candidate_topk_blocks` 当前仅支持 2048；属性保留、host 校验限定 2048，未来扩展只放宽校验不改接口；非默认值测试用例已删除。
    - **2026-09-07 更新**：已放宽为 (0, 2048] 内 64 的倍数（commit a346acdcb，pin 语义验证需要 topk_blocks=64）；BASE_TOPK=2048 上限不变。

## 8. 实施拆解（文件级）

| 文件 | 变更 |
|---|---|
| `op_host/quant_lightning_indexer_v2_tiling.h` | TilingData +3 字段；ParaInfo/常量（CANDIDATE_* 索引） |
| `op_host/quant_lightning_indexer_v2_tiling.cpp` | 属性解析与校验（mode 互斥、范围、consumer 必传输入）；**`GetGSize` 910b 分支放宽允许 gSize=32（适配点 A2）**；TilingData 写入（workspace 计算不变，A6） |
| `op_kernel/arch22/quant_lightning_indexer_v2_kernel_arch22.h` | **`InitTilingData` 改 v1 式推导（A1）：`s1BaseSize=4` 固定、`mBaseSize = s1BaseSize×gSize`**；删除/降级 `M_BASE_SIZE=256` 常量；Init 传参/新 GM 张量（candidateTopkIndexInGm/candidateTopkIndexOutGm）、Vec1 调用点注入 |
| `op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h` | **Vec0 `cuProcEleNum` 补 `CeilAlign(·, 32)`（A5，同步 v1:289）**；S4a/S5a/S6a 实现、globalBlockTopkUb_、CleanInvalidCandidateOutput（不动 ProcessLD） |
| `tests/ut/op_host/arch22/` | **补 g=32 tiling 单测（A10）** |
| `op_host/quant_lightning_indexer_v2_def.cpp` | 新 input/output/attrs（仅 ascend910b 配置） |
| `op_host/quant_lightning_indexer_v2_infershape.cpp` | candidate_topk_index 输出 shape/dtype |
| `op_kernel/quant_lightning_indexer_v2.cpp` | arch22 入参透传（arch35 分支不动） |
| `examples/` + `docs/aclnnQuantLightningIndexerV2.md` | aclnn 用例与接口文档 |
| `torch_extension/` | schema overload + python 封装 |
| `tests/pytest/test_quant_lightning_indexer_v2_candidate.py` | **新增**，基于 `test_quant_lightning_indexer_v2_single.py` 修改（§6.5） |
| `tests/pytest/quant_lightning_indexer_v2_golden.py` | `select_candidate_blocks_ref`（numpy+torch 双实现）、`GeneralizedQLIV2` 的 mode=1/2 golden 路径 |
| `tests/pytest/result_compare_method.py` | `check_result_candidate`（集合比较 + -1 槽核对） |
| `tests/pytest/qliv2_test_utils.py` | case_name/row 列扩展（candidate 参数入 sidecar） |
| `README.md`（两算子） | 参数表、约束（block_size 一致性、同 forward 有效期、910b 无 LD 说明） |
| **A11** arch22 cube/vector：`KeyNd2NzForPA`/`GetKeyScale` 改用 `keyStride0`/`keyDequantScaleStride0`（对齐 arch35，tiling 字段已存在未消费；0 或紧凑值兜底原公式） | key 0 轴非连续 |
| **A12** op_host tiling.cpp：删除 candidate 的 layout_q=BSND 限制，加 TND 输入校验（cu_seqlens_q 必传） | TND 放行 |
| **A13** arch22 kernel：验证 TND 下 candidateTopkIndexIn/Out GM offset 与 CleanInvalidOutput 的 TND 输出分支（已有 outputLayout==TND 分支） | TND candidate |
| **A14** torch_extension：py/csrc 的 TND 分支透传与输出 shape（ConstructOutputTensor 已有 TND 分支） | TND 封装 |
| **A15** arch22 kernel：使能 output_idx_offset（入口 SetGlobalBuffer + 每行标量读 + 输出前 int32 向量 Adds，参照 arch35 vector:680/856 与 IndicesAddOffset）；candidate_topk_index 不加 offset（相对块号契约） | offset 使能 |

**明确不做**：arch35 (950) 全部路径；LD/ProcessDecode 相关改动；metadata 算子代码；workspace 布局变更；TilingKey 变更。

## 9. 实现期实测平台约束（2026-09-07 调试结论，v220 / 910b）

实现与调试期间实证的平台铁律，均已在代码注释中标注，后续维护必须遵守：

| # | 约束 | 违反表现 | 正确做法 |
|---|---|---|---|
| P1 | 向量指令 count 必须 64 对齐 | `Duplicate(count=1)`（pin 单元素写）触发 aicore 异常 507015 | 少量元素写入用标量 `SetValue`（须配 P2）或并入 64 对齐向量链 |
| P2 | 标量 UB 写/读与 V 管道互不被 `PipeBarrier<PIPE_V>` fence | pad 块 -1 填充被 SortAll 抢跑覆盖；排序后 `GetValue` 二分读到 stale 数据（mode=2 IoU=0） | 标量读 V 写结果前 `SetFlag/WaitFlag<HardEvent::V_S>`（参照 CANN topk_v200 实现）；标量写后对称 S_V；或彻底纯向量化 |
| P3 | UB→UB `DataCopy`（走 MTE 管道）不被 `PipeBarrier<PIPE_V>` 等待 | 跨 tile MergeSort 回拷累加器后，下一 tile 的 MrgSort 读到 stale（tile1 块整段缺失 + -inf 位型+8k 垃圾×4） | 回拷改 int32 视图 `Adds(+0)` 分块（`MergeSortVecCopy`）；**float 域 Adds 会把 0xFFFFFFFF（-1 的负符号 NaN 位型）规范化为 0x7FFFFFFF**，必须 int32 |
| P4 | v220 Cast：s32→f32 `CAST_NONE` 合法（vconv_s322f32）；**f32→s32 `CAST_NONE` 为 assert no-op**（仅 RINT/FLOOR/CEIL/ROUND/TRUNC 有指令） | fp32 算术链构造的 idx 经 Cast 后保持浮点位型垃圾（score 位型泄漏进索引字段） | 索引构造全程 int32 域（`ArithProgression` + Sub/Maxs/Mins/Mul 链）；确需 f32→s32 时用 `CAST_RINT` |
| P5 | kernel `.o` 编译自 `build/binary/ascend910b/src/` 源码副本（cmake configure 时拷贝，make 不刷新），受 `gen/*.done` 门闩控制；且该路径下 `compile_stop.flag` 会阻塞失败后的重试 | 连续三轮"修复无效"实为旧二进制（AGENTS.md 教训复现，部署 .h 源码是新的、编译产物是旧的，标记检查被假象通过） | build_install.sh 流水：rsync 源码副本 → 删 `.done` 门闩 → make → **校验 .o mtime 为本次** → 安装 → `cmp` 部署 .o 与构建 .o 一致 |

## 10. 当前验证状态（2026-09-07）

- pytest **87 用例全绿**（2026-09-08 R10 修复 + B=2/B=4 变长补齐后：24 基础 + 12 m1 大遍历 + 5 m2 大补 + 6 pa_gap 大 + 24 mask0 大遍历 + 4 B=2 变长 + 3 B=4 变长 + 其余 TND/offset/门禁；总时长 ~32 分钟）：
    - 22 基础用例（mode 1/2/3 × cmp_ratio 1/2 × g32/64 × decode/prefill/tail/sparse2048/pin64 单/多 tile/B=4 变长/S1 尾块/tiny/cand64×mode2）全过，零 aicore
    - **大 shape 直接入矩阵全过**：big16k_m1 / big128k_m1 / big128k_m1_b2（B=2 变长）/ big1m_m1（131072 块 pin（强制保留最新 token 所在块）抽样验证）/ big1m_m3（回归）——§6.6 抽样行官方两级规则比对 + 全行有效性 + pin 检查，总时长 ~4 分钟
    - 等价回归全过：pa_gap_m3_regress（紧凑场景现网行为不变，R8）、off_zero_regress（零偏移与不传一致）
    - **A11/A12/A15 实施后（2026-09-08）：tnd_*×7、pa_gap_m1/m2/128k×3、off_m1_decode/m2/off_tnd×3 全部 XPASS 转正；pa_gap_m3_regress（紧凑等价）与 off_zero_regress（零偏移等价）回归保持通过
    - **R6 修复转正（2026-09-08）：candBuf 按行分区（同核 2 行共享是根因）**——big128k_m2 通过；新增 r1_m2_prefill（s1=8 mode=2）作为 R6 小 shape 门禁；大 shape 全遍历 12 用例（q_seq×layout×ratio）全过；xfail 清零
    - **R10 修复转正（2026-09-08）：blkLen=96 时 brmRepeat 整除截断致尾 tile 块分数 stale（§7 R10）**——B=2 变长四件套（big128k_b2_varlen/tnd_m2/mask0_r2 + big1m_b2_m1）+ B=4 变长三件套（[65536, 98432, 131072, 81920]，b1=98432 使尾 tile 窗口行移位覆盖非对齐 vl）全过；修复后全量 84 + B=4 新增 3 = 87 用例全绿
- 大规模生产规格（`/opt/tjj/qli_cand_test/test_big_shape_m1.py`，设备侧 harness）：16K(b=1) / 128K(b=2) mode=1、128K(b=2) mode=2 随机半数候选、1M(b=1) mode=1 全部 IoU=1.000000；**1M 场景 pin 块 131071 确认入选**（131072 块中选 top-2048）
- 大 shape 回归：1M mode=3（IoU=1.0 抽样行精确比对 + 双跑确定性）、1M cmp_ratio=2、qseq 256..2048 扫描全过
- 提交序列：`aede3e62e`（功能实现）→ `a346acdcb`（五处正确性修复：pad 块 int32 向量算术填充 / MergeSortVecCopy 位精确回拷 / pin 纯向量化（全局块号比较）/ mode=2 CountGE 窗口边界 + CAST_RINT + isOutI32 区域重叠 / V_S fence）

## 11. TND 与 key 0 轴非连续支持（2026-09-07 需求追加，待实施）

### 11.1 需求（2026-09-08 用户澄清后修正）

1. **layout_q = TND（仅 Q 侧）**：q 为变长拼接 `[T, G, D]` + cu_seqlens_q；**layout_k 固定 PA_BBND（K 仅支持分页布局，不支持 TND）**——q=TND 时 metadata 强制要求 layout_k=PA_BBND。candidate 三 mode 全支持；现网 kernel 已有 TND 模板分支，本轮打通 host 校验（GetS1Size 的 TND 分支补齐 s1Size=T）、candidate 的 GM offset 与输出布局。
2. **key 0 轴非连续**（PA_BBND）：key 与 k_scale 的第 0 维物理 stride 可大于块逻辑大小（block_table 指向的物理块之间存在间隙，如池化重组后的非紧凑存储）。

### 11.2 arch35 参照机制（已实现，直接移植）

- **key 主体**（cube `KeyNd2NzForPA`）：`blkTable.GetValue(bIdx*maxBlockNumPerBatch + s2BlkId) * constInfo_.keyStride0 + s2BlkOffset*headDim`（arch35/quant_lightning_indexer_v2_service_cube_arch35.h:437）；
- **k_scale**（vector `GetKeyScale`）：`blockId * constInfo_.keyDequantScaleStride0 + startBlockTableOffset`（arch35/..._service_vector_arch35.h:456）；
- **host 侧**：tiling.cpp 从 acl tensor `keyStridesVec_[0]` / `keyDequantScaleStridesVec_[0]` 取真实 stride 经 `set_keyStride0`/`set_keyDequantScaleStride0` 下发（tiling.h 字段已存在）；
- **arch22 现状差异**：`KeyNd2NzForPA` 硬编码 `blk * kCacheBlockSize * kHeadNum * headDim`（service_cube_arch22.h:317）、`GetKeyScale` 硬编码 `blockId * kCacheBlockSize_`（service_vector_arch22.h:157）——均假设紧凑存储；tiling 字段已下发但 **arch22 kernel 未消费**。
- **实施补丁（A11）**：kernel 两处寻址改用 `keyStride0`/`keyDequantScaleStride0`（0 时兜底原紧凑公式，现网行为不变）；host `CheckKeyContiguous` 的 0 轴放行从仅 arch35 扩展到全部 PA_BBND。**关键发现：aclnn 动态调用下 `GetDynamicInputStride` 恒为空**（仅 TensorV2 图模式携带非连续描述，exe_graph TensorV1 的 GetStride 亦为空）——新增可选属性 `key_stride0`/`key_dequant_scale_stride0`（def.cpp + tiling 下发，csrc 自动取 `key.stride(0)` 传入，调用方无感），GetDynamicInputStride 有值时优先。

### 11.3 适配点（A11–A14，见 §8 表）

**TND 偏移结构（A13 核心）**：`candidateOutOffset = cu_seqlens_q(bIdx) × kHeadNum × candBlocks + n2Idx × candBlocks`（batch 级前缀，CalcRunInfo 已含 TND 分支），行偏移 `cuS1Idx（batch 内行号）× candBlocks`——与主输出 `indiceOutOffset` 完全同构；输出布局 `[T, N2, K]`（csrc ConstructOutputTensor 已有分支），无效行清理走既有 `outputLayout == TND` 分支（kernel_arch22.h:418）。**实施补丁（A12/A13）**：host GetS1Size 补 TND 分支（s1Size = q.shape[0]，原仅 BSND 赋值导致 TND 下 s1Size=0、consumer shape 校验 expectSize=0 误报）；candidate host 校验放行 layout_q∈{BSND,TND} 且 TND 时强制 layout_k=PA_BBND + cu_seqlens_q 必传。

### 11.4 测试用例设计（§6.3 矩阵新增行）

**TND（q=k=TND，cu_seqlens 变长）**——pytest id：`tnd_m1_decode` / `tnd_m1_prefill` / `tnd_m2_rand` / `tnd_m3_regress` / `tnd_r2_m1` / `tnd_m1_pin64`：

| 用例 | B | seqs_q | seqs_k | ratio | mask | mode | 要点 |
|---|---|---|---|---|---|---|---|
| tnd_m1_decode | 2 | [1,1] | [1024,2048] | 1 | 3 | 1 | 变长 TND decode；**验证 GM 行对位（R7）** |
| tnd_m1_prefill | 2 | [8,4] | [1024,2048] | 1 | 3 | 1 | TND prefill 行级 pin；seqs_q 非对称 |
| tnd_m2_rand | 1 | [1] | [4096] | 1 | 0 | 2 | TND 消费（候选输入 GM offset 走 TND 前缀） |
| tnd_m3_regress | 2 | [4,4] | [1024,2048] | 1 | 3 | 3 | TND 现网回归 |
| tnd_r2_m1 | 1 | [1] | [1024] | 2+[1] | 3 | 1 | TND + cmp_ratio=2 + residual |
| tnd_m1_pin64 | 1 | [1] | [8192] | 1 | 0 | 1 | TND 多 tile + pin（candBlocks=64） |

**key 0 轴非连续（PA_BBND，key/k_scale stride（步长）翻倍，块间间隙）**——pytest id：`pa_gap_m1_decode` / `pa_gap_m2` / `pa_gap_m3_regress` / `pa_gap_128k`：

| 用例 | 要点 |
|---|---|
| pa_gap_m1_decode | `as_strided` 构造 stride=2×紧凑 的 key/k_scale，block_table 只指向偶数物理块；mode=1 |
| pa_gap_m2 | 同上 mode=2（GetKeyScale 的 stride 路径同时覆盖） |
| pa_gap_m3_regress | **紧凑场景回归（R8）**：现网行为 bit 级不变的等价性验证 |
| pa_gap_128k | 大 shape 128K + 非连续 + 官方两级规则抽样比对 |

**大 shape 交叠**（并入 pytest，§6.6 抽样机制）：`tnd_big16k_m1` / `tnd_big128k_m1`（TND 无 block_table，用 cu_seqlens 拼接总池）；`pa_gap_128k`（非连续 × 128K）；`off_tnd_m1`。

**xfail（预期失败）标注**：A11/A12/A15 实施前，tnd_*（host 拒绝 TND）、pa_gap_m1/m2/128k（kernel 紧凑寻址读到 gap 段零值）、off_m1_decode/off_tnd（offset 被静默忽略）以 `pytest.mark.xfail(strict=True)` 入矩阵——实施通过后 XPASS 自动报警提醒摘标记；`pa_gap_m3_regress`（紧凑等价回归）与 `off_zero_regress`（零偏移等价回归）当前实现即应通过。

### 11.6 output_idx_offset 使能（A15，2026-09-07 追加调查）

**现状（A15 已实施，2026-09-08）**：原为死参数（入口收指针未绑定未消费）。现已使能：入口 SetGlobalBuffer + 经 InitVecCandidateTensor 传 tensor 与有效标志（注意 **InitParams 先值拷贝 constInfo**，之后再改 constInfo 标志不生效——标志必须走传参）；主输出路径（ProcessVec1 的 needCopyOutGm 分支）与 LD（低延迟归约）路径（ProcessLD 搬出前）各加一处消费：GM 标量读行偏移（读 offset 无 V→S 竞态）+ int32 向量 Adds（64 分块、零偏移零开销、-1 槽位精确不变）。**注**：910b 上 LD 未启用（metadata AICPU 内核 supportFd_ 仅 ASCEND950 置位，fdUsedVecNum=0，isLdCoreEnable 恒 false，ProcessDecode 的 ProcessLD 为死路）——LD 路径消费点为将来 LD 启用的对称实现。

**语义（arch35 已实现，直接移植）**：每行一个 int32 偏移，输出拷 GM 前对 sparse_indices 逐元素 `+= offset`（`IndicesAddOffset`：int32 向量 Adds，64 对齐）；用于 TND 多请求聚合（各请求 KV cache 起始不同，kernel 内相对 → 输出绝对）。消费点：arch35 vector:681（`outputIdxOffsetGm.GetValue(outputIdxCoreOffset + rowIdx*kHeadNum)`，GM 标量读无 V_S 竞态）与 :856（非零才加，零偏移路径零开销）；`outputIdxCoreOffset` 与 indiceOutCoreOffset 同构（TND 前缀）。

**与 candidate 的契约（设计决策）**：offset 仅作用于 sparse_indices；`candidate_topk_index`（source 输出/consumer 输入）一律为**加 offset 前的 batch 内相对块号**——否则 mode=2 掩蔽需同步减偏移，且跨层共享（shared_attn.candidates）时双方 offset 可能不同会导致掩蔽错位。跨层传递绝对块号的换算由 torch 封装层提供工具函数（与 §1.3 "expand 工具"同理）。

**测试用例**：

| 用例 | 要点 |
|---|---|
| off_m1_decode | BSND + output_idx_offset=[1000]，sparse_indices 每元素 +1000（官方比对：golden 行 + offset），candidate 输出不受影响（仍相对块号） |
| off_tnd_m1 | TND + 每 batch 偏移（cu_seqlens_k 前缀作为 offset），输出绝对 KV 位置 |
| off_zero_regress | offset=0 行为与不传 bit 级一致（零偏移路径零改动） |
| off_m2 | mode=2 + offset：掩蔽用相对候选，输出 sparse_indices 仍加 offset（consumer 场景契约） |

### 11.5 实施顺序（2026-09-08 全部完成）

1. ~~A11~~ ✅ kernel 寻址 + host 放行 + stride attr 通路（GetDynamicInputStride aclnn 动态调用恒空）；
2. ~~A12/A13/A14~~ ✅ TND（仅 Q 侧）：host 放行 + GetS1Size TND 分支 + GM offset 同构验证；
3. ~~A15~~ ✅ output_idx_offset 使能（InitParams 值拷贝陷阱经 InitVecCandidateTensor 传标志）；
4. ✅ 大 shape 交叠：pa_gap_128k / tnd_big128k / off_tnd 全过。
5. ✅ 大 shape 全遍历（2026-09-08）：12 用例（q_seq {16K,128K,1M} × layout {BSND,TND} × ratio {1,2}）全过（16K/128K 批 65s、1M 批 7m15s）。**修复 harness pin 检查公式**：mask_mode=3 下行级有效长度必须除 cmp_ratio（`(act_k−S1+i+1)//ratio`，与 kernel/golden 一致；原式漏除在小 shape 因 clamp s2 掩盖碰巧通过，大 shape 行 0 暴露——错误的参考检查比没有检查更危险）。

**部署要点（踩坑记录）**：attr 变更后编译产物 hash 改变（18e4fb→d566），但 `bin/quant_lightning_indexer_v2.json`（bin 选择配置）残留旧映射——**必须删除该 json 与 binary_info_config.json 强制重生成**，否则运行时按旧映射找不到新 .o（stat file failed），或加载旧产物（新改动静默不生效）。build_install.sh 已加 autogen 头清理，需同步加 bin config 清理。

## 12. 新接口规范 py 适配层（2026-09-09）

面向新芯片接口规范（mxfp4/uint8/e8m0 descale、candidate_block_indices/candidate_block_length 命名）的 **py 分发封装**：按规范签名收参，内部映射到已注册的 `quant_lightning_indexer_candidate`（分别固定 candidate_mode=1/2），**不改 csrc / 算子校验 / 数据类型**；调用侧按芯片选择接口（本后端 910b 走本适配层，新芯片走其自带实现）。

**新增入口**（`torch_extension/quant_lightning_indexer.py` 末尾；pip 包 `cann_ops_transformer.ops` 与源仓库 `torch_extension/__init__.py` 同步导出）：

- `quant_lightning_indexer_candidate_source(...)` — 规范接口1（source）：返回 4 元组 `(sparse_indices (T1,N2,k), sparse_values, candidate_block_indices (T1,N2,cb), candidate_block_length (T1,N2))`；
- `quant_lightning_indexer_candidate_consumer(...)` — 规范接口2（consumer）：输入 `candidate_block_indices (T1,N2,cb)` + `candidate_block_length`，返回 2 元组 `(sparse_indices, sparse_values)`（布局随内部：B=1 BSND 4 维 / TND 3 维）。

**映射约定（规范项 → 内部接口）**：

| 规范项 | 映射 |
|---|---|
| `q_descale` / `k_descale` | `query_dequant_scale` / `key_dequant_scale`（仅改名） |
| `candidate_block_indices` | `candidate_topk_index`（仅改名；块级相对块号；BSND 路径自动补/去 batch 维） |
| `seqused_q`（规范注释为每 batch **key** 截断） | `seqused_k`（歧义点①：若实为截断 query 行数则改一行） |
| `candidate_block_length` | mode=1 输出：py 公式 `vl(i)=clamp((act_k−S1+i+1)//ratio, 0, K)`（act_k=K×ratio+residual，与 golden 同式）；mode=2 输入：**忽略**（op 内部已按 mask 规则与 K 取小截断，语义冗余） |
| `candidate_topk_blocks=-1`（无机制默认） | source 场景取 2048；consumer 由 `candidate_block_indices.shape[-1]` 推导 |
| layout 参数消失 | q 恒 3 维 (T1,N1,D)：有 `cu_seqlens_q`=TND 直传；无=B=1 BSND（q/w/q_descale 补 batch 维，输出端去掉）；**B>1 且无 cu_seqlens_q 显式报错**（batch 维丢失防御） |
| `metadata` 缺省 | py 自动调 metadata 算子生成（shape 推导；含一次 `.item()` 主机同步，调用方可预生成传入绕过） |
| `output_idx_offset`（仅接口2） | 透传（A15 既有支持，仅作用于 sparse_indices） |

**本后端限制（py 层显式报错，非 op 校验改动）**：`block_table` 必传（key 仅支持 PA_BBND 分页布局）；`return_value=True` 不支持（内部入口硬编码 False，不改 csrc）。

**验证**：pytest `test_qli_newapi*` 4 用例（BSND B=1 prefill mask3+ratio2+residual / BSND B=1 decode mask0 / TND B=2 变长 mask3 / 错误路径×5）——source/consumer 与旧入口 bit 级一致 + metadata 自动/手传 bit 级一致 + block_length 公式对照 + B>1 无 cu_seqlens_q 防御。既有 87 用例零影响（旧入口抽查回归通过）。

**全量矩阵经新接口重跑（2026-09-09，`qli_cand_test/test_qli_newapi_full.py`）**：87 用例中 **76 全过 + 11 规范限制跳过 + 0 失败**（31m54s，机制=monkeypatch `npu_run` → 适配层，主比较逻辑/官方两级规则/pin/全行有效性原样复用）。跳过项两类均为规范表达能力边界而非适配层缺陷：① B>1 且 BSND ×9（新接口 q 恒 3 维，多 batch 必须 cu_seqlens_q/TND——含 big128k_b2_varlen/big1m_b2_m1 等）；② mode=1 + 非零 output_idx_offset ×2（off_m1_decode/off_tnd_m1，规范 source 无该参数，仅 consumer 有）。覆盖确认：pa_gap 非连续（大 shape ×6）、pin64、g64、tiny(s2=1)、sparse2048、mask0 大 shape 全遍历、TND B=2/B=4 变长、off_m2（consumer+offset 透传）、off_zero_regress（source 零偏移等价回归）均过。

**规范侧待澄清（不阻塞本适配层）**：seqused_q 注释 keys 与命名矛盾；`ori_sparse_indices` 未在输入列表；k_descale 末维 `2` 语义（新芯片侧消费）；mode=1 输出 BSND 4 维变体（本适配层统一 3 维 (T1,N2,·)，调用侧在新芯片侧由其实现给出）。
