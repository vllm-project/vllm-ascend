# GBSA A5 非量化偏移预计算与 K/V Gather 两阶段性能优化方案

状态：已放弃阶段二的数据 gather；当前工作区按最新简化方案从 `ba61fcc6` 重新实现，不再使用 `43183069` 的链接式描述协议。独立保留 `1≤blockY≤128` 接口约束及相关测试。第 11～15 节为历史实现和实测记录，不代表当前工作区代码。

当前实现：见 [GBSA偏移预计算简化实现方案](gbsa偏移预计算简化实现方案.md) 第11节。AIV scalar完成偏移、长度、跨页拆分及非PA双块配对，在UB组表后用MTE3批量写GM；Cube GetValue直接搬运，不判断页边界或计算配对。不开VF，不沿用位域/链接描述。重实现代码未提交，本轮parent、GM直写candidate及UB版ub_candidate的独立包、精度和性能产物保存在 `perf/gbsa_scalar_offsets/`，不与历史结果混算。

代码基线：`ba61fcc6`，已支持 A5 非量化 `residualBlockMode=1`，包括 PA、FD 和虚拟 residual block。该版本 custom kernel 与 metadata 已编译通过；15 个上板差分用例全部通过，mode 1 与等价 mode 0 输入的输出最大差值均为 0。该验证为差分验证，不代表独立 golden 验证。

开发分两阶段推进，并分别保留可测试的版本：

1. **阶段一：AIV 预计算偏移，Cube 成对分段搬运。** 参考 BSA 生成 QK/PV 共用的搬运描述，满足条件时一次搬运两个块，K/V 数据仍从原始 GM 直接进入 L1。
2. **阶段二：AIV 将 K/V 数据 gather 到连续 GM tile。** 在阶段一公共地址映射的基础上增加 UB 数据中转，Cube 改为连续加载。

先完成阶段一的功能、泛化和性能验收，再开发阶段二。保留原始基线、阶段一和阶段二，分别评估偏移预计算/成对搬运的收益，以及数据 gather 的增量收益。

两个阶段均面向当前 A5 非量化支持的整个 D≤512 范围；D=128 只是现有性能基线，不能作为资源分配或实现范围的限制。前文中的收益分析为开发假设；阶段一实施与验收记录见第 11、12 节，阶段二实施记录见第 13 节。

## 1. 性能基线与目标

2026-09-24 使用 `msprof op` 测试当前版本，配置如下：

| 参数 | Decode | Prefill |
| --- | ---: | ---: |
| B | 1 | 1 |
| QS | 1 | 16384 |
| KVS | 16384 | 16384 |
| D | 128 | 128 |
| Nq:Nkv | 16:1 | 16:1 |
| blockY | 16 | 16 |
| topK 容量 | 820 | 820 |
| 数据类型 | FP16 | FP16 |
| KV 布局 | 非 PA，TND | 非 PA，TND |
| Mask | causal | causal |
| softmax_precision | 0，高精度 | 0，高精度 |
| residualBlockMode | 1 | 1 |
| FD 核数 | 28 | 0 |
| Kernel 耗时 | 30.894 µs | 26020.818 µs |

完整 KV 范围内保留 820/1024 个块，稀疏率为 19.921875%。Prefill 各行按照可见完整块数选取同一比例的有效块，count 随行变化，不完整 causal 尾块由 mode 1 隐式补入。

每个 case 为一次 `msprof op` 采样，参数为 `--aic-metrics=BasicInfo,PipeUtilization --warm-up=1 --launch-count=1`。耗时为 GBSA kernel 时间，不包括输入构造与 metadata 执行。

原始数据位于仓库根目录：

- `perf/gbsa_topk820_decode/OPPROF_20260924104306_CLGNCOKHPFPYVOPE/`
- `perf/gbsa_topk820_prefill/OPPROF_20260924104410_ZDPRDFAQHBNLUBGI/`
- 精度日志：`perf/gbsa_topk820_precision.log`
- 测试脚本：`attention/generic_block_sparse_attention/tests/perf/gbsa_topk820_perf.py`

Prefill 各核平均流水线时间：

| 指标 | 平均时间 |
| --- | ---: |
| AIC 总时间 | 25.951 ms |
| AIC scalar | 25.813 ms |
| AIC MTE2 | 21.412 ms |
| Cube 计算 | 2.327 ms |
| AIV vector | 3.105 ms |
| AIV scalar | 2.382 ms |

各流水线时间存在重叠，不能相加。仅凭 scalar 占比不能将全部时间归因于偏移算术；结合 loader 实现，小块地址解析、GM 标量读取和 DMA 指令下发是本次优先优化对象。

## 2. BSA 参考改动的分类

参考仓库：`/home/lch/code/ops-transformer`。指定提交主要修改 `attention/step4_block_sparse_attention`。下表按作用分类，不将不同分支、不同矩阵布局的实现直接混用。

| 提交 | 实际作用 | 借鉴方式与限制 |
| --- | --- | --- |
| `bf7cf924`、`37c900e3` | 两个片段通过 `ndNum=2` 合并搬运，调整配对顺序 | 可参考减少 DMA 下发；GBSA 保留原 token 顺序，不直接照搬物理地址重排 |
| `037a8077` | AIV 预解析所有行，将搬运描述表写入 GM，Cube 读取描述表加载 K/V | 借鉴地址解析前移；这里写入 GM 的是描述表，不是 K/V 数据 |
| `1a8cdbd9` | 逐行解码、同步诊断，处理 V 配对 DMA 问题 | 参考描述表可见性验证；K 与 V 的 L1 布局需分别验证 |
| `5924445a` | 逐 tile 解码，提前一轮准备下一 tile；QK 保存搬运信息，PV 复用 | 本方案流水嵌入的主要参考 |
| `05fe17b8` | 修正 P 的 UB 实际占用、两路 AIV 的 32B 写入边界，以及 UB 重写前等待位置 | 直接纳入缓冲区与同步设计约束 |
| `bf916b38`、`1eabd3ca` | 动态尾块、decode 有效长度及搬运信息复用 | 借鉴末尾有效长度的处理；GBSA 继续使用自身 residual 协议 |
| `ed4243f4` | 固定小块地址信息缓存、QK/PV 复用和减少 scalar 工作 | 参考信息复用；固定 regionSize=8 的实现不适用于任意 blockY |
| `b1acb55e`、`9b65bfdf` | FD 调度与 workspace 隔离，去除公开 FD 开关 | 借鉴空间隔离；Step4 按 tile 分片，GBSA 当前按虚拟 block 列表区间分片 |
| 另一分支 `07e114bd9` | 从转置 QK 变为普通 Q×Kᵀ，去掉已有事件覆盖的 MTE2 barrier | 其矩阵布局更接近 GBSA；借鉴避免重复同步，不直接迁移整套 kernel |
| 正式 BSA `48a55f29f` | L0 stage 的 CeilDiv 修正、softmax UB/bank 与指令优化 | 属于计算与片上布局优化，不属于本次 K/V 数据 gather |

参考实现的 `PackedRegionDecodeOneTileSplitToGm` 主要生成描述信息，`PackedRegionRs8Tile listRing[PRE_LAUNCH + 1]` 保存 QK 已读取、供 PV 复用的信息。它的三份缓存从 QK 开始存活。阶段一借鉴地址预计算和复用思路，但 GBSA 需要按任意 blockY 和动态 tile 重新计算描述容量。阶段二的 K/V 数据更早生成，也不能直接套用三份缓存。

另一个区别是：参考实现中相当一部分 Vector decode 通过 AIV scalar 的 UB GetValue/SetValue 完成。把工作放到 AIV 不等于已使用 SIMD 批量计算。

## 3. 两阶段数据流与收益归属

当前路径：

```text
sparseIdx / pageTable ──→ QK scalar 解析 ──→ K 分段 GM→L1 ──→ QK
          └────────────→ PV scalar 再解析 ─→ V 分段 GM→L1 ──→ PV
```

阶段一路径：

```text
sparseIdx / pageTable
          │
          ▼
AIV：计算偏移、裁剪片段、生成 pair/single 描述
          │
          ▼
      描述表 GM ───────→ QK / PV 复用
                         │
原始 K/V GM ───────────→ Cube：每次两块 GM→L1
                         │ 尾块或不满足配对条件时 single
                         ▼
                       QK / PV
```

阶段一新增地址描述工作区与 UB 临时区，不分配 K/V 数据中转空间。地址解析和配对只执行一次；QK/PV 分别执行 K/V 搬运，仍需读取描述及下发 DMA。

阶段二路径：

```text
sparseIdx / pageTable → AIV 共用地址映射
                              │
原始 K/V GM ─→ UB chunk ─→ 连续 K/V GM tile
                              │           │
                              ▼           ▼
                         K 连续加载    V 连续加载
                              │           │
                              QK          PV
```

现有 D=128 基线的 tiling 为 `qBaseTile=16、kvBaseTile=512、mm1L1TileN=256`。对完整、非 PA 且满足配对条件的 tile：

| 版本 | QK 的 Cube 搬运次数 | PV 的 Cube 搬运次数 | K/V 数据中转 |
| --- | ---: | ---: | --- |
| 原始基线 | 32 | 32 | 无 |
| 阶段一 | 16 | 16 | 无，只传描述 |
| 阶段二 | 2 | 1 | AIV 经 UB 拼接到 GM |

这是该配置的结构分析，不是测量结果。PA、非整除 blockY、tail 和 D 方向 L1 切分会改变次数。两阶段都记录实际 pair/single 或 chunk 数量，不能把 64→32→3 当作全部场景的固定结果。阶段二仍保留 QK 的 L1 N/D 切分，不强行一次全载 K。

## 4. 公共地址映射与阶段一配对规则

继续使用 task/FD 分片已 rebased 的 `gTaskSparseBlockIdx`。对分片内部的 compact 坐标：

```text
compactToken = tileIdx * kvBaseTile + tileInnerToken
listPos      = compactToken / blockY
blockInner   = compactToken % blockY
logicalToken = GetSparseBlockId(listPos) * blockY + blockInner
```

复用 `GetSparseBlockId<hasResidualBlock>`。阶段一在 sparse block、PA 页、tile 和 QK L1 N 子块边界生成连续片段描述；阶段二根据同一映射增加 UB 数据 chunk 切分。有效 tile 长度继续由当前 validKvTokens 决定，不重新遍历整条列表。

### 4.1 偏移共用与 AIV 执行

- 非 PA 共用逻辑 token 映射，分别加 K/V 基址。
- PA 共用逻辑页、物理页与页内 token；最终偏移分别使用 `kStride0_`、`vStride0_`，不假定最终数值偏移相同。
- GM 地址乘法使用足够宽的整数；不得通过 FP32 中转大地址。
- 阶段一即在 AIV 上批量准备偏移，规则性的 block 起点和 stride 运算使用 VF；PA 页表访问、片段裁剪、描述下发仍需 scalar 配合。明确记录实际 VF/scalar 分工，不能仅移动原 scalar 循环就称为 SIMD 优化。
- 两路 AIV 按 KV token 区间分工，每路为自身区间生成 K/V 共用的映射；阶段二每路同时搬运对应的 K 和 V。分工独立于 softmax 的 Q 行划分，没有有效 Q 行的 AIV 也可能需要准备数据和参与同步。
- UB 按有界窗口读取当前 tile 所需的索引、生成描述，不按整个 topK 分配；窗口间保留推进状态。虚拟 residual block 不触发越界 sparseIdx 读取。

### 4.2 一次搬运两个块

一次两块指一次 `copyGMToL1` 多矩阵搬运下发，例如 `ndNum=2`，不是连续调用两次单块 copy。AIV 预计算片段源偏移、长度、目标位置及配对参数，Cube 消费描述。

配对条件：

1. 两段在 compact 顺序中相邻且形状相同，并位于本次 L1 加载范围内；不跨 QK L1 N 子块强行配对。
2. 源矩阵 stride 满足 copy 接口的单位、对齐、范围和方向要求。PA 物理页可乱序，不能为正 stride 改变 token 顺序。
3. 分别计算 QK nZ 与 PV zN 的目标矩阵 stride；共用逻辑映射不等于共用 L1 搬运参数。
4. D 方向使用本次实际宽度。D>256 时，QK 对不同 L1 K 子块复用 N 方向描述，仅补充 D 内偏移和本次宽度，不重复解析 sparseIdx/pageTable。

奇数段、不同长度的 residual/tail、跨页后不满足条件的片段，以及不可编码的 stride，生成 single 描述。K/V 各自配对条件不同则允许分别走 pair/single。所有回退均消费已计算描述，不退回逐块重复解析地址。

描述容量由实际 tile、任意 blockY、PA 跨页及 L1 子块边界推导，不沿用 BSA 固定 regionSize=8 或固定 32 段上限。记录的字节数以最终字段和对齐为准；地址宽度、单位、pair 标记与有效条目数必须定义清楚。

### 4.3 阶段二的连续数据格式

K/V 目标 GM 格式均为 `[kvBaseTile, paddedD]`，保留 FP16/BF16 位模式，`paddedD` 满足 32B 行对齐。QK 将其作为列主序 Kᵀ 视图读取，PV 作为行主序 V 读取；gather 不转置 K。

搬运支持原始 D 的实际字节数与 stride。D padding 不改变计算宽度，KV padding 不参与 softmax；所有阶段仍使用实际 `kvSTileSizeAct`。两路 AIV 的描述区和数据区写入都须拥有独立的 32B 块。

## 5. 两阶段 prelaunch 与缓冲生命周期

两个阶段均保留 `PRE_LAUNCH=2`，准备阶段提前 QK 一轮。Prepare 表示阶段一的地址描述生成或阶段二的数据 gather：

| 轮次 | AIV 下发顺序 | AIC 下发顺序 |
| --- | --- | --- |
| 初始化 | Prepare 0 | 加载 Q |
| 0 | Prepare 1 → Softmax 0 | QK 0 |
| 1 | Prepare 2 → Softmax 1 | QK 1 |
| 2 | Prepare 3 → Softmax 2 → Rescale 0 | QK 2 → PV 0 |
| 3 | Prepare 4 → Softmax 3 → Rescale 1 | QK 3 → PV 1 |
| 后续 | Prepare(i+1) → Softmax(i) → Rescale(i−2) | QK(i) → PV(i−2) |

这是下发顺序，不表示各阶段等时。目标是让 Prepare(i+1) 与 QK(i) 重叠；尾部不准备越界 tile，只 drain 已存在的计算。两阶段不使用 L0，维持 QK/PV 顺序后继续使用当前 `CalcCrossMm1Mm2PrefixSumL0ABStages()`。

### 5.1 阶段一：GM 描述槽位

建议先为每个物理 AIC 配置四份轻量 GM 描述槽位，保留到对应 PV 消费结束。QK/PV 读取同一份预计算描述；避免为任意 blockY 分配很大的 Cube 栈上 snapshot 数组。

```text
AIV 等待 Free → 写描述 → MTE3 Ready
  → AIC 等待 Ready → QK 使用描述 → PV 使用描述
  → 描述消费完成 → Free
```

回收必须晚于 Cube 对该槽位的最后一次 scalar 读取；第一版可在 PV 加载阶段结束时回收。描述 Ready 的等待需约束实际读取描述的 scalar 路径，不能只约束后续 MTE2。

BSA 的三份 QK→PV snapshot 可作为后续对照优化：仅当所有字段已快照、容量有界且无明显寄存器/栈开销时，才让 GM 描述更早回收。它不是第一阶段的必要前提。

### 5.2 阶段二：K/V 数据槽位

K/V 各保留四份槽位：

```text
dataGmStages = PRE_LAUNCH + prepareAhead + 1 = 4
slot = tileIdx % dataGmStages
```

生成 tile 3 时 V0 尚可能未读入 L1，同时存在 tile 0/1/2/3。三槽位可以增加等待实现，但会削弱提前准备效果。

```text
AIV 等待 Free → 两路 AIV 写完各自 K/V → MTE3 Ready
  → AIC 等待两路 Ready → QK 读取 K → PV 将 V 读入 L1
  → V 的 MTE2 读取完成 → Free
```

数据 Free 可早于 PV MMAD/rescaleO 完成，因为后续只使用 L1 的 V；不能在 QK 完成时释放同槽位中的 V。

### 5.3 同步约束

- 阶段一 Ready 覆盖完整描述，阶段二 Ready 覆盖全部 chunk 的 K/V 写入。
- UB 描述窗口或数据 chunk 的 MTE3 完成后才能重写，等待必须位于重写之前。
- 两路 AIV 的空区间仍执行所需握手。
- 独立分配跨核 flag 和本核 HardEvent，检查与 S/P/LO、FD combine 和 SyncAll 的冲突。
- 初始化、0/1/2/3/4/5 tile 短任务、多 task 切换、最后 drain 的 set/wait 必须配对。
- 使用对应 pipe 的局部事件，避免每 tile 增加 SyncAll/PIPE_ALL 或重复 barrier。

## 6. 分阶段 GM workspace

### 6.1 阶段一

```text
descriptors[physicalAicNum][4][alignedDescriptorBytesPerTile]
```

包含两路 AIV 各自 32B 对齐的写入区域。容量覆盖任意 blockY 下的最坏片段数、pair/single 参数及头尾信息，不按 QS/topK 分配全量描述，也不预分配阶段二的 K/V 数据区。

### 6.2 阶段二

```text
gatherK[physicalAicNum][4][kvBaseTile][paddedD]
gatherV[physicalAicNum][4][kvBaseTile][paddedD]
```

新增数据区为 `2 * physicalAicNum * 4 * kvBaseTile * paddedD * sizeof(KV)`。D=128 基线对应 28 MiB；其他 D 必须结合它实际选中的 kvBaseTile 重新计算。公共地址映射在 UB 中生成即可，不要求阶段二继续保留已不再使用的完整 GM 描述表。

### 6.3 与既有 workspace 的关系

```text
现有 pipeline/LO GM 中转区
→ 当前阶段的描述区或 K/V 数据区
→ FD partial LSE 区
→ FD partial O 区
```

host `CalculateWorkSpace()` 与 kernel 保持一致，新增区域之后重新安排 FD 偏移。使用 uint64_t 计算容量并进行 host 溢出检查；维持 libapi workspace 协议。tilingdata 传递模式、容量、偏移及必要的行数参数。

## 7. 面向整个 D≤512 范围的 UB 设计

### 7.1 预算维度与公共公式

UB 分配必须联合考虑实际 D、qBaseTile、kvBaseTile、softmax 精度、LO 缓冲策略和当前优化阶段。不能按 D=128 剩余容量设置统一的描述大小或数据 chunkRows。

沿用 `UBufTileHelper` 统一管理，按当前实现定义：

```text
Q = RoundUp(qBaseTile / 2, 16)
T = RoundUp(kvBaseTile, 32)
Du = RoundUp(D, 32)          // 现有 LO/GO 行跨度
Dg = RoundUp(D, 16)          // FP16/BF16 gather 的 32B 对齐跨度
bS = sizeof(ElementS)       // 高精度 4 bytes，低精度 2 bytes
bKV = sizeof(ElementKV)     // 非量化 FP16/BF16 为 2 bytes

U_SP    = 2 * Q * T * (bS + bKV)
U_GO    = Q * Du * sizeof(float)
U_STATS = 3840 bytes        // 当前固定 stats/LSE 布局，修改布局时重新计算
U_MASK  = Q * T             // 保守保留当前预留区
U_FIXED = U_SP + U_GO + U_STATS + U_MASK

U_LO_resident = loBufNum * Q * Du * sizeof(float)
U_LO_chunk   = 2 * loRows * Du * sizeof(float)
U_ADDR       = 对齐后的索引、偏移、描述窗口与临时寄存器落地空间
U_DATA       = 2 /* K/V */ * 2 /* ping/pong */ * dataRows * Dg * bKV
```

按当前 248 KiB 可用预算计算，具体空间顺序可以调整，但所有区间必须互不覆盖。row stride、32B 对齐及每种数据类型分别计算；相同 D 不同 softmax 精度也需重新求解。索引/描述窗口有界，D 方向子块尽量复用同一 N 映射。

### 7.2 阶段一：只增加地址工作区

```text
U_FIXED + U_LO + U_ADDR <= U_AVAILABLE
```

先预留必要的地址窗口，再选择 LO 的完整双缓冲、完整单缓冲或 GM 中转。地址窗口按可用 UB 缩小并分批写入完整 GM 描述槽位，不能要求最坏情况下整份描述同时驻留 UB。

D≤512 均需考虑以下情况：

- 资源宽裕时，维持当前 LO 模式，地址工作区使用独立区域。
- 当前 LO 双缓冲几乎占满 UB 时，评估完整 LO 单缓冲；确保 PV/rescaleO 使用实际缓冲数量配对同步。
- LO 需要 GM 中转时，先扣除地址工作区，再计算 LO 双缓冲 chunk 行数。
- 不把 `D>256 && qBaseTile>64` 当成新设计唯一的中转判断依据；新增空间后应按预算决定。任何模式变化都需进行相应精度和流水验证。

### 7.3 阶段二：地址区、K/V chunk 与 LO 联合分配

```text
U_FIXED + U_ADDR + U_LO + U_DATA <= U_AVAILABLE

dataRows <= (U_AVAILABLE - U_FIXED - U_ADDR - U_LO)
            / (4 * Dg * bKV)
```

K/V 各用 ping/pong，且同一数据窗口复用地址计算。若 LO 同时通过 GM 中转，则联合选择 `loRows` 和 `dataRows`，均需至少有一个合法行块；不能让 LO 先独占全部剩余空间。按搬运接口要求对行数/字节数取整，不能套用 D=128 的 64 行。

若资源不足或所需 chunk 太小，tiling 可选择阶段一或原路径，维持已支持输入的功能范围；性能路由不能退化成只允许 D=128。对资源足够的各类 D 均提供第二阶段验证用例。

### 7.4 当前高精度布局的代表性预算

下表是**加入两阶段新增空间之前**的现状，用来展示必须处理的资源差异。单位为每路 AIV 的 KiB；预留区已计入。低精度、D 非整齐值和其他 tile 组合使用公式重新计算。

| D | qBaseTile | kvBaseTile | 当前 LO 模式 | 当前剩余 UB |
| ---: | ---: | ---: | --- | ---: |
| 128 | 16 | 512 | 完整双缓冲 | 116.25 |
| 128 | 128 | 128 | 完整双缓冲 | 44.25 |
| 256 | 32 | 512 | 完整双缓冲 | 92.25 |
| 256 | 64 | 256 | 完整双缓冲 | 44.25 |
| 256 | 128 | 128 | 完整单缓冲 | 12.25 |
| 512 | 16 | 384 | 完整双缓冲 | 70.25 |
| 512 | 32 | 256 | 完整双缓冲 | 96.25 |
| 512 | 64 | 128 | 完整双缓冲 | 0.25 |
| 512 | 128 | 128 | LO 经 GM，当前每份 3 行 | 0.25 |

特别是 D=512、qBaseTile=64，虽然当前 LO 没有经过 GM，中间空间也已基本用尽；两种 D512 紧张组合都不能在末尾简单追加空间。阶段一实施中确认非量化 mask 预留区没有实际使用，因此改为复用该区域；由此保留原有 LO 模式/行数，详见 11.3。阶段二新增数据缓冲仍须重新计算预算。

D=128、qBaseTile=16 时，阶段二 64 行 K/V 双缓冲消耗 64 KiB，仅作为公式的一个可行示例。S/P/LO/GO 均有跨阶段生命周期，第一版不在缺少事件保护时重叠借用这些区域。

## 8. 两阶段代码适配点

| 位置 | 阶段一 | 阶段二增量 |
| --- | --- | --- |
| host tiling / tilingdata | 地址模式、描述 workspace 与全 D UB 预算 | 数据 gather 模式、K/V workspace、LO/data chunk 联合预算 |
| `gbsa_layout_utils.hpp` 及 AIV block 级准备函数 | 公共映射、VF 偏移准备、pair/single 描述 | 复用映射，增加 GM→UB→GM 数据搬运 |
| 主 kernel | Prepare 前推、描述 Ready/Free、短任务 drain | 数据槽位 Ready/Free 与数据 chunk 流水 |
| `UBufTileHelper` / `CalcUBufTileInfo()` | 有界地址窗口与必要的 LO 模式调整 | K/V 双缓冲与 LO 共同分配 |
| QK `SparseKL1TileNLoad` | 读取预计算描述，按 pair/single 搬运；D 子块复用 | 读取连续 K，并沿原 L1 N/D 切分 |
| PV `SparseVBaseTileL1FullLoad` | 复用同一映射，使用 PV 布局参数成对加载 | 连续 V 全载，MTE2 完成后释放数据槽位 |
| DispatchPolicy / interface / key | 保留原模式，引入偏移预计算模式 | 增加数据 gather 模式用于独立对比 |
| 测试 | 描述正确性、成对 DMA、全 D 精度与性能 | gathered K/V 一致性、数据槽位与两阶段性能差异 |

保留同一份主 kernel/MMAD 逻辑，由 DispatchPolicy 透传编译期加载模式；需要新增 key 时使用独立位并逐条展开。保持原有代码风格，不引入 Plan/LoadContext 包装简单统计信息，不迁移 Step4 的 packed 输入协议。全量化默认使用原加载模式，public API 和 metadata 调度协议不变。

## 9. 泛化兼容与阶段验收

| 场景 | 两阶段共同要求 |
| --- | --- |
| 任意 blockY≤128 | tile 可从 block 中间开始，跨多个 block；不假设整除 |
| PA | 跨页、随机物理页顺序；不因成对 DMA 或 gather 改变逻辑顺序 |
| residual mode 0/1 | count=0、residualId=0、恰好完整块、奇数尾段 |
| FD | 按 rawBegin/rawEnd 的局部坐标准备，独立核空间，覆盖 residual-only 分片 |
| D≤512 | 计算全 D 资源预算，验证非对齐 D、L1 D 切分与 LO 单/双缓冲和 GM 中转 |
| kvBaseTile=128/256/384/512 | 描述和数据容量随 tile 变化，遵守高低精度 softmax 能力 |
| FP16/BF16、高低精度、LSE | 保持 dtype、有效长度、softmax 顺序与 LSE 行为 |
| 多 batch/KV head、seqused | 正确 storage prefix、实际长度与 head stride |
| 空任务及短列表 | 不执行非法搬运，握手及 FD neutral 输出正确 |

### 9.1 阶段一交付

1. 完成全 D UB 预算、地址准备、描述工作区、成对 Cube 搬运和同步。
2. 检查描述与 CPU 逻辑映射一致；重点验证 pair 的源/目的 stride、QK/PV 不同 NZ 布局，以及不能配对时的 single 路径。
3. 小用例用独立 FP32 golden，大用例与原加载路径做差分；记录所有实际启用的 UB/LO 模式。
4. 形成可独立复现的阶段一版本，保存编译产物信息、输入种子和性能报告，完成后再进入阶段二。

### 9.2 阶段二交付

1. 复用阶段一映射，增加数据 gather 和连续加载；gathered K/V 与 CPU 选取结果逐元素检查，搬运保持位模式。
2. 验证两路 AIV、0/1/2/3/4/5 tile、跨任务复用、尾块及 D512 LO/data chunk 并存的生命周期。
3. 在阶段一的同一批输入上验证输出精度与性能，分别比较原始路径和阶段一。
4. 根据实测确定最终 tiling 路由，保留三个版本的可对比性。

headDim 覆盖不只取 128/256/512：至少包含小 D、32/64/128/256 对齐边界两侧、384 和 512，并交叉 qBaseTile=16/32/64/128、高低精度、PA、FD、多个 KV head。按实际已支持输入约束选择案例。

## 10. 性能对比与风险

统一使用相同输入、稀疏模式、dtype、tile、设备和 msprof 参数比较三个版本。若为了容纳工作区而改变 LO 模式或 tile，单独记录该变化，避免将其收益误计入偏移准备或成对搬运。

| 比较 | 要回答的问题 |
| --- | --- |
| 原始路径 → 阶段一 | 地址解析复用和 pair DMA 能否降低 Cube scalar 与总耗时 |
| 阶段一 → 阶段二 | 进一步减少 Cube 小块搬运，能否覆盖新增数据中转成本 |
| 原始路径 → 阶段二 | 最终端到端 kernel 性能是否改善 |

记录 kernel 总时间、AIC/AIV scalar、Cube/vector、MTE2/MTE3、pair/single 比例、GM workspace、UB 地址窗口和 LO/data chunk 行数。性能用例除已有 D=128 decode/prefill 外，必须覆盖 D≤256 与 D>256 各 tiling 档位，特别是 D=512 的 qBaseTile=64/128 资源紧张组合。

阶段一主要风险是描述 GM 读写、Cube scalar 读取描述的成本、VF↔scalar 数据依赖、paired DMA 的格式约束，以及新增地址区引起的 LO 模式变化。它不增加 K/V 数据的 GM 中转流量。

阶段二增加原始数据读取之外的 GM 写入和再次读取。以 D=128 基线完整 tile 为例，K+V 原始读取 256 KiB，数据 gather 路径逻辑 GM 流量约 768 KiB。实际 HBM 流量受 L2 影响，不直接按三倍估计；其他 D/tile 必须重新计算。还要检查 AIV MTE3 与 P→L1、MTE2 与 LO 读回的竞争，以及 decode 启动开销。

最终以精度和 kernel 总时间验收，不能仅凭 AIC scalar 下降判断成功。阶段一和阶段二分别验收并保留结果；D=128 基线不代替整个支持范围的功能和资源验证。

## 11. 阶段一实施记录（2026-09-24）

本阶段只传递地址描述，**没有**将 K/V 数据 gather 到 GM。计算主流水、FD 调度、metadata 协议和公开 API 不变。

### 11.1 已落实的实现

- `DispatchPolicy::PRECOMPUTE_SPARSE` 控制原 QK/PV loader 对描述的消费。A5 非量化实例传 `true`；全量化保持默认 `false`。当前使用独立 baseline/候选 custom 包对比，不增加运行时模式或新 tiling key。
- `PrepareSparseLoadInfo<isPaged, hasResidualBlock>` 使用当前 task 已 rebased 的稀疏列表。VF 批量计算 `idx * blockY` 的逻辑起点，且结果在有界索引窗口内跨 tile 复用；PA 页查表、边界裁剪、配对和 64 位最终 GM 地址仍由 AIV scalar 计算。不是所有地址计算都已 SIMD 化。
- 每个 MM1 L1-N 子块按 N 分成两半，两路 AIV 各准备一半；无有效 Q 行或无有效 KV 半块的 AIV 仍完成握手。窗口大小与 topK/QS 无关。
- 每半块拥有独立 32B header（有效描述数）和描述区，按 compact 顺序组织。原 `SparseKL1TileNLoad` / `SparseVBaseTileL1FullLoad` 调用共用消费函数；QK 的 D 子块复用同一组描述。
- 四份 GM 槽位维持 Prepare(i+1)、QK(i)、PV(i−2) 的生命周期。Ready 等待使用 `PIPE_S`；PV 消费完成后释放槽位。跨核 flag 为 7…10 / 23…26，避开已有 S/P/LO 与 SDK SyncAll 的 11…14。
- 本地地址准备事件使用 ID5，各次使用均 set/wait 配对。UB 描述输出后等待 MTE3 完成，才复用暂存。
- 增加 `sparseLoadInfoOffset`、`sparseLoadTileWords` 两个 tiling 字段；FD partial 工作区接在描述区之后。direct-kernel 回归脚本同步更新了字段顺序和工作区容量。

### 11.2 描述格式与缓存一致性

所有 GM 偏移/stride 单位均为元素；源偏移保留 uint64，不经 FP32 中转。

| 格式 | word 0 | word 1 | word 2 | word 3 |
| --- | --- | --- | --- | --- |
| PA，32B | K offset uint64 | V offset uint64 | K stride uint32 + V stride uint32 | length uint32 + ndNum uint32 |
| 非 PA，16B | K/V 共用 offset uint64 | stride uint32 + length uint16 + ndNum uint16 | — | — |

只有相邻片段等长、源偏移递增、stride 满足本实现的 uint32 范围及 16 元素对齐要求时，才将其配对为 `ndNum=2`。其余情况使用 single，不重排 token。该范围/对齐检查是本实现的保守选择，不代表 A5 硬件只支持 32 位 stride。

GM 容量统一按 PA 的上界预留；非 PA 只写有效的紧凑记录并补齐 32B 边界。若 `N=mm1L1TileN`，每 tile 的 uint64 word 数为：

```text
ceil(kvBaseTile / N) * 2 * (N/2 + 1) * 4
```

首轮上板发现了槽位复用时的 scalar cache 旧数据问题：MTE3 Ready 只解决执行顺序，不会自动使 AIC scalar cache 失效。现于 QK 的首个 D 子块对 header 和实际读到的每条 64B cache line 失效；后续 D 子块与 PV 可复用缓存。扩大到 58 个用例后仍全部通过 golden，且与基线逐元素一致。

### 11.3 实际 UB 窗口与索引复用

审查非量化 kernel 后确认 `maskStartOffset` 仅预留空间，没有创建/使用 mask tensor。参考 BSA，将地址窗口放到该区域，不与任何活跃的 S/P/LO/GO 张量重叠。UB 尾区取 `max(原 mask 预留字节数, U_ADDR)`，不是两者相加。全量化不采用此路径。

```text
U_ADDR = RoundUp((N/2) * 4, 32) + (N/2 + 1) * 32
N=256: 4640B
N=128: 2336B
```

索引窗口在同一 task 的 L1 半块和 FA tile 之间保留，只有请求区间超出窗口时才重新 GM→UB 和 VF 处理；任务切换时重置。与 BSA 一次加载整行不同，这里最多缓存 N/2 个 index，空间不随 topK 增长。

穷举 D=1…512、qBaseTile=16/32/64/128、高低精度当前合法 tile，均无 UB 溢出。D=512、qBaseTile=64 保留原 LO 完整双缓冲；D=512、qBaseTile=128 保留原 GM 中转每份 3 行。早期独立增加地址区曾分别改成单缓冲/每份 2 行，造成性能回退，该分配方案已撤销。

### 11.4 验证与性能状态

- `tests/ut/op_kernel/test_sparse_load_stage1_device.py`：29 组输入 × residual mode 0/1，共 58 例，含 FP16/BF16、高低精度、D 边界值、PA 页乱序、K/V 不同页 stride、双 batch/多 KV head、FD 与描述环多次复用。先以 baseline custom 包对独立 FP32 golden 验证并保存输出，再在候选包下逐元素对比。
- `tests/ut/op_kernel/test_sparse_load_stage1_geometry.py`：描述重放、配对顺序/容量、紧凑记录边界、超范围 stride 回退，以及全 D UB 预算。该 CPU 模型测试不是实际 DMA 的替代；搬运由上板 golden 验证。
- 原 `test_sparse_kv_geometry.cpp` 的有效长度、FD rebased 区间和 residual 测试继续通过。
- `tests/ut/op_kernel/profile_sparse_load_stage1.py`：默认复现原 D128 decode/prefill，支持 `--dim` / `--group` / `--qs` / `--precision` 做资源档位对比。所有性能产物位于 `perf/gbsa_stage1/`。

首个通用 32B 描述版本虽然精度通过，但 prefill 性能回退：26.030ms → 35.478ms。最终紧凑描述 + 索引窗口 + UB 复用版本为 26.063ms（+0.13%，基本持平）；decode 30.992µs → 30.581µs。D256/G32 仍回退 4.02%，D512/G64 改善 8.60%，D512/G128 改善 0.84%。因此尚不能声称全面优化成功。完整配置、基线重复采样、构建包与数据见 `perf/gbsa_stage1/README.md`。

未覆盖或尚未验收的范围不得视为已通过：public API 仍拒绝 LSE 输出，本次 API 测试不包含它；阶段二的数据 gather 尚未实现。


### 11.5 BSA 对照与性能归因

对照 `step4_block_sparse_attention` 的 `PackedRegionLoadPackedRowToUb`、`PackedRegionDecodeOneTileSplitToGm`、`PackedRegionRs8LoadHalf/IssueHalf` 和主 kernel，逐项实验，而非只比较“工作是否移到了 AIV”。

| 差异 | BSA 参考 | GBSA 实施与结果 |
| --- | --- | --- |
| 描述紧凑度 | rs=8 的固定 pair/tail 编码 | 任意 Y/PA 首版统一 32B 过重；非 PA 压至 16B，prefill 从 35.478ms 降至 26.221ms |
| 索引重复搬运 | 任务开始一次读取整行 packed index | 有界窗口跨 tile 保留；AIV 平均 MTE2 从 2.716ms 降至 0.812ms，不随 topK 增长 UB |
| UB 预留复用 | ubPacked/ubList 放在 maskStartOffset | GBSA 同样复用已确认未使用的 mask 预留；D512/G128 从 2.800ms 恢复至 2.456ms，原始基线 2.469ms |
| QK→PV snapshot | 固定 RS8 数组、固定长度 pair/tail 搬运 | 试过三槽、每槽 128 个 uint64 的有界缓存，仅非 PA/Y≥16；58 例精度通过，但 prefill 增至 27.241ms，已撤销，保留实验 patch 与 profile |
| 描述 GM 生命周期 | 不同 task/tile 使用不同地址 | GBSA 使用每核四槽有界环；额外需要 Free/Ready 和 scalar cache 失效，不能照抄其省略同步的做法 |
| 泛化成本 | RS8 路径能使用固定块长度、紧凑字段 | GBSA 支持任意 Y、PA 随机页和不同 K/V stride，需要动态片段长度、边界与配对判断；不重排 token |

快照实验中，prefill 的 AIC 平均 MTE2 虽然由 17.819ms 降至 16.830ms，AIC scalar 却由 24.414ms 增至 25.365ms，总时间变差。由此判断该通用 snapshot 实现不值得保留；它新增的数组读写/解包是待进一步指令级确认的候选原因，不能仅凭源码断言已发生寄存器 spill。

D256/G32 的不带 snapshot 版本仍约回退 4%；AIC 平均 MTE2 由 1.036ms 降至 0.984ms，但 scalar 由 0.967ms 升至 1.018ms。与 D128 相比，它每个 FA tile 有更多 L1 子块和半块描述，控制/读取成本的摊销不同。现有证据支持“描述开销抵消部分搬运收益”，尚不能把差额全部归到某一条指令。

因此，本阶段的地址预计算和 pair DMA 功能已通过精度验证，但不声称已实现全面性能提升。后续若继续优化阶段一，应优先量化描述传递/同步与动态搬运下发的指令成本；阶段二再单独评估数据 gather，不能用阶段二结果掩盖阶段一的回退。

## 12. 单层描述流优化（2026-09-24，阶段一保存版本）

第 11 节保留上一版的实现和实验记录；本节覆盖当前描述协议。最终包编译通过，58/58 例精度通过且与原始 baseline 逐元素一致。

### 12.1 Cube 只消费一条描述流

原先 `LoadSparseTile` 是“半块循环 → 读取 count → 描述循环”。当前改为单层循环：读取 source/control → 下发 copy → 取 control 中预计算的 nextWord/nextToken，直到覆盖本次 L1 范围。

- QK 的 L1 N/D 切分、L0 MMAD 循环不变，单层化只针对 GM→L1 的 sparse 描述消费。
- 两路 AIV 仍独立生成各自半块，不新增 AIV 间同步。每半块最后一条描述直接链接到下一半块首地址，中间预留空洞不读取。
- 不再读取半块 header/count，不再在 Cube 上计算 `dstToken += len * ndNum`，也不再计算下一半块的位置。
- QK 加载起点始终在 L1-N 边界，PV 起点为 0，因此描述起点直接为 `tileOffset * 4`（uint64 word）；移除了 QK/PV 的 `sparseLoadTileN` 透传参数。
- 空半块不写描述；有效 token 是连续 compact 前缀，消费者在上一有效记录的 nextToken 处停止，不访问空半块。

### 12.2 字段与容量

非 PA 仍为 16B `[sourceOffset, control]`；PA 仍为 32B `[kOffset, vOffset, k/vStride, control]`。control 位布局：

| bit | 内容 |
| --- | --- |
| 0…31 | 非 PA 的 source stride；PA 此部分为 0 |
| 32…39 | 本条搬运的单块长度 len |
| 40…41 | ndNum，1 或 2 |
| 42…53 | 下一条描述相对当前 FA tile 的 uint64 word 索引 |
| 54…63 | 下一段相对当前 FA tile 的 compact token 起点 |

下一条信息利用原字段的空余位，不扩大单条描述。sourceOffset 保留 uint64，不经 FP32；PA K/V stride 仍分别保存。当前 `kvBaseTile<=512、halfN<=128`，最末 nextWord<=2048、nextToken<=512，均可编码。未来若增大这些 tiling 上界，必须重新检查此 ABI，不能无条件复用窄字段。

每半块容量为 `halfN * 32B`，去掉旧 32B header，首地址自然按 64B 对齐。GM tile 容量为 `ceil(kvBaseTile / N) * N * 32B`；每路 AIV 的 UB 窗口为 `RoundUp(halfN*4,32) + halfN*32`，N=256 时为 4608B，N=128 时为 2304B。仍复用未使用的 mask 预留，不改变 LO 模式/行数。

QK 首个 D 子块仍需对实际读到的每条 64B scalar cache line 失效，后续 D 子块与 PV 复用。该一致性操作不因循环简化而删除。四槽 Ready/Free 生命周期不变。

### 12.3 验证口径

与上一版 `perf/gbsa_stage1/final_*` 对比本轮增量，同时保留最初 baseline 对比。精度使用相同 58 例独立 FP32 golden 与原始 baseline 逐元素对比；CPU 测试新增单层流的跳转、字段上界、缓存线遍历、空/单 token、512 token 和 QK/PV 分段重放。

最终性能与证据归档到 `perf/gbsa_stage1_flat/`，分别报告 kernel 总耗时和平均 AIC scalar 时间，不以减少源码循环层数替代性能验收。

### 12.4 实测结果

相对上一版双层循环，五组 kernel 首测耗时变化为：Prefill -5.59%、Decode -1.01%、D256/G32 -6.88%、D512/G64 -5.78%、D512/G128 -2.57%。对应平均 AIC scalar 为 -5.81%、-4.52%、+0.80%、-8.73%、-8.80%。因此保留本轮实现，但不声称所有场景 scalar 都下降。

交替复测 Prefill 为 26.032ms→24.729ms，D256/G32 为 1.426ms→1.325ms；收益方向一致。D256 scalar 的小幅上升同样复现。Decode 小幅改善缺少独立重复样本，不作为稳定收益承诺。完整报告见 `perf/gbsa_stage1_flat/README.md`。


## 13. 阶段二：AIV gather 连续 K/V（2026-09-24）

阶段一已提交为 `43183069`，作为独立的地址预计算/单层消费对照。阶段二不再保留 GM 描述表，非量化 DispatchPolicy 的第三个布尔量为 `GATHER_KV`。全量化仍默认 false，保留原 sparse loader。

### 13.1 实际数据流

```text
有序 sparseIdx + 可选虚拟 residual + 可选 PA 页表
             │  两路 AIV 各处理 FA tile 的一半
             ▼
  共用 K/V token/page 映射 → 原始 GM K/V
             │ MTE2（稀疏块/页/chunk 边界分段）
             ▼
  UB: [K0][V0][K1][V1]，每块 gatherRows × RoundUp(D,16)
             │ MTE3（每个 chunk 连续写）
             ▼
  GM: [物理核][四槽][K/V][kvBaseTile][RoundUp(D,16)]
             ├─ QK: 按原 L1-N/D 切分，每片一次连续 GM→L1
             └─ PV: 一次连续 V tile GM→L1
```

保留原 MMAD、softmax、rescaleO 和 FD 调度。`SparseKL1TileNLoad` / PV 原加载函数通过 `if constexpr (GATHER_KV)` 使用连续搬运，删除 GM 描述参数和标量 cache 失效操作；不是另写一套矩阵乘主循环。两路 AIV 不互相等待，有效 token 少于半 tile 时，空半块仍发送 Ready。

偏移均按 uint64 元素地址计算；PA 共享逻辑 token/page 查找，但分别使用 K/V page stride。跨页、非对齐 D、稀疏块边界在 AIV 分段，D 向 16 对齐存储，每行补零。行连续且 D 对齐时，将块内所有行合为一次整段 DMA，避免逐行 stride DMA；其他布局保持逐行 strided copy。DMA 的 uint32 source stride 不足以表示实际行距时退化成单行拷贝，不截断 GM 地址。索引窗口仍由 L1-N/2 限定，不随 topK 增长；虚拟 residual 在本地窗口中补入。

### 13.2 UB / GM 预算

新增空间来自未使用的 mask 预留和 UB 尾部余额，不覆盖活跃的 S/P/LO/GO/statistics；沿用 `UBufTileHelper`，无新增独立规划结构体。

- 索引：`I = RoundUp((L1_N/2)*4,32)`。
- 每个 gather 行占用：`4 * RoundUp(D,16) * sizeof(KV)`，即 K/V 各双缓冲。
- 先按原规则计算 LO 是否走 GM、LO 缓冲个数及 LO 行块大小，再从剩余空间计算最大 gatherRows。
- gatherRows 取不超过预算和索引容量的最大 2 的幂；当前 D≤512 的全部 tiling 组合至少能放一行。
- GM K/V：`physicalAIC * 4 * 2 * kvBaseTile * RoundUp(D,16) * 2B`。放在既有 LO workspace 后，FD partial workspace 随后分配，互不重叠。

高精度代表组合：D128/Q16 为 64 行，D256/Q32 为 32 行，D512/Q64 为 1 行，D512/Q128 为 2 行。后两者的原 LO 模式和行数不变；小 gather 行块可能带来较大指令/同步开销，不能仅凭 Cube 循环减少判断整体收益。

### 13.3 同步与流水

保留 `PRE_LAUNCH=2`：先 gather tile0；循环 i 在 softmax i 前 gather i+1。四槽而非三槽允许准备 i+1 时保留等待 PV 的 i-2 槽位。

- AIV 等待 Free 后复用槽位，所有 K/V 的 MTE3 写结束后发 Ready。
- Cube 的 Ready 等待挂在 MTE2 上，两路 AIV 都 Ready 后才读 gather 数据。
- Cube 在 PV 之后由 **PIPE_MTE2** 发 Free。阶段一的 PIPE_S 释放只能保护地址读取，不能用于阶段二的数据寿命。
- UB 双缓冲使用 MTE3_MTE2 / MTE2_MTE3 的事件 4、5，kernel 开始初始化 Free，结束回收；不在每个 tile 重复初始化。

### 13.4 验证

完整 custom 包编译通过。58/58 个设备用例通过独立 FP32 golden，且与原始 baseline 输出逐元素一致；包含 residual0/1、FP16/BF16、高低精度、PA 页随机排列及不同 K/V page stride、FD、D1～512 的对齐边界、空任务、residual-only 和多槽复用。CPU 新增 `test_kv_gather_geometry.py`，检查全部 D/Q/精度组合的 UB 容量以及全部 Y=1～256 的分段 gather 重放。最终连续 DMA 版另补 58 例单 KV 头用例，均通过独立 golden（这组不声称有保存的 baseline 精确对比）。共 116 次最终包精度调用。全量化、arch22 与 LSE 输出未做本轮设备回归。

阶段二测试包、原始 profiler 和性能对比保存在 `perf/gbsa_stage2/`；性能结论以该目录最终报告为准，不把 Cube scalar 的减少等同于总耗时下降。

### 13.5 性能验收结果

阶段二功能通过，但性能未达标。最终版相对阶段一 `43183069`，五组 kernel 变化分别为 Prefill +2.84%、Decode +2.63%、D256/G32 +37.70%、D512/G64 +277.78%、D512/G128 +105.37%；AIC scalar 分别下降55.40%、11.96%、53.18%、31.70%、31.04%。Decode只有一次采样，不视作稳定差异。

最终 Prefill 两次为25.306621/25.306270ms，阶段一本轮复测24.515721ms。未经连续行DMA合并的中间版出现过29.786707ms样本，也保留在报告中，未剔除。不能只用scalar降低宣称整体加速。

BSA参考地址预计算没有本版额外的K/V数据GM中转。D512保持原LO模式后，只能容纳1～2行gather，产生大量小chunk DMA与握手；后续须评估联合UB预算/生命周期复用和数据中转隐藏，而非把D128当作唯一开发范围。完整指标、推断边界与复现命令见 `perf/gbsa_stage2/README.md`。阶段二保留在工作区供审查，未覆盖系统custom，未额外提交。


## 14. 逐 block/page 片段立即写回（索引窗口版，历史实现）

### 14.1 范围修正

当前 A5 非量化 blockY 的有效范围为 **1～128**。主算子 `ParseBlockShapeAttr` 和 metadata `CheckGbsaBlockAttrs` 同步收紧；其他架构/量化的原有固定128规则不变。注意不修改 kvBaseTile 的256/384/512范围、不修改 headDim≤512，也不恢复 topK≤256限制。第11～13节中的Y>128测试是旧规格下的历史记录，不代表当前接口仍接受它们。

### 14.2 单层流式搬运

删除“外层固定chunk、内层多块拼接”的组织。当前一次迭代只处理一个合法片段：

```text
len = min(blockY - 当前块内偏移,
          当前AIV半tile的剩余有效token,
          bufferRows,
          pageSize - 页内偏移)      // 最后一项仅PA
```

当前段的K/V读入UB后，立即由MTE3写到压紧GM目标，不等待其他稀疏块填满固定chunk。目标偏移为 `(halfBegin + done) * RoundUp(D,16)`，循环按实际 `len` 推进。

K/V共用源token和PA页表查找，但各自使用pageStride。每段分别读取K/V，MTE2_MTE3事件确保读取完成后写出，再设置MTE3_MTE2，保护UB复用。段号模2选择 `[K0,V0]` / `[K1,V1]`，每次可能写不同长度，写回量始终是实际len而不是buffer容量，避免把上一段残留数据写出。

索引窗口仍独立于数据buffer，容量为L1-N/2。每路AIV的任务局部compact位置单调前进，所以仅在当前索引达到缓存窗口末尾时重新预取。VF仍只做blockId×Y；页定位与DMA下发仍由AIV scalar执行。

### 14.3 UB分配

沿用原LO模式与行数，移除2的幂限制，也不再用索引容量限制数据行数：

```text
maxRows    = floor((gatherBudget - indexBytes) / (4 * RoundUp(D,16) * sizeof(KV)))
bufferRows = min(blockY, maxRows)
```

D128/Q16/高精度/T512：maxRows=123。

- Y16：bufferRows=16，K/V每份4096B，四份共16KiB。
- Y128：bufferRows=123，K/V每份31488B，四份共125952B；块不跨其他边界时按123+5行处理。
- Y127且源块起点在页内127：可拆为1+123+3行；不填满123行再写回。
- D512/Q64仍受原LO驻留预算限制，最多1行；D512/Q128最多2行，未以收紧blockY掩盖这些资源限制。

移除“索引容量限制数据行数”是安全的：一段只访问一个稀疏索引，数据buffer再大也不会要求同时缓存同等数量的索引。

### 14.4 未改变的部分与验收

GM四槽布局、Ready/Free管线、PRE_LAUNCH=2、每路AIV负责半个FA tile，以及QK/PV连续GM→L1加载均保持不变。Ready仍在整路AIV的半tile写完之后发出，**不是每段写完就通知Cube启动QK**。

这项修改减少了固定chunk拼接逻辑，但小Y可能增加MTE3写回及握手次数，不能只凭单层循环推断加速。与第13节包逐例对照，新增Y128/127在123行UB容量、PA跨页和FD分片下的边界配置；测试与性能产物独立归档到 `perf/gbsa_blockwise_gather/`。

## 15. 参考 SMLA 的直接索引 gather（已放弃，历史记录）

参考 `sparse_flash_mla_csa_block_vector.h` 的 `ProcessVec0L`：直接从 GM 取稀疏索引，由 AIV scalar 计算源地址，以 MTE2/MTE3 和 UB 双缓冲搬运。该参考每轮处理两个 sparseBlockSize 单位、积累若干行后写出；GBSA 按当前要求保持一个 block/片段读入后立即写出，不照搬其固定16行聚合及按物理地址排序的双块合并。

### 15.1 索引与搬运

- 删除 `CalcSparseBlockStarts`、索引 UB、窗口参数，以及原索引路径的 MTE2_S / S_V / V_S 同步。
- AIV 的每个半 tile 只在进入时计算 compactBegin/blockY 和余数。外层按 sparse block 推进，调用现有 `GetSparseBlockId<hasResidualBlock>`，显式块和虚拟 residual 共用后续流程。
- 每个外层 block 只读取一次索引；块内根据 UB 容量、PA 页边界拆段，不再为每个小段重新读索引或除以 blockY。AIV/tile 边界切开的同一个块仍会在各自半 tile 内独立读取，未引入跨 tile 缓存。
- K/V 共用逻辑 token 和页表查找，各自保留64位 page stride；任意 D 的补齐、多头行跨度及 DMA stride 溢出的单行回退保持原样。
- 每段执行等待 UB Free、搬入 K/V、MTE2_MTE3 同步、写回 K/V、设置 UB Free，段号选择 ping/pong。目标偏移为 `(halfBegin + done + copied) * RoundUp(D,16)`。

### 15.2 空间与流水

删去索引暂存后：`maxRows = gatherBudget / (4 * RoundUp(D,16) * sizeof(KV))`，`bufferRows = min(blockY, maxRows)`。K/V仍各双缓冲；原 LO 驻留策略和 LO 行块数不变。D128/Q16/高精度T512的最大行数从123变为124，Y16仍使用16行；Y128拆成124+4。D512紧张组合仍可能只能搬1～2行，不能据此宣称解决全部 gather 性能问题。

GM四槽位、两路AIV各负责半tile、PRE_LAUNCH=2、整半tile Ready、PV完成GM读后Free、QK/PV连续加载均不改动。

### 15.3 验证方式

重新编译 kernel+metadata 独立 custom 包，不覆盖用户安装目录。以第14节的已编译包作为本次直接对照，不混用第13节固定chunk数据。验证独立FP32 golden、改动前后输出一致性、PA/非PA、FD、residual0/1、D≤512、高低精度及blockY边界。五组性能配置沿用Prefill、Decode、D256/G32、D512/G64、D512/G128，用同一脚本和 msprof op 重新采集；产物与最终结论保存在 `perf/gbsa_direct_gather/`。

新包已完成132次golden验证（默认66次+单KV头66次），全部通过；默认组与第14节包输出逐元素完全一致。CPU UB/映射、FD/residual helper回归通过。主算子和metadata均接受Y=1/128、拒绝Y=0/-1/129/256；主算子的Y上限在执行阶段host tiling检查，而非必定在GetWorkspaceSize时拒绝。未执行全量化、arch22和LSE输出设备回归。

2026-09-24 回退记录：用户要求舍弃本版并返回方案一。已恢复方案一 kernel、host workspace/tilingdata、DispatchPolicy和直接kernel测试协议；删除当前生效的K/V gather路径。历史包、性能、精度和完整工作区补丁保留于 `perf/gbsa_direct_gather/abandoned-stage2/` 及其父目录。未删除已有性能产物，未改动用户的 `.gitignore`。本次源码回退未重新编译，`build_out` 中上一轮包仍是方案二，不应当作恢复后的方案一安装。
