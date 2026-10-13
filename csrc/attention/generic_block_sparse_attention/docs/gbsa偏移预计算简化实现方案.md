# GBSA A5 非量化：偏移预计算简化实现方案

状态：第11节“UB组表 + AIV配对”和第12节连续源区间合并已提交，当前以历史修复后的 `cdbe117e` 为基线实施第13节 epilogue 组件化。第1～10节为历史GM直写实验，不代表当前实现；历史章节保留当时的提交哈希。产物分别见 `perf/gbsa_scalar_offsets/`、`perf/gbsa_scalar_iterations/` 和 `perf/gbsa_offset_epilogue_refactor/`。

开发基线：`ba61fcc6`，即方案一提交 `43183069` 的父提交。实现时从父提交的原始 QK/PV 加载函数重新修改，不在当前链接式描述协议上继续叠加。独立保留用户已明确的 `1 <= blockY <= 128` 接口约束及相应测试。

## 1. 最新职责划分

AIV 用普通 scalar 代码计算偏移，通过 `GlobalTensor::SetValue` 保存以下信息：

- 每个 L1-N 组的搬运块数。
- 每个搬运块的 K/V 源偏移。
- 每个搬运块的实际 token 长度。

这里的搬运块不是固定等同于一个原始 blockY 块：组首尾先按当前 L1-N 有效范围裁剪，PA 跨页时再拆成两个搬运块，拆出的每块独立计数、保存偏移和长度。

Cube 用 `GlobalTensor::GetValue` 取得块数、偏移和长度，继续调用原有 `copyGmToL1B`。**Cube 不判断 PA 页边界，不解析页表，不从物理地址反推页内位置，也不再次拆块。**

非 PA 输入有序，Cube 保留一次搬两块的逻辑；PA 每条搬运块单独 copy。此区别沿用既有 `PAGED_CACHE_FLAG_` 编译期属性选择，不在 Cube 加运行时 PA/跨页判断。

AIV 不生成完整 copy 参数：不计算配对、ndNum、pair stride、L1 目标偏移。长度属于必须保存的块信息，不再留给 Cube 从 blockY 或 pageSize 推算。

不使用 VF、UB 描述缓存、位域打包、链接式描述、Cube snapshot 数组、新的 Plan/Context/record 结构体、新头文件或通用搬运框架。不 gather K/V 数据，不修改 MMAD、softmax、rescaleO 的计算与布局。

兼容 FD、residualBlockMode=0/1、现有 D<=512、高低精度 softmax 和已有 attention tile/L1 切分，不扩大原计算路径的支持范围。

## 2. 对原框架的最小改动

父提交的两个加载函数是直接参照：QK 的 `SparseKL1TileNLoad` 和 PV 的 `SparseVBaseTileL1FullLoad`。

最初将其中读取 sparseIdx、查询 PA 页表、计算源偏移及本段有效长度的逻辑，移入 `gbsa_layout_utils.hpp` 的普通 `__aicore__` 函数 `PrepareKvOffsets`。当前按第13节将执行逻辑封装为现有框架下的 epilogue block；使用原有局部变量和循环风格，不引入回调或多层 writer 封装。

Cube 原加载函数保留名称、Tensor/Layout 构造、`GetTile`、`copyGmToL1B` 和原有事件位置。改为按块数遍历偏移表，直接使用表中长度构造 tile；目标位置按已搬运长度累加。

通过既有 DispatchPolicy 风格添加编译期开关，仅 A5 非量化实例启用预计算路径。共享头文件中的全量化及其他架构继续保留父提交原路径。PA、residual 沿用已有模板参数，不增加运行时泛化标志或新 tilingkey。

## 3. L1-N 分组与 AIV 分工

以 QK 的一个 L1-N 块为一组：

```cpp
N = mm1L1TileN;
groupNum = CeilDiv(tileTokens, N);
groupBegin = groupIdx * N;
groupTokens = min(N, tileTokens - groupBegin);
```

每组只描述自己需要的 token。同一原始稀疏块跨两个组时，分别生成各组对应的部分，不跨组配对。

两路 AIV 交替处理完整组：

```cpp
for (uint32_t groupIdx = GetSubBlockIdx(); groupIdx < groupNum; groupIdx += 2) {
    // 计算并写出本组搬运块的偏移、长度，最后写入块数。
}
```

T=512、N=256 时，两路 AIV 各处理一组。T<=N 时只有 AIV0 写表，AIV1 仍参与 Ready/Free。为保持实现简单，本版接受单组时的负载不均，不增加半组协议；需单独测量其性能影响。

## 4. GM 格式：块数 + 偏移/长度

使用一个 `GlobalTensor<uint64_t>`，不定义结构体。每组头部独占 64B，仅第一个元素有效，其余元素用于对齐，不读取、不要求清零：

```cpp
groupInfo.SetValue(0, blockCount);
// [1..7] 保留。
```

每个搬运块固定三个 uint64 元素：

```cpp
base = 8 + i * 3;
groupInfo.SetValue(base,     kOffset);
groupInfo.SetValue(base + 1, vOffset);
groupInfo.SetValue(base + 2, rows);
```

统一字段排列使 QK 固定读取 K 偏移、PV 固定读取 V 偏移，长度读取方式完全相同，不需要根据 PA 切换解析格式。非 PA 的 K/V 相对偏移相同，两个字段保存同值；PA 的 K/V page stride 可能不同，分别计算。

偏移使用 uint64 元素单位，相对于原加载函数已确定的 batch/head（非 PA）或 cache/head（PA）基址，不包含 QK 的 embedInnerOffset。rows 为该搬运块实际 token 数，不是字节数；其值大于 0。

blockCount 是**拆分后的搬运块数**，不是原始 blockY 块数，也不是配对后的 copy 调用次数。每组必须满足：

```cpp
sum(rows[i], i = 0 .. blockCount - 1) == groupTokens;
```

PA 不预留“可能存在的第二页”字段；跨页就追加一条正常记录。Cube 只读实际 blockCount 条记录，不依赖未初始化字段。

## 5. AIV 计算与拆分

### 5.1 沿用有效范围和 FD/residual 协议

保持父提交 `CalcSparseKvSize`、FD 分片和 `gTaskSparseBlockIdx` 的 task 局部基址逻辑。只处理已算出的有效 tileTokens，不重新扫描整行或重复计算 causal 可见长度。

```cpp
compactBegin = tileIdx * kvBaseTile + groupBegin;
firstBlock = compactBegin / blockY;
firstInner = compactBegin % blockY;
sparseBlockCount = groupTokens == 0 ? 0 : CeilDiv(firstInner + groupTokens, blockY);
```

sparseBlockCount 是 AIV 遍历的原始逻辑块数，与输出 blockCount 区分命名。通过原 `GetSparseBlockId<hasResidualBlock>` 解析当前 FD 子列表及虚拟 residual 块；仅含虚拟块、blockId=0 时也不读取无效 sparseIdx。

### 5.2 非 PA

沿原逻辑依次取得 blockId，首块从 firstInner 开始，其余从块首开始。每块长度按组内剩余有效 token 裁剪：

```cpp
blockCount = 0;
dealtLenAccum = 0;
for (uint32_t i = 0; i < sparseBlockCount; ++i) {
    blockId = GetSparseBlockId(..., firstBlock + i, ...);
    inner = i == 0 ? firstInner : 0;
    rows = min(blockY - inner, groupTokens - dealtLenAccum);
    sourceToken = uint64_t(blockId) * blockY + inner;
    sourceOffset = sourceToken * tokenStride;
    // SetValue 保存 {sourceOffset, sourceOffset, rows}。
    ++blockCount;
    dealtLenAccum += rows;
}
groupInfo.SetValue(0, blockCount);
```

AIV 不做成对判断，不计算 pair stride，不排序。输入有效 idx 按序的契约保持不变。

### 5.3 PA：跨页拆块全部在 AIV 完成

每个原始块先按组边界裁剪得到 rows，再按照父提交的页边界逻辑计算本段：

```cpp
sourceToken = uint64_t(blockId) * blockY + inner;
pageInner = sourceToken % pageSize;
firstRows = min(rows, pageSize - pageInner);
page0 = blockTable.GetValue(sourceToken / pageSize);
kOffset = uint64_t(page0) * kPageStride + pageInner * tokenStride;
vOffset = uint64_t(page0) * vPageStride + pageInner * tokenStride;
// 追加 {kOffset, vOffset, firstRows}，blockCount++。
if (firstRows < rows) {
    page1 = blockTable.GetValue(sourceToken / pageSize + 1);
    // 追加 {uint64_t(page1)*kPageStride,
    //       uint64_t(page1)*vPageStride, rows-firstRows}，blockCount++。
}
```

当前 pageSize=128、blockY<=128，一个原始块至多拆成两段。这只是 AIV 拆分和容量推导的当前约束，不在 Cube 保留任何“两页”逻辑。若将来放开更小 pageSize 或更大 blockY，再扩展 AIV 拆分及容量，Cube 遍历方式不变。

只在当前有效 rows 确实跨页时读取下一页，避免 residual/序列尾部越界查询。不同物理页不要求连续，K/V 页跨度分别处理。

例如 blockY=127、sourceToken=127、需要127行，逻辑页0/1映射到物理页7/3，D=128、单KV头、无页padding：

```cpp
// tokenStride=128，kPageStride=vPageStride=16384。
blockCount = 2;
// {kOffset, vOffset, rows}
{130944, 130944,   1}  // 7*16384 + 127*128
{ 49152,  49152, 126}  // 3*16384
```

Cube 依次搬1行、126行，目标连续推进127行，不关心两条记录原本来自同一 blockY。若该组只需首1行，则 AIV 只生成第一条，blockCount=1。

## 6. Cube 直接消费偏移和长度

### 6.1 普通单块路径

```cpp
blockCount = groupInfo.GetValue(0);
dealtLenAccum = 0;
for (uint32_t i = 0; i < blockCount; ++i) {
    sourceOffset = groupInfo.GetValue(8 + i * 3 + offsetField);
    rows = groupInfo.GetValue(8 + i * 3 + 2);
    // 用 rows 构造原来的 L1/GM tile，调用 copyGmToL1B。
    dealtLenAccum += rows;
}
```

offsetField 在 QK 中固定为0，在 PV 中固定为1，不是运行时新增判断。QK 给 sourceOffset 加 embedInnerOffset，构造目标 `MakeCoord(0, dealtLenAccum)`；PV 按组遍历，构造目标 `MakeCoord(groupBegin + dealtLenAccum, 0)`。

这里不需要 blockY 取余、pageSize、pageTable、pageInner 或剩余长度 min。rows 已在 AIV 按所有边界确定。QK 不同 D 子块及 PV 共用同一份不可变偏移表。

### 6.2 非 PA 一次搬两块

仅在既有非 PA 模板路径内加入配对：读取相邻两条偏移及长度，两个长度都等于 blockY 时尝试一次搬运两块；首尾部分块和落单块单独搬运。不跨组配对，不越界读取 i+1。

```cpp
sourceStride = offset1 - offset0;
// 满足已有 copy 接口的 stride 对齐及可表示范围时：
copyGmToL1B(l1Tile, gmTile, 2, sourceStride, dstStride);
// 消耗两条记录，dealtLenAccum += rows0 + rows1。
```

有序输入无需交换两块，但有序不等于 stride 必然满足 DMA 的对齐/范围限制，不能删去必要的接口约束。不满足时退回单块 copy。

ndNum、sourceStride、dstStride 由 Cube 现场生成，不写入 GM。QK/PV 各自沿用原 L1 布局计算目标跨度，不能共用 QK 的目标跨度代替 PV。PA 编译期禁用成对分支，直接走上述单块遍历，不需要 PA 页判断。

## 7. 流水、槽位与跨核可见性

保留父提交 PRE_LAUNCH=2：循环前 Prepare(0)；迭代 i 时 AIV 先 Prepare(i+1)，再执行原有 Softmax(i)、RescaleO(i-2)，Cube 执行原有 QK(i)、PV(i-2)。沿用原循环边界，只对实际存在的 tile 预计算。

每个物理 AIC 预留4个 GM 偏移表槽位，slot=tileIdx%4。迭代 i=2 附近同时保留 tile0 供 PV、tile1/2 及提前准备的 tile3，不按 task/topK 分配全量描述。

SetValue/GetValue 是 scalar GM 访问，不能直接套用 UB→GM 的 MTE3 Ready 协议：

1. AIV 等待对应槽位 Free，对其负责的组写完偏移和长度，最后写 blockCount。实际空组计数写0；未消费的预留组不必清零。
2. 按本地 A5 CANN 接口完成所写 cache line 的写回/失效和完成排序，再通过 PIPE_S 发 Ready。
3. Cube 在 PIPE_S 等待两路 AIV；没有分组任务的 AIV 也必须发 Ready。
4. 首次 QK 消费某组时，先失效 header 缓存、读取 blockCount，再失效有效记录范围，随后 GetValue。后续 D 子块和 PV 复用不可变表，不重复清理整份表。
5. PV 完成该 tile 的全部 scalar 表读取后，通过 PIPE_S 向两路 AIV 发 Free。

使用现有 CrossCore 接口及未占用 flag id；kernel 入口初始化4槽 Free，退出回收，task/FD 切换沿用相同生命周期。Free 保护的是偏移表 scalar 读取，不必等待读取原始 K/V 的 MTE2 搬运完成。

没有描述 UB，不增加 S_MTE3/MTE3_S 事件。cache 维护目标、完成顺序及生成指令必须用本地接口和小规模跨核回环验证，不能因简化方案省略可见性保证，也不新增自制 cache 原语。

每组 header 独占64B，记录区从下一个64B边界开始，组/核/槽位均64B对齐。两路 AIV 只写各自完整组，不共享计数或末尾 cache line。

## 8. Workspace 与 UB

容量按实际可能输出的搬运块数估计，不按成对 copy 次数估计：

```cpp
maxSparseBlocks = CeilDiv(N + blockY - 1, blockY);
maxCopyBlocks = isPaged ? min(N, 2 * maxSparseBlocks) : maxSparseBlocks;
groupBytes = AlignUp(64 + uint64_t(maxCopyBlocks) * 3 * sizeof(uint64_t), 64);
groupsPerTile = CeilDiv(kvBaseTile, N);
tileBytes = groupsPerTile * groupBytes;
offsetInfoBytes = uint64_t(physicalAicNum) * 4 * tileBytes;
```

maxSparseBlocks 覆盖组起点落在 blockY 内部的最坏情况；PA 每个原始块最多两段，且每段至少一个 token，故以 N 再限制上界。容量允许保守，不要求对每个 Y 的页对齐关系作复杂优化。host 和 kernel 使用相同公式及单位，必要乘加使用64位。

physicalAicNum 取主 kernel 实际配置的物理 AIC block 数，不使用两倍 AIV 数。偏移表放在父提交已有 pipeline/LO 工作区之后、FD partial 之前，保留对齐和溢出检查。只新增以下 tilingdata 字段，并同步 host/kernel mirror 和直接 kernel 测试：

- kvOffsetInfoOffset：workspace 字节偏移，uint64。
- kvOffsetTileWords：单槽 uint64 元素数，uint32，转换前检查范围。

组大小由已有 N、blockY、PA 属性和格式常量推导，不增加 Plan 或多余字段。它与 topK 无关，但依赖 blockY 和 PA；不能沿用旧方案“容量与 blockY 无关”的说明。

例如 T512/N256/Y16：maxSparseBlocks=17。非 PA 每组512B，每 tile1024B，四槽每核4096B，28核共112KiB；PA 按保守34段容量每组896B，每 tile1792B，四槽每核7168B，28核共196KiB。实际 Y16 对齐时不会跨128页，仍仅写实际生成的块数和记录，不填满容量。

本方案不新增 UB 空间，不修改父提交 UBufTileHelper、S/P/LO/GO 划分及 D512 LO 中转策略；不限于 D128。标量局部变量仍有寄存器或栈成本，不能承诺零开销。

## 9. 文件级实施顺序

1. 从 ba61fcc6 的相关 kernel/host 实现重新修改，保留方案一及实验产物、独立 Y 范围修正和用户无关修改，不 reset --hard 清理仓库。
2. gbsa_layout_utils.hpp：初版放置 PrepareKvOffsets，当前仅保留公共几何辅助函数和格式常量；偏移执行逻辑已迁入第13节的 epilogue block。
3. tiling.cpp/.h、kernel_common.hpp：两个字段及偏移表 workspace，不改 tile 选择、LO 和 FD 分核算法。
4. generic_block_sparse_attention_kernel_arch35.h：绑定偏移表 GM，接入 Prepare/Ready/Free，透传表基址，不改 CalcUBufTileInfo。
5. QK/PV 原加载头文件：按块数 GetValue 偏移及长度，直接 copy；非 PA 保留双块，PA 单块，不移动原 MMAD/事件位置。
6. DispatchPolicy/interface：仅非量化启用，沿用原模板参数传递风格。
7. 测试同步新表格式、workspace 协议，删除旧 control/链接协议依赖，保留原精度 golden。

若性能不足，先给出测量与原因，不自行切回 VF、UB 描述缓存或复杂协议。

## 10. 验证与验收

正确性覆盖：

- CPU 重放 Y1～128、T128/256/384/512、L1N128/256：逐条比较偏移/长度，验证拆分后计数、容量上界、长度和等于 groupTokens、无目标空洞或重叠。
- 非 PA 完整双块、首尾部分块、奇数块、stride 对齐/范围、多 KV 头、64位地址；验证实际成对 copy 的目标间距。
- PA 的 Y127 跨128页、随机/倒序物理页、不同 K/V page stride、末页不足；跨页增加计数，所有实际 copy 均为单块，Cube 不再查页表或计算页内位置。
- FD 非零 rawBegin、residual 两种模式、纯虚拟块且 ID=0、完整尾块显式给出、空任务、组/tile 切到块内部。
- D1/31/32/33/64/65/127/128/129/255/256/257/384/511/512，覆盖 QK 的 D 子块复用和 LO GM 路径。
- 超过4个 tile、多个 task、单组、短尾组重复运行，检查旧计数/偏移/长度及跨核死锁。
- 独立 FP32 golden 上板校验，并与父提交相同输入对比。仅改搬运应保持数值结果，不通过放宽容差掩盖差异。
- 先编译 kernel+metadata 到独立 custom 包，不覆盖用户安装路径；全量化/arch22 至少编译回归。

性能以父提交为主基线，43183069 仅作额外参考。沿用 Prefill/Decode、D256/G32、D512/G64/G128，补充 PA、不规则 Y 和单组场景。

用相同数据、设备、脚本及 msprof op 参数，记录总时长、AIC/AIV scalar、MTE2 和 cache/同步现象，重复测量重要升降。预期 Cube 移除重复的稀疏索引解析、页表查询、源偏移和长度计算，仍保留表读取、非 PA 配对 stride 和目标位置推进；不承诺消除全部 scalar。

不能用 AIC scalar 下降代替总耗时改善，也不能把 AIV scalar 阶段称为 VF 加速。性能是否改善以本轮实际报告为准，不能把设计目标当作已达成的收益。

## 11. 最新修订：UB组表，双块配对移入AIV

GM逐字段SetValue版精度通过，但prefill首轮从26.03ms回退到31.06ms，Cube scalar同样上升。按用户追加授权，改用UB临时表和MTE3批量写出，并将双块配对、长度匹配及source stride计算移至AIV。保留普通for循环、原QK/PV加载函数和现有copyGmToL1B调用，不使用VF、位域、链接、snapshot结构体或新头文件。

### 11.1 表格式和count口径

每组仍有64B头部，首元素是本组copy记录数。后续为连续uint64字段：

- 非PA：`[sourceOffset, rows, ndNum, sourceStride]`，每条32B，K/V共用相对偏移。
- PA：`[kOffset, vOffset, rows]`，每条24B，固定单块搬运，不保存ndNum/stride。

跨PA页仍由AIV拆成独立记录；Cube不查页表、不判断页边界。非PA两个完整blockY长度相同且stride可编码时，AIV合成一条ndNum=2的记录。因此count现在表示copy调用数，`sum(rows * ndNum)`等于groupTokens；不再等于拆分后、配对前的块数。

AIV用previousLen/previousOffset普通局部变量完成配对，不排序。Cube只读取offset、rows、ndNum、sourceStride，构造原L1/GM tile并调用copy，累加目标token位置，不再进行长度匹配、相邻偏移求差和stride可编码检查。

### 11.2 UB及同步

每个AIV仅使用一个L1-N组大小的UB临时表，放在现有未使用的maskStartOffset，不按四个GM槽位扩大UB。

写UB后用S_MTE3事件保证可见，DataCopy批量写入当前GM组；用MTE3_S事件等待该临时UB可复用。Ready在PIPE_MTE3发布，Cube仍在PIPE_S等待并刷新读取缓存；四槽Free/Ready生命周期、PRE_LAUNCH=2和提前一tile准备保持不变。AIV不再逐cache line执行GM scalar写回。

容量测试穷举D1～512、Q tile16/32/64/128、高低精度、Y1～128、PA/非PA，检查临时表与248KiB可用UB的边界。D512/Q64/Y1非PA最紧档位需要4160B表空间，现有mask起点249600B，末端253760B，小于253952B；该档位比名义mask预留多64B，但后方没有存活buffer，仍不挤占LO/GO。D512/Q128的LO GM路径使用max(maskBytes, tableBytes)作为预留，穷举结果LO分块行数与父版本相同。

### 11.3 Workspace

分配使用配对前的保守块数上界，避免依赖实际idx和配对成功率：

```cpp
maxSparseBlocks = CeilDiv(N + blockY - 1, blockY);
maxCopyBlocks = isPaged ? min(N, 2 * maxSparseBlocks) : maxSparseBlocks;
recordWords = isPaged ? 3 : 4;
groupBytes = AlignUp(64 + uint64_t(maxCopyBlocks) * recordWords * 8, 64);
offsetInfoBytes = physicalAicNum * 4 * CeilDiv(kvBaseTile, N) * groupBytes;
```

基线T512/N256/Y16/28核，非PA新增140KiB，PA新增196KiB。完整组非PA实际产生8条pair记录，仅搬64+8*32=320B；不能把保守分配容量等同于实际MTE3流量。

D256/G32/T512/N128非PA新增168KiB；D512/G64或G128/T128/N128非PA新增42KiB。上述topK820测试配置已启用host FD workspace预留，父版原有对齐吸收起始padding，故总workspace增量等于表空间。其他未预留FD的配置还需加起始512B对齐的padding。

### 11.4 验收口径

分别保存parent、candidate（GM直写）、ub_candidate（本节版本）的独立custom包及日志，不覆盖用户安装目录。复用独立golden和父版本逐元素对比；性能观察kernel总耗时、Cube scalar、Cube MTE2和包含metadata的同步端到端时延。

最终UB版132次上板golden通过，其中66例与父版逐元素一致。5卡两轮复测中，prefill kernel下降8.25%、Cube scalar下降11.96%、Cube MTE2下降28.96%、含metadata端到端下降7.83%；五场景kernel/Cube scalar均下降，但decode和D256的含metadata端到端分别上升6.10%和4.27%，不可宣称全面端到端加速。完整报告见 `perf/gbsa_scalar_offsets/README.md`。

## 12. Scalar优先的五轮迭代

以 `d3acecf9` 为基线，试验：连续源区间合并、描述cache预取、L1地址线性化、Cube消费循环两次展开、8B紧凑计数头。每个候选均通过66次上板独立golden及父版逐元素对比。

当前只保留连续源区间合并：非PA中，上一记录尚未配对且满足
`sourceOffset == previousOffset + previousLen * tokenStride` 时，扩展该记录的rows；
否则继续原来的完整blockY双块配对/单块追加。`previousLen==0`仍表示上一条已经配对，不能继续扩展。
只在同一个L1-N组内合并，不跨group、attention tile、FD任务或PA页。

例如Y16、D128、单KV头，idx=[0,1,2,4,6]：前三块合为
`{offset=0, rows=48, ndNum=1, stride=0}`；后两块仍配为
`{offset=8192, rows=16, ndNum=2, stride=4096}`。
五个稀疏块由两条copy记录完成，token顺序和QK/PV计算顺序均不变。
多KV头时“连续”按tokenStride判断，不要求不同token的同一head字节紧挨。

QK/PV加载代码、prelaunch、Ready/Free同步、UB划分及workspace容量均不变。
新增分支只在AIV组表阶段；Cube因记录数减少而减少GetValue、缓存维护和copy参数装配。
收益依赖稀疏idx中连续区间的数量，不能保证对完全离散的列表同样有效。

2卡单轮筛选中，prefill Cube scalar从22.727ms降至18.761ms（-17.45%），
kernel从24.054ms降至20.154ms（-16.21%），端到端从25.207ms降至21.101ms（-16.29%）。
其余实验未显示出值得保留的综合性能收益，已撤回；8B头部节省空间但未优于重点场景的scalar表现。
以上为筛选数据。第5轮后段遇到其他用户任务，干扰样本已排除并补跑。
最终14份交错复测报告确认：prefill kernel下降16.15%、Cube scalar下降17.45%、
MTE2下降2.34%、端到端下降16.26%；decode和D512/G64端到端分别上升0.72%和1.70%，
不能宣称全面端到端加速。选定版另补66次单KV头golden通过，共132次精度调用。
完整五轮数据、环境限制和复现方法见 `perf/gbsa_scalar_iterations/README.md`。

## 13. 偏移预计算封装为 epilogue block

以 `cdbe117e` 为重构基线，保留第11/12节的 UB 组表、PA 拆页、双块配对和连续区间合并算法。
新增 `attn_infra/epilogue/block/gbsa_block_epilogue_prepare_kv_offsets_arch35.hpp`，
使用 `BlockEpilogue<EpilogueAtlasA5PrepareKvOffsets<isPaged, hasResidualBlock>>` 特化，
在 kernel interface 中组装并作为模板组件传给 `GbsaRegularKernelArch35`，仅用于 A5 非量化。

构造时从 `Arch::Resource` 的 `maskStartOffset` 绑定原 UB 暂存区，保存 tile、L1N、blockY、pageSize
及 token/K/V page stride，并计算固定 groupWords；没有新增 Plan、Context 或描述结构体。
`operator()` 只接收本次描述输出地址、已按 FD 分片重定位的 sparseIdx、batch 页表、tileIdx、
实际 tileTokens、局部显式 count 和 residualId。原 block/group/copy 循环和本地 S_MTE3/MTE3_S 事件保持不变。

kernel 仍负责四槽 GM 地址、tile0 预热、提前一 tile 调用和跨核 Ready/Free；
QK 等待两个 AIV 的描述就绪，PV 消费后才释放槽位。QK/PV 消费接口及复制实现没有改动。
`CalcKvOffsetGroupWords`、`GetSparseBlockId` 等公共辅助函数仍在 `gbsa_layout_utils.hpp`，
新组件显式包含该头文件，不依赖外部 include 顺序。

UB/GM 容量、tiling 数据及 key、A2/全量化路径均不改变。D512 LO 中转沿用原 UB 分配。
重构验证先进行独立 golden 及同输入逐元素输出对比（基线在2卡、候选按用户要求在6卡），
性能比较则两版均在6卡，以 AB/BA 顺序记录 kernel、Cube scalar/MTE2、AIV scalar 和同步端到端时延。
6卡存在其他并行任务，采样保留占卡快照；不能将共享负载下的端到端差异归因于此次重构。
产物保存至 `perf/gbsa_offset_epilogue_refactor`，不覆盖已安装的 custom 包。

验证结果：完整编译通过，候选132次输出O的golden检查通过，其中66次与基线逐元素一致；未验证LSE。
四个性能场景的kernel平均耗时变化为prefill +1.33%、decode +2.21%、D256 -0.28%、D512 +1.31%，
这些是共享6卡下的观测差值，不作为收益或回退结论。D512第二轮基线Pipe计数异常，原始数据保留；
详细限制和逐核计数审查见上述perf目录的README及pipe_counter_audit.md。
