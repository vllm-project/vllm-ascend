# Arch35 非量化 blockY 泛化

## 范围和输入约定

- 仅 Arch35、FP16/BF16、quantType=0 放开 `blockShape=[1,Y]`，`1 <= Y <= 256`，无需 Y 对齐。
- attention KV tile 固定 128，PA page size 固定 128；现有 headDim 约束不变。
- Y=128 保留原普通/FlashDecode 调度条件；其他 Y 使用普通任务分核。所有 Y 共用同一套统计、搬运和 QK/PV 计算逻辑。
- 稀疏索引按逻辑位置递增，有效索引占据 `sparseBlockCount` 指定的前缀；前缀内无 -1。topK 与 Y 由调用者配套设置。

## 实现

`gbsa_layout_utils.hpp` 中的 `CalcSparseKvSize` 根据当前 Q 的 causal 可见上界，累计有效 token 数，写入主 kernel 的 `validKvTokens` 变量。kernel 直接按 128 计算循环次数和当前 tile 长度。不使用统计结构体或定长 tile 数组，因此支持 tile 数大于 topK（最大 256 × 256 / 128 = 512）。

主 kernel 不使用 `generalizedY` 区分计算路径。普通任务选择 `[0, validTopK)`，FD 选择调度指定并裁剪后的 `[rawBegin, rawEnd)`；稀疏列表指针统一重定位到 `sparseIdxBase + rawBegin`，统计数量为 `rawEnd - rawBegin`。之后 QK/PV、softmax 首块判断及 buffer 轮转均使用从 0 开始的分片局部 tile 编号。FD 仅保留调度、partial 输出及 combine 分支；空分片写 neutral partial。

Dense GM 起点仅定位到 batch/head；PA 定位到 cache/head。页表行、page size 和 K/V 各自的物理页 stride 通过普通参数传给矩阵类，没有搬运上下文结构体。

接口通过 `MmadAtlasA5BsaQK<isPaged>` / `MmadAtlasA5BsaPV<isPaged>` 将 PA 配置放入 DispatchPolicy。矩阵类从 `DispatchPolicy::PAGED_CACHE_FLAG` 获取编译期常量，原有 L1 搬运循环使用 `if constexpr (PAGED_CACHE_FLAG)`，不再以 blockSize 是否为零判断 PA。`operator()` 和搬运函数不再单独声明 PA 模板参数，主 kernel 保持普通函数调用。`strideKVRow` 作为 `uint64_t tokenStride` 普通参数传入，直接计算元素偏移 `pageOffset + sourceToken * tokenStride`；不再调用 layout/coord 求偏移。调用方须保证 GM tensor 已定位且自身坐标为零。构造搬运 tensor 时仍保留 layout 元数据。

FP8 复用调用在主 kernel 中已完成物理页定位，因此在接口层使用 PA 配置为 `<false>` 的 DispatchPolicy，搬运时仅传入 token stride，避免二次查页；不改变其原有页内搬运方式。

QK 的 `SparseKL1TileNLoad` 和 PV 的 `SparseVBaseTileL1FullLoad` 均使用外层逐稀疏块、内层按 PA 页边界分段的 `for` 循环；同一稀疏块的索引只读取一次，跨页时不再重新计算 gathered/Y 下标。没有另起 loader 分支。主 kernel 给 QK/PV 的实际 tile 长度已由统计结果裁剪，无需在搬运中重复检查有效块和序列上界。

每段直接按实际大小调用现有 `copyGmToL1B`，在同一 L1 tile 中接续写入；未增加自定义短段搬运或对齐补拷贝。完整 attention tile 的有效长度为 128，最后一个 tile 可不足 128，按实际长度进入计算及 softmax。

按输入契约默认有效稀疏索引和 PA 页表合法。`CalcSparseKvSize` 只统计有效 token 数，不返回校验结果；PA 不再执行第二遍逐页检查。空 FD 分片仍写 neutral partial。

Host tiling 和 metadata 参数检查同步放开上述范围；PA 容量校验改用 page size，不再使用 Y。Arch22 和 FP8 的 Y 限制仍为 128。

## 验证记录（2026-09-17）

移除 PA 逐页预校验和统计函数的非法输入返回分支，将 QK 与 PV 搬运均改为逐稀疏块、块内跨页分段的双 `for` 循环，统计函数仅返回有效 token 数。CANN 9.2 下 10 个 device 模板编译通过；CPU 有效长度测试 4,147,200 例和 FD 子范围 92,205 例通过；Ascend950PR 上 48 组普通 Dense/PA 数值回归及 4 组 FD causal 裁剪/空分片混合回归通过。未做性能对比；调用方须保证索引和页表符合输入契约。

## 历史验证记录（2026-09-15）

将 PA 模板配置迁入 DispatchPolicy 后，通过 10 个模板编译及 52 组非量化上卡回归（48 组普通任务、4 组 FD causal 裁剪/空分片混合合并）。本轮只调整模板参数归属，未修改搬运计算公式，未做性能对比。

将源偏移改为直接乘加、将 PA 判断改为 `isPaged` 编译期分支后，再次通过 68 组非量化上卡数值回归及 10 个模板编译（含 FP8）。本轮未进行性能对比。

移除新增头文件及统计/搬运结构体、恢复原有稀疏搬运循环后，已重新通过以下 CPU 长度/页检查、FD 分片测试、68 组上卡测试及 10 个模板编译。Host 和 metadata 测试记录来自此前适配阶段，本轮未修改其实现。

- CPU 长度/页检查测试：4,147,200 例通过，遍历全部 Y=1..256，检查有效块数、token 数、因果裁剪、尾块、有效及非法物理页、512 tile 容量及乘法溢出边界。实际 L1 切段与搬运由上卡测试覆盖。
- FD 统一路径新增 92,205 例 CPU 分片测试通过，覆盖非零列表起点、分片终点、空范围及 causal 裁剪。
- Host tiling：28 个测试通过，其中泛化用例内部遍历 11 个新 Y × FP16/BF16 × Dense/PA；包含越界 Y、Arch22/FP8 拒绝测试。
- Metadata scheduler：16 个测试通过，覆盖新 Y 禁用 FD。
- CANN 9.1、Ascend950PR 实机：统一路径修改后重新运行 68 组 O/LSE 对照测试，全部通过。48 组为 Y={1,3,16,17,63,64,127,128,129,192,255,256} × FP16/BF16 × Dense/PA；12 组为 Y=128 的八分片 FD/合并，分别覆盖普通分片、causal 尾块与空分片混合、全部分片为空；8 组为 Y=255/256、topK=2。
- 上卡用例使用 D=128，包含多 batch/head、GQA、Q padding、空选择、因果尾块、稀疏跳块、非连续物理页、页内起点及 K/V 不同 page stride。
- 统一路径修改后 10 个 kernel 模板重新编译通过：Dense/PA × FP16/BF16 × LSE 开关，以及两个 FP8 输出类型。FP8 仅做编译回归，未做数值上卡测试。
- Metadata ACLNN 和 AICPU 源文件语法检查通过；`git diff --check` 通过。

上卡脚本直接编译并加载生产 kernel，手工构造 tiling/metadata，不安装系统 OPP；不等同于完整 ACLNN/Torch API 端到端验证。本次未做性能基准对比，不能据此断言 Y=128 无性能回退。

## 复现核心测试

在仓库根目录执行 CPU 几何测试：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -Werror attention/generic_block_sparse_attention/tests/ut/op_kernel/test_sparse_kv_geometry.cpp -o /tmp/gbsa_sparse_geometry_test
/tmp/gbsa_sparse_geometry_test
```

在有空闲 Ascend950 设备的环境中执行设备测试（按实际空闲设备修改 `--device`）：

```bash
source /usr/local/Ascend/cann/set_env.sh
python3 attention/generic_block_sparse_attention/tests/ut/op_kernel/test_blocky_device.py --device 2
python3 attention/generic_block_sparse_attention/tests/ut/op_kernel/test_blocky_device.py --device 2 --ys 128 --fd
python3 attention/generic_block_sparse_attention/tests/ut/op_kernel/test_blocky_device.py --device 2 --ys 128 --fd --fd-case clipped
python3 attention/generic_block_sparse_attention/tests/ut/op_kernel/test_blocky_device.py --device 2 --ys 128 --fd --fd-case empty
python3 attention/generic_block_sparse_attention/tests/ut/op_kernel/test_blocky_device.py --device 2 --ys 255,256 --topk 2
python3 attention/generic_block_sparse_attention/tests/ut/op_kernel/test_blocky_device.py --build-only
```

Host 测试须使用仓库 `test_op_host_main.cpp` 初始化 `OpImplSpaceRegistryV2`，加载被测 tiling 动态库；直接链接普通 `gtest_main` 不足以启动该测试。与当前 CANN 库链接时使用 `_GLIBCXX_USE_CXX11_ABI=0`。
