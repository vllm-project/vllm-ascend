# A2/A3 cmp_ratio=1/2 适配说明

## 1. 范围与结果

本次为前向 SparseFlashMla 的 CSA 增加压缩倍率 1 和 2，配套修改 SparseFlashMlaMetadata。A2/A3 HCA 保持仅支持 128 的原有逻辑。950 的倍率范围不变；不涉及梯度算子、压缩算子或上游索引器的实现。

| A2/A3 模式 | 修改前 | 修改后 |
| --- | --- | --- |
| SWA，无 cmp KV | 1 | 0 |
| CSA，有 cmp KV 和 cmp 索引 | 4 | 1、2、4 |
| HCA，有 cmp KV、无 cmp 索引 | 128 | 128 |

其余约束保持现有实现，例如 cmp causal mask 为 3、CSA TopK 容量为 512/1024、ori 窗口为左 127/右 0。ratio=1/2 不意味着扩大其他输入规格。

## 2. 倍率在代码中的传递

```text
调用方 cmp_ratio=1/2
  Metadata Host: IsCmpRatioSupportSmla → ParamsCheck
    AICPU: cmpRatio_ → GetRevertS2Size → CalcCmpBlockRange
      → block/cost → 分核 → metadata
  主算子 Host: CheckSingleParaCmpRatio → cmpParams.cmpRatio
    arch22 Kernel: constInfo.cmpRatio
      → 压缩有效范围 → gather / 逐行 mask → attention
```

倍率是运行时 tiling 参数，不是 tiling key 的模板维度。两次调用必须使用同样的倍率、有效长度、residual，改变这些参数后应重新生成 metadata。

## 3. 适配点与实现决策

| 层次 | 文件 / 符号 | 本次处理 |
| --- | --- | --- |
| 主算子 Host | [sparse_flash_mla_tiling.cpp](../op_host/sparse_flash_mla_tiling.cpp)，`CheckSingleParaCmpRatio` | CSA 增加 1/2；保持 HCA=128，SWA 改用 0，更新报错 |
| Metadata Host | [metadata_check.h](../../sparse_flash_mla_metadata/op_host/sparse_flash_mla_metadata_check.h)，`IsCmpRatioSupportSmla` | 与主算子允许集合一致，更新错误信息 |
| Metadata AICPU | [metadata_aicpu.cpp](../../sparse_flash_mla_metadata/op_kernel_aicpu/sparse_flash_mla_metadata_aicpu.cpp) | 保留已有运行时乘除公式、residual 范围校验及 block/cost 逻辑 |
| CSA Kernel | [csa_kernel.h](../op_kernel/arch22/sparse_flash_mla_csa_kernel.h) | 已通过 cmpRatio 计算长度和 cmpS2IdLimit，无需新增 ratio 模板 |
| CSA gather | [csa_block_vector.h](../op_kernel/arch22/sparse_flash_mla_csa_block_vector.h) | 沿用 cmpS2IdLimit 检查索引，重点验证 causal 边界 |
| HCA Kernel / mask | [swa_kernel.h](../op_kernel/arch22/sparse_flash_mla_swa_kernel.h) | 保持原有实现，不增加倍率 1/2 支持 |
| tiling / 内存 | Host SplitBalanced、DoOpTiling | 保持 S2=512 和原缓冲区配置；增加序列长度会增加循环次数，不直接扩大片上基本块 |
| 接口 / 绑定 | 现有整数属性透传 | 不修改 ABI、输出 shape、dtype 或 tiling key |

Kernel 与 AICPU 的相关公式已经参数化，所以本次不为“有 Kernel 变更”而改写等价计算。真正的生产代码变更是两处 Host 倍率白名单。

## 4. 长度、residual 与 causal 边界

令 Lc 为压缩后有效长度，r 为倍率，residual 为余数：

```text
L = Lc * r + residual
p = L - Lq + query_index
visible_cmp = clamp((p + 1) / r, 0, Lc)
```

非负坐标下除法向下取整，负范围由具体分支裁剪。r=2 时 residual 只能为 0 或 1。cmp_mask_mode=3 且 r!=1 时 residual 必须同时传给 Metadata 和主算子，即使余数为 0。

例：原始各 batch 长度 [3,3]，压缩有效长度为 [1,1]，residual 为 [1,1]，压缩 TND 前缀和为 [0,1,2]。不能将 ori 前缀和 [0,3,6] 逐项除以 2 得到 [0,1,3] 后当作压缩前缀和。

CSA 调用方必须实际生成对应倍率的 KV、索引和分页表。倍率 1 的压缩长度等于原长度且不需要 residual；倍率 2 的 residual 取 0 或 1。相同原始长度下，从倍率 4 改为 1/2 会增加压缩 KV 条目数量，应重新预算输入 cache；不能只改变算子属性。A2/A3 的 HCA 仍传 128。

## 5. 测试与验收

算子源库随附的验证用例：

- Host tiling UT：覆盖 A2/A3 的 CSA=1/2 成功、HCA 非法倍率拒绝、缺少 residual 和 SWA 非零倍率拒绝。
- Metadata API UT：通过 ParamsCheck 检查主接口与前置接口的倍率规则一致。
- ratio2 数值回归：覆盖 FP16/BF16、BSND/TND/PA_BBND、residual=0/1、边界压缩长度、双 batch 和多 query 行，并比较 attn_out 与 LSE。

vLLM Ascend 侧另有组网路由单测，验证 ratio 0/1/2 均进入原生 SparseFlashMla 路径。算子源库的 CANN UT/pytest 未复制到 `csrc` 发布目录。

上板前须重新编译/安装修改后的主算子与 Metadata，并在算子源库执行 Host UT、Metadata API UT 和真实算子数值用例。A2、A3 都要覆盖 ratio=1/2，已有 ratio=4/128 用例也需回归；Host UT 不执行 AICPU 任务切分，必须由真实 Metadata + 主算子调用补齐。

## 6. 已知边界与后续验证

arch22 的 TND 压缩长度读取当前直接使用 cu_seqlens_cmp_kv 相邻差值，Metadata 则优先使用 seqused_cmp_kv。对“有效长度小于存储长度”的 TND 输入，这两个口径需要另行统一。本次不改变原有长度接口语义，新增 TND 用例使用二者一致的长度，不将这种带 padding 的有效长度覆盖场景声明为已解决。

CSA 多核调度、G=1/128、非均匀 batch、aclgraph、极端空范围还应在目标平台验收时扩展覆盖。普通模式本次保持现有流水和内存设计，性能结论必须由 profiling 给出。

## 7. 当前仓库的模型编译范围

当前仓库按 Aurora 调用范围裁剪编译模板；前面列出的源库通用能力及其回归矩阵不等于本仓库裁剪后的支持范围。主算子只编译 BF16、TND Q、PA_BBND KV、SWA/CSA。BSND、非分页 KV、HCA 和两个独立 ori sparse 模板不再编译；host 会拒绝这些未编译的布局、模式或 dtype。

| 构建目标 | 裁剪前 key 数 | 当前 key 数 | 保留的硬件特化 |
| --- | --- | --- | --- |
| A2/A3 | 320 | 6 | CSA 的 `HEAD_RATIO_ONE=0/1`，`SPLIT_G=IS_VEC_S2PHYADDR=0` |
| A5 | 320 | 12 | `HEAD_RATIO_ONE=0`，保留 split-G 和 CSA 物理地址向量化 |
| Host | 320 | 14 | 两个设备集合的并集，仅用于 key 编码及校验 |

两个设备集合共有 4 个 key，共用部分分别编译。A2/A3 不编译 A5 专用标志组合；A5 不编译 A2/A3 的单头专用模板。确定性级别仍进入 host key，因此两边都保留 `BATCH_CONSISTENCY=0/1`。`FLASH_DECODE` 固定为 0，decode 调度由 metadata 驱动。模板声明的参数顺序、位宽和取值不变，保留的 key 编码不变。

`DeepseekV41EagerAttentionImpl._native_attention` 的 C0 走 SWA，C1/C2 走 CSA；cache spec 固定 BF16，query 与 KV 需要同 dtype。压缩比例、TopK 和请求长度都是运行时参数。编译保留全部本地 head 数边界，包括 TP 后只剩一个 query head 的 CSA。

无需 CANN 的矩阵回归：

```bash
python3 -m unittest discover -s tests/ut/ops -p test_aurora_tiling_keys.py -v
```

该检查枚举实际头文件的预处理结果，覆盖两种架构、模型调用布局、C0/C1/C2、单头和确定性边界，并检查 key 声明不变。它不执行 CANN 编译或 NPU kernel。key 数不包含 dtype 对编译任务数的影响，也不能直接换算整包编译耗时；耗时及数值结果需在配套 CANN/NPU 环境中重编译后验证。
