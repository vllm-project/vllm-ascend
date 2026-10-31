# DeepSeek V4.1 Flash 的 vllm-cann MegaMoE 适配说明

当前分支为 `integrate/ds-v41-flash-2db854`，按用户更正，以 `2db8549c548731611baeb169fa1ef13aa55fc6aa` 为直接基线，重新排列个人远端 `integrate/cann-megamoe-three-pr` 的 9 个提交。未保留此前误用 v0.31 基线时产生的代码适配。

更换基线后，代码与原整合分支保持一致；原基线 `7e37e29a7` 相比 `2db8549c5` 仅额外固定一条 Mooncake 依赖，本分支采用用户指定基线的 requirements.txt。仓库配套记录为 vLLM `v0.30.0`，对应提交 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`；Engram 保留从 `nvidia.engram` 导入的 v0.30 路径。

重新验证：32 项独立主机检查、8 项抽取权重测试、4 项抽取 A5 配置检查通过。完整 pytest、NPU 算子及端到端精度仍未验证。外部 MegaMoE 的 yOut 参数错位问题见文末补充，不在此次更换基线中修复。

## 1. 目标与当前状态

目标是在 `vllm-cann` 上运行 DeepSeek V4.1 Flash，并使用 MegaMoE；复用基线已有能力，保留三个原始 PR 的新增功能。

| 项目 | 内容 |
| --- | --- |
| 基线 | `2db8549c548731611baeb169fa1ef13aa55fc6aa` |
| 本地整合分支 | `integrate/ds-v41-flash-2db854` |
| 来源 | v0.27 开发的 !18、!31、!51，以及当前基线的适配修正 |
| 本次补齐 | A5 的 `2I=4608` 校验放行、DeepSeek MXFP4 权重预处理、超容量回退时的权重复用 |
| 验证状态 | 主机逻辑检查通过；真实 NPU 上 Flash 的启动、推理精度和性能尚未验收 |

用户已验证原始三个 PR 可运行 Flash，基线已在 Qwen 上运行 MegaMoE；这两项历史结果不能直接证明整合版本已通过 Flash 验证。

用户在 `fix/a5-megamoe-4608` 上反馈启动时出现 `aclnnMegaMoe ... yOut ... null`。该分支只有两处补丁，没有三个 PR 的适配；第 7 节记录这一待定位问题。

## 2. 复用与保留的功能

| 功能 | 原始方案与基线差异 | 最终处理 |
| --- | --- | --- |
| MegaMoE 执行入口 | 原始 PR 新建独立 `MegaMoEBackend`；基线已有 `FusedMC2CommImpl` | 复用基线执行类，不新增第二套后端 |
| 路由专家 MXFP 权重 | 基线已有逐专家 NZ 权重和 E8M0 scale 列表 | 复用父类预处理和 `get_fused_mc2_weights()`；补齐 DeepSeek 子类入口 |
| 通信组和对称 buffer | 原始 PR 有独立 MegaMoE 组；基线已有 MC2 组及 buffer | 复用 MC2 组、懒分配和接收容量处理 |
| 通信组销毁与恢复 | 基线已有失效标记和 `update_group()` 重绑定 | 保留基线实现 |
| 共享专家融合（!51） | 基线只向 MegaMoE 传递路由专家权重 | 增加共享权重、scale 和独立权重类型的传递 |
| A5 容量提前判断（!18） | 基线选中融合后才检查输入是否超限 | 选择前检查容量；超限继续选择 MC2、AllGather 或 AllToAll |
| P/D/混合容量规则（!18、!31） | 包括无 KV 配置的混合部署情况 | 统一选择、分配、执行检查的容量计算 |
| 空闲 DP rank（!18） | 原始方案区分真实任务与 dummy | 保留 V1 的协调优化；V2 使用基线已有的 DP 元数据机制 |
| Router | 基线已有 A5 Gate、hash 和 vision 路由 | 保留基线实现，仅在 hash 查表路径处理 ID 类型与 padding |

原始 PR：[!18](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/merge_requests/18)、[!31](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/merge_requests/31)、[!51](https://gitcode.com/xujiuxu9/vllm-ascend-dsnext/merge_requests/51)。

### 2.1 为什么需要额外补齐 DeepSeek 权重

Qwen 跑通不能覆盖 DeepSeek 的权重入口。Flash 的 `deepseek_v4_fp8` 配置在 `expert_dtype="fp4"` 时选择 `ds_w4a8_moe`，对应 `AscendW4A8MXFPDSDynamicFusedMoEMethod`。

该子类原先重写了 `process_weights_after_loading()`，未生成父类 MegaMoE getter 所需的 `cann_mega_moe_*_list`。三个 PR 加最终适配提交 `87b09d837` 也没有补齐这个子类，因此仅放开 4608 后会出现缺少权重属性的异常。

入口：[vllm_ascend/quantization/configs/fp8_config.py](../../vllm_ascend/quantization/configs/fp8_config.py)，`AscendDeepseekV4FP8Config.get_quant_method()`。

## 3. 本次补齐的修改

| 修改 | 行为与目的 | 代码入口 |
| --- | --- | --- |
| 放行 `2I=4608` | 将 4608 加入 A5 支持集合；其他 H、EP、专家数、Top-K 校验保留。修改的是插件校验，不代表外部算子包已支持此形状 | `vllm_ascend/ascend_config.py`：`_is_a5_megamoe_supported_by_config()` |
| DeepSeek MegaMoE 预处理 | 记录原始 dtype，将 packed FP4 和 scale 按字节解释，再调用父类生成逐专家 NZ 权重和 E8M0 scale | `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py`：DeepSeek 子类 `process_weights_after_loading()` |
| 防止重复处理 | 已有 `_mxfp4_transformed` 时直接返回 | 同上 |
| 权重重载恢复 | 从逐专家列表恢复 checkpoint 的 ND 形状和 dtype，清理旧列表，下次加载重新准备 | 同文件：`restore_weights_for_rl_loading()` |
| 普通 GMM 回退 | 原始堆叠参数已释放时，复用 MegaMoE 权重的转置视图和 scale 的字节视图；不复制第二套完整专家权重 | 同文件：DeepSeek 子类 `_resolve_gmm_weights()`、`_resolve_gmm_scales()` |
| 动态 EPLB | DeepSeek MegaMoE 预处理遇到动态 EPLB 时明确拒绝，避免权重所有权冲突；未开启 MegaMoE 的既有 EPLB 路径保留 | 同文件：`process_weights_after_loading()` |

代码：[配置校验](../../vllm_ascend/ascend_config.py)、[DeepSeek 权重处理](../../vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py)。

### 3.1 输入和权重布局

`T` 是本次实际进入算子的 token 行数，可能含 padding；`H=hidden_size`；`I=moe_intermediate_size`；`E_local` 是当前 rank 的本地路由专家数；`K` 是 Top-K。

| 参数 | Flash W4A8MXFP 路径中的布局 |
| --- | --- |
| `hidden_states` | `[T, H]` |
| `topk_ids` | `[T, K]`，调用前转为 int32 |
| `topk_weights` | `[T, K]`，调用前转为 float32 |
| `w1` | 长度为 `E_local` 的列表；每项逻辑形状 `[2I, H/2]`，packed FP4，物理 NZ 格式 |
| `w2` | 长度为 `E_local` 的列表；每项逻辑形状 `[H, I/2]`，packed FP4，物理 NZ 格式 |
| `w1_scale` | 每项 `[2I, H/64, 2]`，E8M0 |
| `w2_scale` | 每项 `[H, I/64, 2]`，E8M0 |
| `sym_buffer` | 按配置容量和 MC2 通信组分配；各参与 rank 的分配参数必须一致 |

以上 scale 形状对应当前 group size 32、相关维度可整除的 Flash 路径。`view(dtype)` 保留字节位模式，不是重新量化。

普通 GMM 回退使用相同权重存储，但以 `[K_dim, N_dim]` 的转置视图读取；scale 转为 uint8 并交换前两维。真实 NZ 描述符和 GMM 接口兼容性仍需 NPU 验证。

## 4. 最终执行顺序

```text
加载配置、建立模型与通信组
    ↓
AscendDeepseekV4FP8Config.get_quant_method()
    → ds_w4a8_moe
    → DeepSeek 子类 process_weights_after_loading()
    → 父类生成 cann_mega_moe_*_list
    ↓
每次模型 forward：结合各 DP rank 的 token 信息选择通信方式
    ↓
DeepseekV41MoE.forward()
    → AscendMoERunner._forward_impl()
    → AscendRoutedExperts.forward_impl()
    → 量化方法 apply()
    ├─ FUSED_MC2 且 use_mega_moe=True
    │    → FusedMC2CommImpl.fused_experts()
    │    → get_fused_mc2_weights()
    │    → _apply_cann_mega_moe()
    │    → cann_ops_transformer.ops.mega_moe()
    └─ 普通通信方式
         → dispatch → GMM/激活/GMM → combine
         → 复用同一批逐专家 NZ 权重的视图
```

模型、Attention、Engram 和加载框架复用基线，本次未新增模型文件。Flash 的模型入口见 [vllm_ascend/models/deepseek_v41/model.py](../../vllm_ascend/models/deepseek_v41/model.py)，`DeepseekV41MoE`。

### 4.1 A5 何时选择 MegaMoE

首先满足 MoE、开启 EP、实际 EP 通信组大于 1、无 LoRA 等通用适用条件。然后同时满足：

| 条件 | 含义 |
| --- | --- |
| `enable_fused_mc2 == 1` | 用户配置 2 后内部转为 1，且配置校验未将其禁用 |
| `is_mega_moe_supported()` | MegaMoE 标志为真；用户配置 2 时依据 `cann_ops_transformer` 包是否存在设置 |
| 主模型，或 draft 量化在允许集合中 | A5 draft 允许 W4A4MXFP、W4A8MXFP、W8A8MXFP |
| 当前 token 数不超过 A5 输入容量 | 超过容量时不选 FUSED_MC2，继续普通通信选择 |

最终执行还要求 `use_cann_megamoe()` 为真，包括硬件能力和实际 EP 规模检查。包存在、配置允许、选中 FUSED_MC2 都不能代替真实算子验证。

代码：[vllm_ascend/ascend_forward_context.py](../../vllm_ascend/ascend_forward_context.py)，`select_moe_comm_method()`、`_select_capacity_and_world_size_moe_comm_method()`、`use_cann_megamoe()`。

### 4.2 输入容量与多 rank

统一容量为 `min(mega_moe_max_tokens // world_size_across_dp, 当前 PD 角色的执行容量)`。这是当前适配代码使用的配置规模，并非直接读取实际 EP group 大小。

| 场景 | 执行容量来源 |
| --- | --- |
| P、混合 PD、无 KV 配置 | `scheduler_config.max_num_batched_tokens` |
| D | 已初始化的 `mc2_tokens_capacity` |

通信选择、A5 对称 buffer 分配、执行入口检查共用这套计算。入口仍保留超限异常作为兜底，不在进入集合通信后临时切换路径。

输入容量与 dispatch 后的接收容量是两件事：后者仍沿用基线按 EP、Top-K、本地专家数计算安全上界和配置接收上限的机制。

Flash 在 A5 必须使用 V2 Runner，见 `vllm_ascend/platform.py` 的校验。V2 的平台 hook 根据 `dp_metadata.num_tokens_across_dp_cpu` 的最大值选择通信方式；三个 PR 移植的“真实任务/dummy 标记”优化位于 V1，不能表述为 Flash V2 已新增相同优化。V2 空闲 rank、图模式和 MTP 的协调需要实机验收。

V2 平台 hook 当前调用 selector 时没有显式传入 draft 身份和量化类型；保留了基线行为，不能仅凭 selector 的 draft 判断认为 V2 MTP 已验证。

### 4.3 共享专家融合

共享专家功能保留，但满足条件才融合；不满足时独立计算共享专家，路由专家仍可使用 MegaMoE。

| 检查 | 当前适配范围 |
| --- | --- |
| 算子接口 | 明确暴露共享两组权重、scale 和两个独立权重类型参数 |
| 量化 | 路由专家 W4A8MXFP；共享专家 W4A8MXFP 或 W8A8MXFP |
| 布局与激活 | 单个共享专家、匹配的 H/I、连续权重、兼容的 SiluAndMul 及 clamp 参数 |
| TP | TP>1 时要求共享权重完整复制 |
| 组合限制 | 不融合动态 EPLB、冗余专家、mix_placement 或不同的共享输入 |

融合结果放在共享输出槽，路由输出槽返回同形状零张量，避免上层重复相加或将路由专属缩放施加到共享贡献。

代码：[Runner](../../vllm_ascend/ops/fused_moe/fused_moe.py) 的 `_forward_with_fused_shared_experts()`、[共享权重](../../vllm_ascend/ops/fused_moe/shared_experts.py) 的 `get_mega_moe_weights()`、[接口和布局检查](../../vllm_ascend/ops/fused_moe/mega_moe.py)。

## 5. 主要代码目录

| 路径（从仓库根目录起） | 作用 |
| --- | --- |
| `vllm_ascend/ascend_config.py` | 配置校验、4608 放行、容量计算 |
| `vllm_ascend/ascend_forward_context.py` | A5 通信选择、容量查询、MegaMoE 执行标志 |
| `vllm_ascend/quantization/methods/w4a8/w4a8_mxfp4.py` | DeepSeek 预处理、权重 getter、普通 GMM 视图、重载恢复 |
| `vllm_ascend/ops/fused_moe/fused_moe.py` | 共享融合选择、内部路由和输出衔接 |
| `vllm_ascend/ops/fused_moe/shared_experts.py` | 共享权重准备、兼容性检查和缓存失效 |
| `vllm_ascend/ops/fused_moe/mega_moe.py` | 共享参数 ABI 探测、scale 和布局检查 |
| `vllm_ascend/ops/fused_moe/moe_comm_method.py` | 复用 FusedMC2CommImpl 调用算子，接入共享参数和统一容量 |
| `vllm_ascend/ops/fused_moe/routed_experts.py` | 传递共享权重和融合结果标记 |
| `vllm_ascend/ops/fused_moe/dataclass/` | 输入权重和融合状态的数据字段 |
| `vllm_ascend/ops/fused_moe/router/fused_topk_router.py` | hash 查表 ID 和 padding 处理 |
| `vllm_ascend/worker/model_runner_v1.py` | V1 真实任务/dummy rank 元数据协调 |
| `vllm_ascend/utils.py` | 防止融合路径跳过必要的 DP 同步 |

`DeviceOperator`、prepare/finalize、parallel state 和 distributed patch 与基线没有净差异；HCCL teardown/restore 继续复用基线。

## 6. 验证结果与待验收项

| 检查 | 结果 |
| --- | --- |
| `tests/ut/ops/test_dsnext_megamoe_port.py` | 32 项主机逻辑回归通过；包含 V2 执行阶段无配置上下文时的首次分配、复用与超限检查，执行生产函数，依赖使用替身 |
| DeepSeek 权重测试 | 8 项生产方法抽取检查通过；覆盖字节内容、幂等、重载、普通布局、EPLB 和回退视图，使用主机张量替身 |
| A5 配置测试 | 4 项生产方法抽取检查通过；4608 及原有 H/EP/Top-K 限制 |
| Python 语法 | 修改的 Python 文件解析通过 |
| Ruff | 本次新增逻辑通过；整组文件检查仍有基线已有的 V1 Runner 导入/行长问题，以及配置文件原有格式差异 |
| 完整 pytest、`bash format.sh ci` | 本机缺少 pytest、运行依赖和 Bash，未通过完整执行验证 |
| Flash 真实权重/NPU | 待验证；当前不能标记“模型已跑通” |

回归测试：[权重单测](../../tests/ut/quantization/methods/w4a8/test_w4a8_mxfp4.py)、[配置单测](../../tests/ut/test_ascend_config.py)、[适配主机检查](../../tests/ut/ops/test_dsnext_megamoe_port.py)。

NPU 验收顺序：先使用真实 Flash 权重完成启动和文本请求，再验证图模式、共享融合与独立共享路径的一致性、容量边界两侧的普通回退、空闲 DP rank，最后增加 MTP 和性能采集。Qwen 需回归以确认已有 MegaMoE 路径保持正常。

部署沿用已验证的 Flash 启动参数，开启 `VLLM_USE_V2_MODEL_RUNNER=1`、`--enable-expert-parallel` 和 `--additional-config '{"enable_fused_mc2":2}'`；已有 additional-config 应合并，避免覆盖模型其他配置。保留此前已解决显存问题的 `--gpu-memory-utilization 0.85`，它不是所有部署的通用阈值。

## 7. 当前实机异常：yOut 为空

用户在 `fix/a5-megamoe-4608` 上反馈的完整堆栈为：

```text
determine_available_memory()
  → V2 profile_run() → _dummy_run()
  → DeepseekV41MoE → AscendMoERunner → AscendRoutedExperts
  → W4A8MXFP apply()
  → FusedMC2CommImpl._apply_cann_mega_moe()
  → cann_ops_transformer/ops/mc2/mega_moe/mega_moe.py:589
  → torch.ops.cann_ops_transformer.npu_mega_moe()
  → 同文件 :200 → _op_module.npu_mega_moe()
  → aclnnMegaMoe：Expected a proper Tensor but got null for argument yOut
```

这说明失败发生在启动时的显存 profiling dummy forward，已经选中并进入 MegaMoE 外部扩展。它不是“缺少权重列表属性”的同一异常，也不是收到请求后的容量溢出。

`yOut` 是底层接口要求的输出张量。当前证据只能确定该参数没有形成有效的 Tensor 描述；不能据此断言是权重、空 batch、4608 形状或算子版本导致，也不能直接给 Python 调用增加未经确认的 `out=` 参数。

### 7.1 官方算子源码核对

官方仓库已下载到 `D:\code\ops-transformer`，核对版本为 `6e7eea192790b260e52d3763ef1c088a069a5369`。使用稀疏检出，当前展开 `mc2/mega_moe`。

| 源码 | 已确认的行为 |
| --- | --- |
| `mc2/mega_moe/torch_extension/csrc/mega_moe.cpp`：`NpuMegaMoe()` | 第 389 行用 `at::empty({bs, h}, ...)` 创建输出 `y`；第 436 行起通过 `ACLNN_CMD` 传入 `y` 和 `expertTokenNums` |
| `mc2/mega_moe/op_api/aclnn_mega_moe.cpp`：`aclnnMegaMoeGetWorkspaceSize()` | 第 87 行检查 `yOut` 非空；检查发生在调用内部 workspace/tiling 逻辑之前 |
| 同一 API 的签名 | 输出之前包含 `rankNumPerServer`、`topkWeightsType`、`combineCommMode` 等属性；扩展与实际加载的 API 库必须使用匹配的参数序列 |

因此不应直接在 vLLM Python 层增加 `out=`：该版本的公开 Python API不接收这一参数，输出由扩展创建。当前优先检查“Python/JIT C++ 扩展与实际加载的 ACLNN 算子库版本是否匹配”，以及 Tensor 到 ACL 描述符的转换是否成功。ABI 不匹配是排查假设，尚未证实。

源码：[C++ 扩展](https://gitcode.com/cann/ops-transformer/blob/6e7eea192790b260e52d3763ef1c088a069a5369/mc2/mega_moe/torch_extension/csrc/mega_moe.cpp)、[ACLNN 接口](https://gitcode.com/cann/ops-transformer/blob/6e7eea192790b260e52d3763ef1c088a069a5369/mc2/mega_moe/op_api/aclnn_mega_moe.cpp)。上述是所下载版本，不能替代用户容器的安装版本。

### 7.2 容器内继续定位

先查实际 Python 包和接口：

```bash
python -c "import inspect; import cann_ops_transformer.ops as ops; print(ops.__file__); print(inspect.signature(ops.mega_moe)); print(inspect.getsourcefile(ops.mega_moe))"
```

没有 `rg` 时使用 `find` 和 `grep`：

```bash
find /usr/local/python3.11.10/lib/python3.11/site-packages/cann_ops_transformer \
  -type f -iname '*mega*'

grep -R -n -C 15 \
  -E 'yOut|y_out|aclnnMegaMoe|at::empty|torch::empty|empty_like' \
  /usr/local/python3.11.10/lib/python3.11/site-packages/cann_ops_transformer/csrc/
```

然后核对已安装算子包的 `aclnn_mega_moe.h` 与 C++ 扩展调用参数；检查 Worker 实际加载的 `libcust_opapi.so` 等库的路径，而非只检查磁盘上存在的文件。还需要首次底层错误前的日志，以及本次 `T/H/2I/K`、权重 shape/dtype、CANN 与算子包版本。

此项尚未修复，不能将整合分支写成已经解决该报错。

### 7.3 整合分支新增异常：Current vLLM config is not set

用户在整合分支启动 profiling 时，执行到 `FusedMC2CommImpl._apply_cann_mega_moe()` 的 A5 容量查询就抛出断言，尚未调用外部算子。

原因：适配新增的运行时容量检查与首次 buffer 分配调用了 `get_current_vllm_config()`；V2 的 CustomOp 执行阶段可能已退出加载模型时的配置上下文。原单测持续 mock 配置 getter，未覆盖这个生命周期差异。

修复：在 `FusedMC2CommImpl.__init__()` 的配置上下文内计算 A5 容量，保存为 `_mega_moe_tokens_per_rank`；执行检查和首次 buffer 分配复用同一值，不在 forward 内读取当前配置。容量规则和超限检查保留，A3 继续使用原 dispatcher 容量。

新增三项回归检查，在初始化后令配置 getter 抛出与实机相同的断言，验证首次 forward、后续 forward 和超限拒绝；修复前均失败，修复后通过。同步更新绕过构造函数的单测和 NPU 算子测试夹具。

该断言与第 7.1 节的 `yOut=null` 是两个问题。修复断言只恢复了进入底层算子的执行链，不代表已经解决 `yOut` 或完成 Flash NPU 验证。

### 7.4 yOut ?????????

????????? Tensor ??????????????? MegaMoE API ??????? libopapi_transformer.so????? yOut ?????????? 33 ????? yOut????? 33 ???? topkWeightsType=0?y ??? 34 ??

ops-transformer ?? `a0d7581a2283c5a0902a5773e4a24bfd4a4a83dc` ???????????????????? sharedExpertQuantOutDtype/sharedExpertQuantResultType??????????????????????????????????????????????????????????????
