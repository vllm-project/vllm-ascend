# fused MC2 融合 Record 实验设计

扩展现有 `dispatch_ffn_combine` 接口，让 W8A8、W4A8、BF16 三套 AscendC 内核在产生本卡专家计数后，直接更新 EPLB 负载。支持的调用默认融合，不新增算子名称或用户开关；不满足条件的调用保留原有 MoE 和独立 Record 路径。

实验分支为 `feat/eplb-fused-mc2-record`，起点为上游 main 的 `8cd3c3762d8a10cd9bd814110c5a3b69999b6269`。本文记录方案与当前实验实现；构建、设备正确性和性能仍须分别验证。

当前实现把 Record 放在同一次 AscendC 算子启动的 `Process()` 末尾，由向量核 0 在原有 `expert_token_nums` 写出后读取并累加。`GetCumsumForMMAIV` 沿来源卡维度做前缀和，因此最后一行的每个元素已经是对应专家的独立计数，不能再对相邻专家做差分。此放置点与下文初始建议的 GMM 就绪信号之前不同，目的是先保持既有通信与矩阵乘重叠；需要通过性能测试确认取舍。

## 实验范围与等价目标

- 同时修改 W8A8、W4A8、BF16 的接口、Host 和 Kernel。
- 首轮框架接入覆盖 A3、V2 Model Runner、实际执行 `dispatch_ffn_combine` 的路径。三种内核分别做算子测试，136 上的 DSV4.1-W4A8 做整网验证。
- MegaMoE、普通 MC2、AllToAll、AllGather，以及旧 Model Runner 的 `moe_load` 记录机制沿用原实现。
- 不改变专家映射、通信策略、FFN、Combine、统计窗口、清零或重排策略。
- 融合前后 MoE 输出及 `expert_token_nums` 保持一致；负载数组与“原 MoE 加独立 Record”逐元素一致，且每次调用只记录一次。

原路径：MoE 产生专家计数，返回后独立启动 Record kernel。新路径：MoE 内部条件累加，返回后框架跳过独立 Record。保留原有专家计数输出，方便其他调用方使用和实验对照。

## 扩展现有接口

在 Python 接口现有参数之后追加以下可选参数，保留原返回值 `(out, expert_token_nums)`：

```python
torch.ops._C_ascend.dispatch_ffn_combine(
    # 现有参数保持原顺序
    ...,
    expert_load_view=None,
    record_enabled=None,
    local_expert_start=0,
)
```

| 参数 | 约定 | 作用 |
| --- | --- | --- |
| `expert_load_view` | 可选，一维 ND、INT32、元素连续的 NPU Tensor | 当前层按全局物理专家编号排列的负载视图，原地读写 |
| `record_enabled` | 可选，标量 BOOL NPU Tensor | 本次执行是否累计，由 Kernel 每次读取 |
| `local_expert_start` | 非负整数属性，默认 0 | 本卡第一个专家在负载视图中的元素偏移 |

两个 Tensor 必须同时存在或同时缺省；不能只传一个。都缺省时执行原计算且不写负载。都存在时，开关为 false 也属于融合路径，由 Kernel 跳过更新。对设备开关不调用 `.item()`，不转换成 CPU 布尔值，不写入 tiling 当作固定属性。

无需新增 `group_list_type`：三套 fused MC2 的输出都按独立专家计数处理，即原 Record 的 `group_list_type=1`。本地专家数沿用 `expertPerRank`。

校验同设备、dtype、维度、连续性、开关元素数，以及 `local_expert_start + expertPerRank <= expert_load_view.numel()`。允许指向更大分配中的连续层视图，正确尊重 storage offset；不通过 `.contiguous()` 复制可变负载视图。负载区间不能与计数输出、输入或 workspace 别名重叠。

直接调用时，半缺省、越界等非法参数明确报错。框架遇到合法但暂不支持的 dtype、布局或执行模式时，在启动前选择原路径，不静默改写参数。

## 参数穿过的层次

1. 框架从当前 MoE 层的 `AscendEplbLayerState` 取得负载视图、`should_record_tensor` 和本地专家起点。
2. 共用 PyTorch schema、Meta 实现与 C++ 适配函数接收三个新参数。
3. 适配层的三个分支分别向 `aclnnDispatchFFNCombine`、`aclnnDispatchFFNCombineW4A8`、`aclnnDispatchFFNCombineBF16` 传递参数。
4. 三套 ACLNN 头文件、实现、OpDef、生成的 Inner 接口保持参数顺序一致。
5. Host 校验参数配对和专家范围，tiling 保存专家起点；是否记录由 Kernel 中的可选地址及设备开关判断。
6. Kernel 入口和 `Init()` 接收负载与开关地址；`Process()` 在原内核执行完后调用共用 Record 辅助函数。

追加 CANN 输入会改变输入槽位布局，需要逐一核对三套实现的输入索引，不能只修改函数签名。新增起点属性追加在现有属性后，避免无意改变原属性索引。无需改矩阵乘的 `L1TileShape/L0TileShape`。

PyTorch schema 必须声明 `expert_load_view` 为可变参数，并使 Meta 与真实调用的输出及别名契约一致。CANN 层也必须正确表达负载 Tensor 的原地读写，通过其支持的输入输出引用方式绑定同一存储，不能把写操作隐藏成普通只读输入。优先验证可选参数、生成接口、原地引用与图模式；检查没有为该引用插入额外复制 kernel。

保留原算子名称，更新现有 ACLNN 签名；此次实验不要求新适配层兼容旧二进制 ABI。重新构建框架 C++ 扩展及三套自定义算子，并核对 136 实际加载路径和构建版本。不能对接口不匹配的旧库尝试调用，也不能在调用失败后重新执行原 MoE，以免重复通信或重复累加。

## Kernel 插入位置与共用实现

三套内核的 `DispatchAndCombine()` 均在来源卡计数汇总后，从 `cumsumMM` 的最后一行取出本卡各专家总量，并由 AIV 的 `coreIdx == 0` 写出 `expert_token_nums`。当前实现选择在同一次 Kernel 的 `Process()` 末尾读取此输出并做 Record，以免延迟原有 GMM 就绪信号。

```cpp
// 语义示意：原内核已由向量核 0 写出每专家计数。
originalKernel(params);
if (vectorCore == 0 && hasRecordInputs && LoadRecordEnabled()) {
    for (int i = 0; i < expertPerRank; ++i) {
        expertLoad[localExpertStart + i] += expertTokenNums[i];
    }
}
```

三套内核共用 `csrc/mc2/common/record_expert_load.h`。辅助函数只负责读取开关、读取旧负载、按 INT32 加法更新，以及流水同步；不含通信、专家映射或统计窗口管理。

当前使用标量 GM 读写，直接按有效专家数循环，不借用原内核的 UB 缓冲区，也不会写入填充专家。若性能测试表明标量访问成为瓶颈，再评估按块搬运和向量加法；该优化需要重新验证 UB 区域及事件生命周期。

仅向量核 0 写入，不新增核间或跨卡同步；读取计数前等待本核原有 GM 写入完成。新增工作位于算子末尾，可能增加尾部串行 GM 访问，必须用总耗时数据判断收益。

统计仍使用原 `expert_token_nums` 的口径，即所有来源卡分配给本卡物理专家的路由记录数。容量截断时也不改为另一种计数：如果已有输出是截断前计数，融合必须与原 Record 一样记录该值。

## 写入竞争与生命周期

各卡的负载数组是各自设备上的存储，按全局物理专家编号索引不意味着共享同一物理内存。Record 阶段不需要跨卡 atomic add 或新增 collective。

同一次 Kernel 仅核 0 写入，同一层、同一统计区间的各次执行须保持串行。第一轮不启用 DBO 或尚未证明正确的跨流并发记录路径，框架在这些模式下沿用已有记录机制；不为强行启用融合而增加原子操作或全局同步。

累加保持原 Record 的读、加、写语义，不清零、不扩容、不替换统计 Tensor。EPLB 控制器继续管理采样、统计窗口和清零。专家映射变化后，记录的仍是当前物理槽位；需要验证重新映射和图回放使用的存储、偏移没有失效。

Record 写入从 MoE 返回后提前到内部计数产生处，必须确认清零、消费统计和跨流访问通过原有依赖正确排序。异常执行不提供事务回滚，与原地 Kernel 写入一样不能盲目重试。

## 框架默认融合和防重复记录

在 `FusedExpertsResult` 增加 `load_recording_handled: bool = False`。它是每次调用的结果，不放进共享通信对象或全局状态。

`FusedMC2CommImpl.fused_experts()` 根据实际算子分支、量化类型、硬件、Model Runner、EPLB 状态与并发模式判断支持性。不能只根据 `enable_fused_mc2` 或 `MoECommType.FUSED_MC2` 判断，因为 MegaMoE 也使用这一通信类型。

```python
if supports_fused_record and eplb_state_is_ready:
    # 传入原始设备 Tensor，不检查 record_enabled 的设备值。
    dispatch_ffn_combine(..., expert_load_view=load,
                         record_enabled=enabled, local_expert_start=start)
    handled = True
else:
    dispatch_ffn_combine(...)  # Record 参数均缺省
    handled = False

return FusedExpertsResult(..., load_recording_handled=handled)
```

`routed_experts.py` 仅在 `load_recording_handled` 为 false 时调用原 `eplb_state.record_load()`。开关为 false 时仍返回 true，表示融合算子已负责执行“本次不记录”的决定；不再启动独立 Record。

状态获取和完整性校验收敛在 EPLB 层状态的辅助方法中，避免不同量化方法重复拼装状态。层状态对象存在但必要状态缺失时保留明确错误，不静默漏记。没有 EPLB 层状态则不进行记录。

| 场景 | 行为 |
| --- | --- |
| A3、V2、实际 plain fused MC2、W8A8/W4A8/BF16、状态及执行模式受支持 | 默认融合 |
| 设备开关 false | 融合 Kernel 不累加，外部 Record 也跳过 |
| 无 EPLB 记录状态 | 原 MoE 计算，不额外记录 |
| MegaMoE、其他通信方式、旧 Runner | 原有路径及原有记录条件 |
| 未验证硬件、DBO/并发模式、合法但不受支持的统计布局或 dtype | 在启动前回退原有路径 |
| 非法参数或内部状态未初始化 | 报错；不吞掉异常继续运行 |

## 图模式

捕图期间固定的是参数地址和支持性分支，记录开关的值由 Kernel 在每次回放时读取。统计 Tensor 与开关保持原 EPLB 对象及存储，不 clone、不在热路径重建、不在 Python 读取设备值。

验证同一张图在开关 false、true、false、true 的连续回放中，只在两次 true 时累计。验证捕图、预热、正常前向的开关行为与原 Record 一致；如果存储或本地专家区间发生变化，遵守现有图更新或重捕获机制，不使用旧地址继续执行。

## 实验对照和验收

不增加用户开关。测试入口显式构造两条路径，使用同一组权重、路由、输入，以及独立但初值相同的负载数组：

- A：不传 Record 参数调用扩展后的原接口，然后调用原 Record。
- B：传入 Record 参数调用同一个接口，不再调用原 Record。

基线阶段先验证“扩展后的接口参数缺省”与 main 原算子的输出及执行表现一致。整网 A/B 可用两套已记录提交或测试专用选择入口分别运行，不把实验控制加入生产环境变量或配置。

| 验证项 | 验收条件 |
| --- | --- |
| 三种量化的 MoE 输出 | A/B 使用相同精度判据；期望在未改变计算的情况下输出一致 |
| 专家计数与负载 | 整数逐元素完全一致，不重复记录，不漏记 |
| 负载初值和非零偏移 | 正确累加，仅本地专家区间改变 |
| 非整齐专家数和空负载专家 | 包括 25 个本地专家，尾部无越界写；零计数不改变原值 |
| 不均衡跨卡路由、TopK、冗余专家 | 与原 Record 的物理专家口径一致 |
| padding 和有效 Token | 按各量化实现已有有效输入处理规则，对照原路径；BF16 不擅自新增 mask 语义 |
| 开关与重复调用 | false 时完全不写负载，true 时每次恰好加一次 |
| 容量边界 | 使用原算子有效用例覆盖接近容量及可安全验证的截断场景，记录口径与原路径一致 |
| 参数缺省、非法输入、回退 | 缺省保留原行为，非法输入报错，未支持路径继续原 Record |
| 图捕获与回放 | 动态开关、存储偏移、统计清零和回放顺序正确 |
| profiler | 融合路径无独立 Record kernel；无为传参、原地引用或开关引入的额外复制、同步 kernel |
| 性能 | 预热后重复测量 MoE 加 Record 总延迟、图回放及整网 Decode；记录波动和资源成本，不只比较 kernel 数 |

136 整网验证使用 DSV4.1-W4A8，保持 A/B 模型、并行规模、EPLB 参数和请求负载一致。在 EPLB 消费或清零前的同一位置比较统计，避免将不同统计周期的数据作比较。确认受支持路径默认融合，并确认 MegaMoE 等回退路径没有因标记被误跳过 Record。

## 实施顺序

1. 明确三套 CANN 原地引用和可选参数契约，贯通接口及缺省路径，完成构建与加载检查。
2. 实现共用 Record 辅助函数，在三个内核计数输出位置接入；完成三种量化单算子 A/B。
3. 接入 EPLB 状态传递、默认选择和 `load_recording_handled`，验证支持与回退分支。
4. 验证图回放、开关和统计生命周期，再做 136 整网正确性与性能实验。

任何阶段没有正确性证据，都不以减少 kernel 数作为成功结论。若默认插入位置增加关键路径延迟，先比较总耗时，再考虑在同一 Kernel 中延后记录，且必须保证计数缓冲区仍有效；不为调整位置新增同步 Kernel。
