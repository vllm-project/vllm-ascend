# DSV41 MRV2 适配移植对照总表（对应 MRV1 / 上游）

> **行号基准**：v1 = `vllm_ascend/worker/model_runner_v1.py`、上游 = `C:\code\vllm`（84030bbe3d，vllm 0.29 base）、
> v2 各文件 —— 均为分支 `main-mrv2-dsv41-v5` @ 本地 HEAD（0a8290db6）。
> **范围**：移植主体 `a30b53c32..ca9cb2fec`（11 提交）+ 优化轮本地提交。
> **专项文档**：attn_utils.py 逐块对照见 `docs/dsv41_mrv2_attn_utils_porting_map.md`（附录 A，本文不重复）。
> **状态标注**：✅ 已逐行核实 ｜ ◐ 推断（有依据未逐行核）｜ ⏳ 待核。

## 一、贯穿全表的四个移植模式

1. **随执行栈换挂载点**：同一语义，v1 的 runner 方法 ↔ v2 的 model_state 钩子 / patch 函数 / 模块级函数。
   根因：MRV2 中段执行栈是上游代码（execute_model / model_state / compiled wrapper），挂载点别无选择。
2. **ContextVar 带外通道**：runner 私有 flag 需穿越上游调用链（上游 `_dummy_run` 丢弃未知 kwargs）时，
   用 ContextVar（复用上游 `override_mrv2_in_profile_run` 先例）；MRV1 全自有链条用显式参数。
3. **patch 框架收纳**：对上游行为的替换统一入 `patch/worker/patch_v2/`（import 时生效、无 global 标志、理由与 patch 同址）。
4. **类型/契约适配**：绑定值类型（list 包装）、元素语义（element_size 泛化）、块尺寸（re-block 后恢复）。

## 二、worker/v2 核心

### 2.1 model_runner.py

| v2 符号（行号@HEAD） | 作用 | v1 / 上游对应 | 状态 |
|---|---|---|---|
| V4.1 eager fallback → `patch_v2/patch_set_forward_context.py`（原 L93-126 内联函数，0a8290db6 移入） | V4.1 runtime-NONE 步（prefill/非均匀 decode）绕过 compiled wrapper | 上游 `skip_compiled=has_encoder_input` 仅覆盖 encoder-decoder（gpu_model_runner.py L4434；消费点 decorators.py L514）；**MRV1 无需对应物**——自有 execute_model 原生传 skip_compiled（L2499）且执行栈无 dynamo 层 | **🗑️ 已删除（b9868130a，2026-09-24 NPU 实测）**：去掉 fallback + 保留 O4 后功能精度正常——compressor/indexer prefill 编译路径在 compiled wrapper 下可用，删除条件（§上游对齐文档"待实测通过后再删"）已达成 |
| `initialize_kv_cache` 内 ring compressor 预备（L276-284） | ratio-2 环压缩器持久 buffer 校验 + Triton core 解析（捕获前必须） | v1 L4478-4484 **逐行同逻辑**（同门控/同 lazy import/同 model.modules() 扫描） | ✅ |
| `prepare_v41_source_rope`（worker/v2/utils.py L85；fd2ecd02b 自 NPUModelRunner 方法迁出，公共 runner 不放模型专属 hook） | V4.1 source RoPE 表初始化（`build` 硬断言依赖） | MRV1 无此方法——经 `initialize_attn_backend` → provider 循环（L5801-5812，机制源自 db5f9e023/#15443）→ `enable_device_metadata` → `prepare_source_rope()` 传递获得；MRV2 metadata task 同步执行，不可翻转 async 开关，故拆出独立 hook 直调 | ✅ |
| `prepare_v41_dummy_ring_state`（worker/v2/utils.py L103；fd2ecd02b 自 NPUModelRunner 方法迁出） | dummy 环页：独立活页 1..num_reqs + 物理页清零（防别名互踩/脏状态） | v1 L4026-4040 dummy-run 内联**逐语义等价**（fill(0)→arange→zero_）；差异：v1 显式 fill0（复用脏表）、覆盖到 num_reqs_padded | ✅ |
| `_dummy_run` skip flag 路由（L786-792 pop + L804 ContextVar with） | `worker.execute_dummy_batch`（上游引擎空闲步，llm_engine L305-307 → multiproc_executor L366）的"只同步不推进状态"语义跨上游栈传输 | v1 显式参数：`_dummy_run` 签名 L3850、环预备门控 L4032、metadata 透传 L4078/L3690；上游 `_dummy_run` 签名无此参（gpu/model_runner.py L744-752） | ✅ |
| `prepare_dummy_attn` 末尾调用 `prepare_v41_dummy_ring_state`（utils） | 捕获/dummy 双入口统一 | v1 在 dummy-run 内联 | ✅ |

### 2.2 worker/v2/model_states/default.py（ffd41ddea engram hooks）

| v2 符号 | 作用 | v1 对应 | 状态 |
|---|---|---|---|
| `_get_engram_history_inputs`（L45-75） | 每步 engram history（边界/CPU 块表/存储块大小）；dummy 返回 None 护 n-gram store | `_get_engram_history_inputs` L3151-3166（runner 方法）；**差异**：v1 在 forward context 内（可查 attn_metadata）、数据为 host 镜像；v2 在 context 前夕（用 kvpp dummy + ContextVar 判定）、数据从 prepare_attn 缓存视图 `.cpu()` 拉取 | ✅ |
| `prepare_inputs` 重写（L77-96） | eager engram 路由（dynamo 区外唯一挂载点） | v1 `_model_forward` 内注入 L3192-3202（capture→graph inputs / 常规→eager 路由） | ✅ |
| `prepare_dummy_inputs` 重写（L98-106） | FULL 捕获绑定固定地址 graph inputs | 同上（v1 capture 分支） | ✅ |
| `prepare_attn` 缓存视图（L166-167） | hook 无 runner 引用，缓存每步 block_tables/kv_cache_config | v1 无需（hook 有 self） | ✅ |
| `full_graph_mode` 接线（L191-194） | 与 MRV1 表达式对齐（行为 no-op，FULL 必为纯 decode） | v1 L3685/L3693 | ✅ |

### 2.3 patch/worker/patch_bind_kv_cache.py V4.1 分支（a30b53c32）

| v2 符号（L41-48） | 作用 | v1 对应 | 状态 |
|---|---|---|---|
| isinstance 门控 + list 包装 + sorted 确定序 + 提前 return | `DeepseekV41CacheLayer` 是 nn.Module，类契约 `kv_cache=[...]`（dsa_v41.py L1113）；裸 tensor 的 `[0]` 索引语义错误；默认路径 layer_index 排序不适用 | v1 L4572-4576 / L4589 同款门控+包装（v1 在自有 bind 路径内联；v2 的 bind 经 patch_attn_utils 路由到本共享函数） | ✅ |

## 三、attention/dsa_v41 系

| v2 修改（提交） | 作用 | v1 / 上游对应 | 状态 |
|---|---|---|---|
| `_request_counts` 掩码长度修正（99218bcf9） | MRV2 `is_prefilling` 真实请求数 vs `query_start_loc_cpu` padded 的错位 | MRV1 padded 语义**保留**（UT `test_mrv1_padded_flag_length_keeps_legacy_counts` 双语义对照）；基础逻辑来自框架（0deca318） | ✅ |
| 零宽选择 `fill_(-1)`（99218bcf9） | 空 compressed source（首个 decode 前）→ 无效标记跳过压缩段 | 框架代码增强；v1 侧是否同补 ⏳ | ⏳ |
| `prepare_source_rope` 从 `enable_device_metadata` 拆分（e22d5d050） | MRV2 metadata task 同步执行，独立 hook 不得翻转 async 开关 | v1 的 `enable_device_metadata` 内联 rope init（0deca318 原状）；拆分后 v1 行为不变（enable 内部自调） | ✅ |
| builder 恢复逻辑 block_size（ca9cb2fec） | **精度修复（用户确认失败语义）**：完整因果链见 `docs/dsv41_mrv2_kv_nan_rca.md`。摘要：v2 re-block 后 builder 槽位数学若沿用 re-block 尺寸，metadata.logical_block_size 比例关系被抹平 → 压缩算子逻辑↔存储换算偏移 ratio 倍 → 写点落错页 → **跨请求 KV 覆写 → 主模型输出 NaN**。修复：builder 恢复 spec 双粒度契约；槽位映射数学不受影响（修复面精确）。UT `test_builder_restores_v41_logical_block_size` 看护 | **v1 无对应**——v1 不做 re-block（v2 特有的 kernel_block_sizes 分组机制），builder 直接拿原始 spec，故 v1 不存在此失效面 | ✅ |
| `_v41_logger` 调试日志清理（2dffca268） | 本轮优化（bring-up 遗留） | 非移植内容 | — |
| `dsa_v41_cp.py` prepare_source_rope forward 删除（a8d65b120） | MRV2 不支持 dsa_cp（不可达）；MRV1 经 `_global_builder.enable_device_metadata()` 自调冗余 | v1 框架原生文件回到 0deca318 原状 | ✅ |
| `sequence_parallel.py` 三个 custom op（7f5128de9） | SP shape math 对 dynamo 不透明（防 shape 烘焙）；修复上游 `sequence_parallel_chunk` 对 `[T, hc_mult, H]` pad 维度错误 | **v1 无需**（无 dynamo 层）；上游 chunk 函数缺陷为不支持多 trailing dims | ✅（commit 记录） |

### 3.1 逐文件详解与优化评估

**① sequence_parallel.py（7f5128de9）**
- 作用：SP 的 `sp_shard` / `sp_reduce_scatter` / `sp_padding_mask` 注册为 custom op（含 fake_impl），
  使 modulo padding 的 shape math 对 dynamo 不透明——普通 Python 实现会把 trace 时的 shape 烘焙进编译图，
  其他 bucket 尺寸捕获时 shard 行数错误；同时修复上游 `sequence_parallel_chunk` 的 pad 错轴
  （`F.pad(x, (0, 0, 0, pad_len))` 对 `[T, hc_mult, H]` 输入 pad 的是 hc_mult 轴而非 token 轴，
  `T < tp_size` 时（如 TP8 上 6-token dspark draft 步）每个 rank shard 到 0 行）。
- 必要性：**必须**——DSV41 的 SP 输入是 3D 且 MRV2 走编译；v1 无此问题（无 dynamo 层 + v1 无该调用形态）。
- 优化候选 **O4（✅ 已完成，perf(sequence_parallel) 提交）**：`_ascend_sp_shard_impl`（L47）与 `_ascend_sp_reduce_scatter_impl`（L82）
  的 `torch.cat([x, zeros])` **无条件执行**——`sp_pad == 0`（decode 常见：TP 对齐的 bucket）时仍整表拷贝。
  **最终实现 = 上游同款搬运**（2026-09-23，对照上游后落地）：
  - 守卫与 pad 形式照搬 `vllm.models.common.ops.sequence_parallel`（`if sp_pad > 0:` + `F.pad(x, (0, 0) * (x.ndim - 1) + (0, sp_pad))` 任意 ndim）；reduce_scatter/padding_mask 同款守卫 + `F.pad`。
  - **别名风险已由设计解决**：上游 custom-op 版 `sequence_parallel_chunk_impl`（model_executor/models/utils.py L1062-1065）明确注释 "a functional custom op must not return a view of an input"——no-pad 切片必须 `clone()`。据此 shard/padding_mask 在 `sp_pad == 0` 时 `out.clone()`（只拷本 rank chunk = T/tp 行，替代原来整表 T 行拷贝，TP8 省 87.5%）；reduce_scatter 输出为集体通信新张量，无别名问题、无需 clone。
  - 保留 custom op 包装（dynamo 形状烘焙危害仍在）；UT 补 no-pad 非视图契约断言 + padding_mask 两用例。NPU 上的行为/性能验证仍建议做一次。
  收益：decode（TP 对齐 bucket）场景每次 SP 调用省整表 pad 拷贝（shard/padding_mask 降为本 rank chunk 拷贝；reduce_scatter 全免）。
  原风险记录（已被 clone 方案消解）：返回值从私有拷贝变为**输入的视图**（别名变化）——需 NPU 验证下游无对视图的 in-place 写。
- 注册方式不一致注记：shard/reduce_scatter 用 `dispatch_key="PrivateUse1"`，padding_mask 用
  `needs_fixed_stride_order` tag——行为已验证，统一需逐个 NPU 验证，不动。

**② deepseek_v41/model.py MTP hidden buffer（e125401b8）**
- 作用：spec（MTP/DSpark/DFlash）+ last-PP 时条件预分配 `_mtp_hidden_buffer`；forward 末段
  hc_collapse 之前 SP gather（hidden_states 与 pre_mix 双 gather）并 stash full HC 残差态供 draft 消费。
- MRV1 对照：v1 的 V4.1 版为 **lazy None（从未分配/填充）**——本修改实为**功能补全**而非仅对齐：
  对齐 deepseek_v4/model.py 同款模式（0deca318 内 L986-996，上游参照 vllm PR #50312）。
- 优化评估：**无可优化点**。SP gather 前置与整表 copy 是 MTP+SP 契约的内在要求
  （draft 消费的是 gathered full HC 态，pre-gather stash 会存成分片态；双 gather 换单 gather 需改 draft 契约）。

**③ dsa_v41.py（99218bcf9 / e22d5d050 / ca9cb2fec / 2dffca268）**
- 四个移植块均已核实必要（详见本文各节）：`_request_counts` 双语义修正（+UT）、零宽选择 fill(-1)、
  `prepare_source_rope` 拆分、builder block_size 恢复（RCA 见 `docs/dsv41_mrv2_kv_nan_rca.md`，精度承重墙）。
- 优化检查：`_request_counts` 内的 `.item()` 均作用于 **CPU 张量**
  （is_prefilling/query_start_loc_cpu 源自 numpy），无设备同步问题；
  本轮新增逻辑均为小粒度坐标/标记计算，无进一步可优化点。

## 四、fused_moe 系（共享 ops，v1/v2 共用代码）

### 4.0 审查结论（2026-09-23，用户审定"通信算子不区分模型"+最小修改原则）

| 文件 | 判定 | 对应提交 |
|---|---|---|
| `force_eplb.py` | **保留 ✅**——`_EXTRA_CTX` 读取修复实际观察到的 MRV2 下 force EPLB 失效（裸属性读在 additional_kwargs 上静默返回 None）；None→透传 topk_ids 修复调用方回写崩溃 | 8746dc38b（曾 0ec71c078 禁用 → 1e9fe4912 撤销，净零） |
| `token_dispatcher.py` | **⚠️ A/B 验证中**——HCCL 独立 comm stream 与 D2H 读序问题，draft dummy 路径实际观察到全零 output_splits；修复（gather 后同步 + 去 non_blocking）经 59928911f 临时移除做 NPU A/B 验证：失败复现 → 修复必要（revert 该测试提交恢复）；无失败 → 修复不必要（保留回退）。收窄 event wait 列 P3 | 641ba3d22 引入 → 59928911f 临时移除（待 NPU 结论） |
| `fused_moe.py` | **⚠️ 下游节点结论作废**——O3 回退（7001be84c）与 A/B（3c2ba9a71）建立在注册表回退失败节点之后，其"可回退"结论未经有效验证；待 247bc9890 恢复注册表后重测 | 52c6ddac5 引入 → 7001be84c 回退（**结论作废待重测**） |
| `moe_comm_method.py` | **↩️ 回退已撤销（247bc9890，NPU 实测驱动）+ 搭车优化已剥离（e59a51dcf）**——219009f60/ca9e7c0cd 的回退在其测试栈上拉起失败（all2all `split sizes doesn't match total dim 0 size`），证明 `activate_moe_comm_method` 逐层 rebind 在真实运行配置下**承重**，此前"单形状恒等休眠"的静态分析判断被 NPU bisect 证伪（af7606669 注册表在位正常 / ca9e7c0cd 回退后失败）。已 `git revert ca9e7c0cd` 恢复注册表 + rebind；随后按最小修改原则剥离两个搭车优化：O1（`_MoECommShapeKeys` 写路径集合 → activate 回归注册表重算单形状检查）与 O3（setup 返回值已无消费者 → 回归无返回形态）。**终态 = 纯修复核心**：形状注册表 + 去重复用 setup + 双参 get + activate rebind（与 641ba3d22 净差仅 docstring）。根因分叉字段待插桩定位（setup 打印 9 字段 key + prepare 处对比层 config vs 发布实例），定位后可评估把防御性 rebind 收敛为启动期断言 | 641ba3d22 引入 → 219009f60/ca9e7c0cd 回退 → 247bc9890 恢复 → e59a51dcf 剥离 O1/O3 |
| `routed_experts.py` | **↩️ 随注册表恢复**——逐层 rebind 回归（247bc9890），mypy assert 一并保留（与 af7606669 功能态唯一净差 = 3 行 assert） | 219009f60/ca9e7c0cd 回退 → 247bc9890 恢复 |

**净差终态（2026-09-23 更新）**：fused_moe/ 目录相对基线 = force_eplb.py（+19/-3）+ token_dispatcher.py（A/B 临时归零）+ moe_comm_method.py/routed_experts.py（注册表与 rebind 经 247bc9890 恢复在位）。

| v2 修改（提交） | 作用 | v1 / 上游对应 | 状态 |
|---|---|---|---|
| `force_eplb.py` 经 `_EXTRA_CTX` 读取 + None 透传契约（8746dc38b，透传逻辑承 0deca3181，**最终态保持**；期间 0ec71c078 曾禁用、1e9fe4912 撤销） | MRV2 的 comm method 存于 forward_context.additional_kwargs，裸属性读法会 AttributeError；经 `_EXTRA_CTX` proxy 读取使 force EPLB 在 MRV2 可用。**乱码是预期行为（用户澄清）**：force EPLB 是理论性能上限验证工具，强制均衡拆分请求到每个 EP、故意破坏输出语义——MRV2 路径对该目的工作正常 | `_EXTRA_CTX` proxy（ascend_forward_context.py L428-488）统一 v1/v2 访问（v1 走 getattr(ctx)、v2 走 additional_kwargs）；透传契约修复 8746dc38b 自身的 None bug | ✅ |
| ~~`moe_comm_method.py` 形状限定注册表 + `activate_moe_comm_method` 逐层 rebind~~ **已回退（219009f60，2026-09-23）** | 原意防 target/drafter 异形共存；核实 DSV41 全家族同 checkpoint 同构（MoE 层共用 `DeepseekV41MoE`，shape key 恒同）→ 注册表恒休眠（单形状 identity 快路径），违背最小修改原则 | **本轮新增，v1 无对应**；v1 同为全局单例设计且 DSV41-on-v1 正常，佐证无两形状场景 | ↩️ 回退 |
| `fused_moe.py` L141 `setup_moe_comm_method(self.moe_config).get(ALLTOALL)`（O3 形态） | runner init 绑定 placement buffer：setup 用**本层 config 重建**实例并返回 → 绑定天然按本层形状，结构上消除基线"读全局单例（最后注册生效）"的错绑窗口 | 基线为 `get_moe_comm_method(ALLTOALL)` 读全局单例 | ✅ |
| `token_dispatcher.py` EP gather 后 `torch.npu.synchronize()`（L636 新增）+ 第二同步 elif 化（L655 存量） | HCCL 独立 comm stream 与 D2H 的序问题（draft dummy 路径观察到全零 output_splits） | **共享 ops 代码（v1/v2 共用）**；第二同步为存量（63c363d3d 时代），首个为本轮新增；性能收窄（event wait）已列 P3 | ✅ |

### 4.1 逐文件详解与优化评估

**① force_eplb.py（0deca3181 透传 → 8746dc38b 补 _EXTRA_CTX；0ec71c078/1e9fe4912 禁用往返已净零）**
- 作用：让 force EPLB（理论性能上限验证工具，强制均衡拆分 EP、输出乱码为预期）在双 runner 下工作。
  `get_force_eplb_topk` 经 `_EXTRA_CTX` 读 comm method（MRV1=ctx 属性 / MRV2=additional_kwargs）；
  无 context（AssertionError）或 comm method 为 None 时透传原 topk_ids（替代旧 `return None`——
  调用方 `topk_ids = get_force_eplb_topk(topk_ids, ...)` 直接回写，None 必崩）。
- 优化评估：无需优化。`build_force_eplb_topk` 的裸属性读法仅 MRV1 调用（L4178），MRV2 不调用，保持原样。

**② fused_moe.py L139-140（c505a4427 回退至基线形态）**
- 终态：`setup_moe_comm_method(self.moe_config)` + `alltoall_comm = get_moe_comm_method(ALLTOALL)`
  （基线原样，get 紧跟 setup 必得本层刚注册实例，placement buffer 绑定正确性不变）。
- 演化：基线两步 → 641ba3d22 形状注册表双查 → 52c6ddac5 O3 返回值合一 → c505a4427 随最小修改
  原则回归基线（O3 收益仅 init 期一次查找，可忽略；注册表回退后属纯 API 偏离）。

**③ moe_comm_method.py（641ba3d22 引入 → 219009f60/ca9e7c0cd 回退 → 247bc9890 恢复，v1 无对应）**
- 设计：形状限定注册表（`_CONFIG_KEY_FIELDS` 9 字段签名 + `_MoECommMethodsByConfig` 双注册表 +
  `activate_moe_comm_method` 逐层 rebind），防 target/drafter 异形共存。
- **回退实验结论（NPU bisect，2026-09-23）**：回退节点（ca9e7c0cd）拉起失败——all2all
  `split sizes doesn't match total dim 0 size`；注册表在位节点（af7606669）正常。证明逐层 rebind
  承重、"单形状恒等休眠"的静态分析（同 checkpoint 同构推演）被实测证伪——真实运行时序下存在
  未定位的配置分叉（候选：setup 逐层重建 vs 注册表去重复用导致的实例/快照错位）。
- **终态：恢复注册表 + rebind（247bc9890）**。根因分叉字段待插桩定位（setup 打印 9 字段 key、
  prepare 处对比层 config vs 发布实例）；定位后可评估把防御性 rebind 收敛为启动期断言。

**④ routed_experts.py（247bc9890 随注册表恢复 rebind）**
- 终态：prepare 前 `activate_moe_comm_method(_EXTRA_CTX.moe_comm_type, self.moe_config, ...)` rebind
  到本层形状实例（NPU bisect 证明承重，见 ③）+ mypy assert（952a3085e 内容随恢复回归）。

**⑤ token_dispatcher.py（641ba3d22 引入 → 59928911f 临时移除，A/B 验证中）**
- 原修复：`_preprocess` 中 EP gather（独立 HCCL comm stream）与 CPU 读 split 直方图之间的序保证——
  去掉 `input_splits` D2H 的 `non_blocking=True` + gather 后 `torch.npu.synchronize()`（修复 draft dummy
  路径全零 output_splits）；`num_local_experts<=1` 分支的同步 elif 化（避免非 EP 场景白同步）。
- **当前状态**：59928911f 临时恢复基线形态做 NPU A/B 验证——失败复现则 revert 该测试提交恢复修复；
  无失败则保留回退并在此记录结论。优化评估（P3）：全设备同步收窄为 event wait，待 NPU 实测。

### 4.2 优化结论汇总

| 编号 | 项 | 判定 |
|---|---|---|
| O1 | activate 守卫的 distinct-shape 集合重算 → setup 写路径维护计数 | **已剥离（e59a51dcf，最小修改原则）**——rebind 承重但 O1 属搭车优化，activate 回归注册表重算单形状检查（每次 MoE 层前向 µs 级集合重建，可接受） |
| O2 | token_dispatcher 全设备同步收窄 | 已录 P3，待 NPU 实测（修复本体另在 A/B 验证中，59928911f） |
| O3 | fused_moe.py setup+get 双查合一 | **已剥离（e59a51dcf）**——返回值无消费者（fused_moe.py 基线两步形态、activate 忽略返回）；fused_moe.py 保持基线形态 |
| O4 | sequence_parallel `sp_pad==0` 时跳过 `torch.cat` 整表拷贝（shard/reduce_scatter/padding_mask 三处） | **✅ 已完成（上游同款搬运）+ ✅ NPU 实测（2026-09-24）**：`if sp_pad > 0:` 守卫 + `F.pad` 任意 ndim（照搬 `vllm.models.common.ops.sequence_parallel`）；no-pad 切片 `clone()` 遵守 functional custom op 非视图契约（上游 `sequence_parallel_chunk_impl` 同款）；decode 场景 shard/padding_mask 降为本 rank chunk 拷贝、reduce_scatter 全免整表拷贝。**回退已被 NPU 证伪**：单独回退 O4 会导致 reduce_scatter padding 形状不对齐——rebase 后调用方（上游新 SP 形态，#17165 已移除冗余拷贝）与旧无条件 cat impl 不自洽，O4 与新调用方是绑定组合，不可拆 |
| — | proxy 双读 / SP 注册方式不一致 / MTP 双 gather | 不动（收益不足、侵入性大或为契约内在要求） |

## 五、models

| v2 修改（提交） | 作用 | v1 / 上游对应 | 状态 |
|---|---|---|---|
| `deepseek_v41/model.py` MTP hidden buffer 条件预分配 + collapse 前 gather stash（e125401b8） | draft 消费 collapse 前 full HC 态；仅 spec+last-PP 分配 | **对齐 deepseek_v4/model.py 同款模式（0deca318 内 L986-996，注明的上游参照为 vllm PR #50312）**；v1 的 V4.1 版原为 lazy None（框架未完成对齐），本轮补齐 | ✅ |

## 六、patch 系（本轮收纳）

| 提交 | 内容 | 归属 |
|---|---|---|
| 0a8290db6 | `_install_v41_eager_fallback` 从 model_runner.py 移入 `patch_v2/patch_set_forward_context.py`（import 时生效、无 global、理由同址）；patch 仅拦截 `vllm.v1.worker.gpu.model_runner` 命名空间，MRV1（ascend_forward_context 自带 import）不受影响。**后继：整个 fallback 已删除（b9868130a，NPU 实测 2026-09-24），见 2.1 表首行** | 本轮优化（非移植，已终结） |
| 2dffca268 / 67ad3be3e / a8d65b120 | dsa_v41 调试日志清理 / worker/v2 tidy（force_eplb import 提顶+模块属性访问、static_forward_context 出循环、full_graph_mode 接线）/ CP forward 删除 | 本轮优化 |

## 七、待核与遗留清单

1. ~~**零宽选择 fill(-1)**：v1 侧对照待核~~ **✅ 已核销（2026-09-23 rebase onto 3fc3db0e0）**：新上游基线自身已合入同款修复（`selected.shape[1] == 0` 守卫 + aclnnInplaceCopy 161002/FULL capture 注释），99218bcf9 的重复实现按 HEAD 侧解决，功能由上游版本承担。
2. **moe 形状注册表**：v1 下的暴露面差异待深核（⏳）——v1 构造顺序是否同样触发 legacy global 错绑
3. **P3**：token_dispatcher L636 全设备同步收窄（event wait），待 NPU 实测（见项目记忆）
4. **P3 并列**：v2 engram hook 每步阻塞 D2H（default.py L74），host-master 架构差异所致，修法评估已入项目记忆
5. **P4（用户暂缓）**：moe_comm_method 注册表 / eager fallback / `_prepare_v41_dummy_ring_state` 三块零覆盖 UT
6. **UP 审计遗留**：`_adjust_dsv4_kv_layout` 命名、v1 `runner_only_attn_layers` 层集合等价性（见附录 A）
7. **force EPLB 定性澄清（用户澄清，非 bug）**：force EPLB 是理论性能上限验证工具——强制均衡拆分请求到每个 EP、故意破坏输出语义，乱码/胡言乱语是**预期行为**，不构成精度问题。曾误判为 bug 并禁用（6b7918f9e），已撤销恢复（5838f7a29）。
   - 测量有效性注记（非 bug，使用工具时留意）：① `log2phy` 非恒等（V4.1 冗余专家/EPLB 重平衡激活）时，round-robin 的逻辑 ID 只覆盖每 rank 的逻辑子集——EP 分配仍均匀（col 均匀分布），验证目的达成，但"上限"精确度略受影响；② `mix_placement` 下 shared expert 列（L611-626 concat）被 force 表整表覆盖——改变 work 分布，测上限时需知悉。
8. **P3（中优先级，结构性方向）**：**dspark 去装饰，对齐上游 V4.1 零 dynamo 架构**。上游 e77daef89（#56214）的解法是结构性回避：V4.1 全家族（target nvidia/amd model.py、draft amd/dspark.py）**零 `@support_torch_compile`**（对照 V4.0 cpu/model.py L452 有装饰器，V4.1 有意全去）→ 无 compiled wrapper 即无需绕过；"有状态 kernel 与图捕获共存"改由 breakable cudagraph 解决——`eager_break_during_capture` 断点（上游 attention.py L624/L805）+ e77daef89 为 `QuantizedActivation` 补的 `_weak_ref_capture_arg` weak-ref + per-backend `_cudagraph_support` 声明（sparse MLA=UNIFORM_BATCH、compressor=ALWAYS）。
   我们侧基础设施大半就绪：dsa_v41.py L51 `dsa_v41_forward` 断点已挂、MRV2 `torch_cuda_wrapper` 已向 breakable_cudagraph 注入 `weak_ref_tensor`；**差 dspark.py L325 去装饰 + NPU 验证**。
   达成后可连带精简：删除 `patch_v2/patch_set_forward_context.py`（无 wrapper 即无需绕过，其存在理由随之消失）；评估 SP custom op 的简化（5cf056b01 的 dynamo 不透明包装专为伺服 dspark 装饰器而存在，纯 eager 下 custom op 走真 impl，理论可退化）。
   验证项：① dspark 的 ACL graph 捕获走 breakable full 路线的可行性（上游 GPU 已验证同路线，ascend 钩子已挂）；② SP custom op 纯 eager 行为；③ draft 捕获粒度与上游 UNIFORM_BATCH 的等价物。
   注意因果链：去装饰是结构性决策，牵动 SP custom op / patch / MR notes 多处，需整体评估后单项实施。

## 八、上游社区对照核实（9b959b865 / e77daef89）

对照上游 vllm 提交 `9b959b865`（DeepSeek-V4.1-Flash 模型定义，#56228）与 `e77daef89`（runner/kv_cache 接线，#56214）核实本分支方案，2026-09-23 完成。

### 8.1 逐项核实结论

| 核实项 | 上游做法 | 我们的做法 | 结论 |
|---|---|---|---|
| **MTP hidden buffer** | `nvidia/model.py` L495-501 条件预分配、L663-668 stash、L676 `if self.use_sequence_parallel and self._mtp_hidden_buffer is None:` | e125401b8 与上游**逐字同构** | ✅ 印证 |
| **SP pad 守卫（O4）** | `sequence_parallel.py` `sp_shard`/`sp_reduce_scatter`/`sp_padding_mask` 均为 `if sp_pad > 0:` 才 pad（`F.pad`，任意 ndim）；custom-op 版 `sequence_parallel_chunk_impl` 另有 no-pad `clone()` 非视图契约 | O4 已按上游同款搬运落地（`if sp_pad > 0:` + `F.pad` + no-pad `clone()`），保留 custom op 包装 | ✅ 印证并已实施 |
| **SP custom op 包装** | **无** custom op 包装（纯 Python 函数） | 三处 custom op（dynamo 不透明 + 上游 pad 错轴修复） | ✅ ascend 特有需求——dspark.py L325 有 `@support_torch_compile` 而上游 deepseek_v4_1/ 全目录 grep 为空，dynamo 边界差异是根因；上游 V4.1 的终态方案（零 dynamo + breakable 捕获）与我们的对齐方向见遗留清单 8 |
| **model_state 结构** | `nvidia/model_state.py`（98 行）：`prepare_inputs`/`prepare_dummy_inputs` 两段式；engram lookback 用设备侧 Triton `_gather_lookback_kernel` 从 `all_token_ids.gpu` gather（零 host 同步） | AscendModelState engram hooks 同源结构；数据源不同——上游 token lookback 设备侧 gather vs 我们 block table host 侧 | ✅ 结构同源，数据面差异由 cache 架构决定（见 8.2） |
| **dummy run 状态冻结** | `prepare_dummy_inputs` 对 persistent buffer 做 `-1` 填充；`_init_model_kwargs(num_reqs=0)` dummy 路径 | `_prepare_v41_dummy_ring_state` + `skip_ring_state_update` ContextVar | ✅ 同类问题不同机制（上游无 dynamo 重追踪压力） |
| **环形缓冲分组** | e77daef89 `kv_cache_utils` `_get_packed_kv_cache_groups` 的 `is_state_bucket` 增加 `CircularBufferSpec` 分支 | 我们的环形分组处理（V4.1 compressor 复用） | ✅ 印证 CircularBufferSpec 归 state bucket 的定性 |
| **lookback 接线** | `lookback_token_ids`（CpuGpuBuffer + `_prepare_lookback_token_ids` numpy gather）；breakable_cudagraph 增 `QuantizedActivation` weak-ref 处理 | 无对应物（ascend 无 breakable_cudagraph QuantizedActivation 路径） | ✅ 无需适配；host 镜像对照已入遗留清单 4 |
| **skip_compiled 补丁** | **无**（上游模型/runner 无此需求面） | patch_v2 eager fallback（0a8290db6） | ✅ MRV2 继承 GPUModelRunner 的 dynamo 边界特有 |

### 8.2 ca9cb2fec（block_size 恢复）与上游双粒度机制对照

上游在 `v1/worker/utils.py` 建立了系统的双粒度保护，与我们 RCA 结论**同源同向**：

1. **spec 层**：`MLAAttentionSpec.storage_block_size`（kv_cache_interface.py L560）注释即双粒度契约——"Token width used to view storage when it differs from the kernel block"；DSV4.1 经 `tokens_per_state=compress_ratio`（attention.py L945）携带压缩比，sparse_mla.py L31-35 注明 per-layer 存储页 = `block_size // ratio`。
2. **builder 层**：`create_metadata_builders`（utils.py L276-288）——re-block 时若 spec 有 `storage_block_size`，builder 拿到 `copy_with_new_block_size(storage_block_size)` 而非任意 kernel size；即**上游拒绝让 builder 收到会抹平压缩比的粒度**。
3. **cache 视图层**：utils.py L447-448 `kernel_block_size = spec.storage_block_size` 同款覆盖。
4. **kernel 选择层**：`select_common_block_size`（utils.py L328+）保证选出的 kernel size 是 manager block_size 的因子且满足后端 `get_supported_kernel_block_sizes()`（DSV4.1 = [64/128]，`MultipleOf` 约束）。
5. **同类失效面上游有案底**：`glm5next/nvidia/attention.py` L96-99 注释——小 block 会 "silently collapses storage_block_size (64//16=4) and only fails later at the opaque C++ assert"，上游在 `get_kv_cache_spec` 前置 assert 守卫。**与我们 NaN RCA 是同一失效类**（块粒度塌缩 → 比例抹平 → 寻址错误），上游的教训是"失败点远离根因、必须前置守卫"。

我们的 ca9cb2fec 在 `AscendDSAV41MetadataBuilder.__init__` 恢复 `block_size=cache_config.block_size` + `storage_block_size=logical//ratio`：与上游同一条不变量（**双粒度契约必须穿越 re-block 存活，builder 不得收到抹平比例的粒度**），实现方向相反但等价——

- 上游：builder 以**存储粒度**工作（逻辑→存储换算在 block-table 层下方完成）；
- 我们（承 MRV1 结构）：builder 以**逻辑粒度**工作（换算在压缩算子内用 ratio 完成）。

两者都让 ratio 对 builder 消费者可见，属"随执行栈换挂载点"模式的合法分叉。另外我们 `_reshape_kv_cache_v2`（attn_utils.py L1061-1071）的 storage!=logical 时取 storage、`indexes_kv_by_block_stride` 时再除 ratio，与上游 L447-448 覆盖语义一致。

### 8.3 核实遗留（不阻塞）

- 上游 DSV4.1 spec 未显式设 `storage_block_size`（走 `tokens_per_state` 隐式路径）；ascend spec 显式设。二者在各自管线内自洽，无需对齐。
- 上游 `glm5next` 式"spec 创建期前置 assert"守卫，我们可在后续为 AscendMLAAttentionSpec 增加同款（P4 级，可选加固）。

## 附录

- **附录 A**：`docs/dsv41_mrv2_attn_utils_porting_map.md` —— attn_utils.py 六块逐行对照（含 v1 行号、element_size 泛化差异、UT 映射）
- **行为对照文档**：`docs/dsv41_mrv2_behavior_review.md`、`docs/dsv41_mrv2_context.md`（既有）
