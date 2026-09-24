# DSV4.1 MRV2：rebase 后的上游对齐分析

分析日期：2026-09-24。结论来自两个本地仓库的实际代码；未运行 NPU 推理。

## 版本和交付范围

- 当前分支：`main-mrv2-dsv41-0922`。
- 原 HEAD：`1213e9104cff4c3ef52960573b754463dbd60c49`。
- 原分支备份：`codex/backup-dsv41-before-rebase-20260924`。
- fetch 后 Ascend main：`5a84871b22e968333b3451f70a93ac58734d996b`。
- 32 个原分支提交已完成 rebase，迁移后 HEAD：`071ebf4d72ef4f428174da9b6f4b74bec3e0fa14`。
- 最终 HEAD：`aba1034b3`，额外一个带 sign-off 的导入清理提交，修复 rebase 后多余 import 和原分支 import 排序问题。
- 上游 vLLM：`C:/code/vllm`，HEAD `ced6857afa`，使用用户已经更新的 checkout，未修改该仓库。
- 相对新 Ascend main：22 个文件，888 行新增、52 行删除，其中包含 9 个测试文件。
- 没有 push，没有实施下述优化或接口修复。原有未跟踪文件保留。

冲突解决保留 main 的 PD recompute 导入、`valid_state_slots` 新接口、LM-head TP/EPLB 行为和细粒度 TP 测试初始化。SP reduce-scatter 保留 main 已有的对齐时免拷贝实现，未改成另一套等价 padding。已用 `range-diff` 核对迁移的 32 个提交。

## 结论与优先级

最有收益的收敛方向是：沿用上游模型自定义 ModelState、设备端 lookback 和 layer cache binding 接口，保留 Ascend 算子所必需的缓存视图、metadata 和图模式适配。不要把替换整个 DSV4.1 模型实现作为本轮 MRV2 的前提。

| 优先级 | 项目 | 建议 | 判断依据 |
| --- | --- | --- | --- |
| P0 | Engram 参数与数据来源 | 删除旧 CPU history hook，接设备 metadata 和上游 lookback | 当前调用与模型签名已明确不匹配 |
| P0 | KV cache 绑定入口 | 把 DSV4.1 绑定接到新版入口；优先局部 layer 方法 | 旧入口特判已不在 MRV2 活跃调用链 |
| P1 | MoE 通信状态 | 维持 target/draft 隔离，修复旧 getter 调用契约 | 一参数调用对当前 shape key 返回 None |
| P1 | DSpark 模型接口 | 复用已有上游 loader，补模型能力声明并核实权重语义 | 概率采样直接读取当前模型未声明的属性 |
| P2 | ModelState 职责 | 将 DSV4.1 逻辑从通用 AscendModelState 移入专用 state | main 和上游均已有模型提供 state 的入口 |
| P2 | dummy ring | 借鉴 valid_state_slots，保留 ring 专属 reset/skip 语义 | 上游通用方法不覆盖清零和 idle-DP 保护 |
| 保留 | 缓存布局与 attention metadata | 保留当前必要桥接，避免全面套用 GPU 布局 | NPU 物理页面、indexer scale 和框架逻辑块仍不同 |
| 保留 | SP、MTP hidden、eager fallback | 保留必要边界，减少重复说明，不机械删除 | 上游支持模型不等于 NPU 编译路径已等价 |

## 1. Engram：先修接口，再收敛到专用 ModelState

现状：

- `vllm_ascend/worker/v2/model_states/default.py::_get_engram_history_inputs` 每步查 group、将 block table `.cpu()`，构造旧 history tuple。
- 同文件 `prepare_inputs` 仍调用 `prepare_engram_inputs(..., history_inputs=...)`。
- 新 main 的 `models/deepseek_v41/model.py::prepare_engram_inputs` 已改为 `lookback_token_ids / query_start_loc / slot_mapping / block_table`，不接受 `history_inputs`。这是确定的参数契约错误，Engram hook 进入时会报 TypeError，而非仅有性能差异。
- Ascend MRV1 已通过 `_get_engram_device_inputs` 使用设备端 metadata；本分支 MRV2 没有跟上这次改动。

可复用上游：

- `C:/code/vllm/vllm/models/deepseek_v41/nvidia/model_state.py::DeepseekV41ModelState` 已实现固定地址 lookback buffer 和 `_gather_lookback_kernel`。
- 它通过 `idx_mapping`、`req_states.num_computed_tokens.gpu`、`req_states.all_token_ids.gpu` 收集每请求 chunk 起点之前的 token；capture 时填 -1，正常执行时原地刷新。
- `vllm_ascend/worker/v2/model_states/__init__.py::init_asecnd_model_state` 已优先使用模型的 `get_model_state_cls()`，无需再给通用 runner 增加模型名分支。

最小方案：

1. 由 DSV4.1 模型提供专用 Ascend ModelState，继承 `AscendModelState`，保留 NPU 的 `prepare_attn`。
2. 复用或薄封装上游 lookback gather。该实现位于 nvidia 包，但函数本身是设备 Tensor + Triton；直接 import 前核对导入依赖并做 NPU 验证，不应因此整体继承 GPU 的 attention 准备路径。
3. 从当前 batch 和 Engram 对应 group 传入设备 `query_start_loc / slot_mapping / block_table`，沿用 MRV1 的完整请求坐标约定。PCP 时不可直接把 rank-local query 边界当全请求历史。
4. 继续在 replay 之前调用 Ascend 的 `prepare_engram_inputs`，capture 只绑定 `prepare_engram_graph_inputs` 的持久 buffer。
5. 删除通用 state 中旧 history helper、CPU 拷贝和仅为旧 history 存储的字段；专用 state 缓存稳定 group id，而非每步扫描。

不能直接只返回上游的 lookback：Ascend 还有图外 Engram lookup/DP collective 和图内固定 buffer 消费约定，必须保留。Idle DP 要参加需要的 collective，但不能更新真实请求历史。异步投机下 lookback 不得包含尚未确认或已拒绝 token，需专门测。

现有 `test_engram_state_hooks.py` 用无签名约束的 Mock，并断言旧 `history_inputs`，因此无法暴露真实签名变化。应改为有真实签名的 stub/autospec，加跨 chunk、prefix hit、请求重排、idle DP 和 reject 回归。

## 2. KV cache：旧特判需要迁移到活跃入口

实际调用链：

`GPUModelRunner.initialize_kv_cache` → 上游 `attn_utils.init_kv_cache` → `bind_kv_cache_to_layers`。

Ascend main 的 `patch_v2/patch_attn_utils.py` 已将这个入口换成 Ascend 的同名函数。然而本分支增加的 `[kv_cache]` 特判只放在旧 `patch_bind_kv_cache.py::bind_kv_cache`。

后果：新入口直接赋值 `layer.kv_cache = allocation`；`DeepseekV41CacheLayer` 及其消费者仍使用 `kv_cache[0]`。普通 cache 可能因此丢掉 block 维，indexer tuple 也不再保留原先的整体结构。仅 rebase 不会自动修复这种语义失配。

建议：

- 最小修复是在新入口逐层识别 DSV4.1 cache，只为这些 layer 保留 list 包装；避免旧代码的“任意一层是 V41，则所有层都包装”逻辑。
- 更贴近上游的方式是在 `DeepseekV41CacheLayer.bind_kv_cache` 定义自己的接收约定，新入口对该类调用方法。上游 `worker/utils.py::bind_kv_cache_to_layers` 本来就是逐层调用 `bind_kv_cache`，V41 compressor/indexer 也各自实现自己的视图转换。
- 不应立即把 Ascend 所有 layer 切为上游默认 binder：main 仍需兼容其他模型的 `(k, v)` allocation。
- 清理已无 MRV2 作用的旧特判前，确认是否还有本地旧入口调用方；MRV1 本身使用独立绑定流程，不能推测两者都走同一条路径。

验证必须经过新的 `init_kv_cache` 入口，覆盖 Tensor cache、indexer tuple、混合 layer、共享 backing 的 data_ptr/stride/offset，以及 ring cache 第一维仍为 num_blocks。

## 3. MoE：缩小公共 API 影响，比增加更多 key 更重要

当前 `moe_comm_method.py` 将 registry 从 `MoECommType` 改成 `(type, expert counts)`，但 `get_moe_comm_method(type)` 的旧调用仍在：

- `platform.py` 的 forward-context 构建；
- `ascend_forward_context.py`；
- `fused_moe.py` 的 all-to-all 查询；
- target 和 DSpark 模型初始化时的 `moe_comm_methods` snapshot。

不传 config 时 `_moe_config_key` 生成 `(type, (0, 0))`，一般无法命中真实专家配置。`routed_experts.py` 在执行时重新获取实例，能覆盖部分路径，但不证明前面的调用安全。该契约问题原分支已存在，不是 rebase 新引入。

新 main 已提供模型级 `moe_comm_methods` 和 forward-context 的 `model_instance` 路径，可作为收敛基础；MRV2 还需核对其上游 forward-context + platform extras 是否确实取得当前 target/draft 实例，不能直接假定 V1 的路径已覆盖 MRV2。

建议优先让每个模型/层持有正确配置的通信对象，复用 main 的选择通信类型逻辑，避免在通用 getter 上留下不兼容的默认参数。若短期保留 shape registry，则必须保留一参数调用兼容行为，并测 target/draft 不同专家数、其他 MoE 模型、EP=1/EP>1 和 capture/replay 切换。不要直接删除隔离，否则会恢复 target/draft 共用错误通信状态的风险。

`force_eplb.py` 改读 `_EXTRA_CTX` 有 MRV2 上下文依据，应与 registry 重构分开判断；上游为 draft 禁用 EPLB，不等价于 Ascend 的强制均衡 dummy routing，可以继续复用上游 draft 配置策略而保留 NPU 必要逻辑。

## 4. DSpark：沿用上游 loader，补齐模型声明

本分支相对 main 没有额外修改 MRV2 DSpark speculator。main 的 `AscendDSparkSpeculator.load_draft_model` 已调用 super，并执行 NPU `post_process` 和 target aux capture 配置；不应再复制一套 loader。

上游 `spec_decode/dspark/utils.py` 已处理 target backend 继承、draft quant config、PP-safe loading、draft EPLB 关闭及 embedding/head 共享。

需复核的模型契约：

- 上游 DSV4.1 draft 声明 `has_own_embed_tokens=False`、`has_own_lm_head=False`、`draft_id_to_target_id=None`。
- Ascend `DSparkDeepseekV41ForCausalLM` 未声明这些属性。概率草稿路径中，上游直接访问 `model.draft_id_to_target_id`；按当前类定义会产生属性错误，除非运行前另有代码注入。
- 缺少 own-weight flag 时，上游 `_should_share` 默认共享；必须依据 Ascend checkpoint/量化后处理契约决定 True/False，不能为了与上游文本一致而盲目设 False。
- 对全词表 draft 显式声明 `draft_id_to_target_id=None` 是最小接口对齐；若实际使用缩减词表则需要真实映射。

这属于新上游兼容审查发现，不是要求扩大本次 rebase 的提交范围。测试需包含真实权重加载后 embedding/head、greedy/概率采样、PP、draft aux 层编号。

## 5. 已与上游对齐或仍应保留的部分

### MTP hidden buffer

分支新增的预分配、last-PP-rank 条件、保存 collapse 前完整 HC states，以及存在 buffer 时先 gather 的策略，与新上游 `nvidia/model.py` 的实现基本一致。应保留，不宜为了少 32 行删除。它与 DSpark 的 mean-pooled aux states 是两个接口，不能互相替代。

上游在非 MTP 时先本地 norm 再 gather，Ascend 当前 gather 后 norm；数值等价性可另测，但这不是消除本分支 MRV2 增量的必要改动。整段继承上游 NVIDIA 模型反而会扩大影响面。

### Sequence parallel

上游 common ops 已有 padding guard、任意 trailing dims 的 dim-0 shard 和 custom collective fallback，但仍是普通 Python 函数，没有本分支阻止 Dynamo 固化 shape 的 custom-op 边界。

保留 opaque custom op、fake shape 和无 padding 分支的 clone；不应直接替换成上游 `sp_shard`，否则重新引入动态图 bucket/别名问题。可减少重复说明，保持内部数学与上游一致。main 已有 reduce-scatter 对齐时免拷贝，本次 rebase 保留了它，避免再产生等价的 padding 改写。

### 缓存分配、逻辑块和 metadata

- 新 main 的 `allocate_kv_cache_main` 仍调用 Ascend `_allocate_kv_cache` 和 `_reshape_kv_cache_v2`，所以本分支 V41 分配/reshape 分支仍在活跃路径，不是旧 API 遗留死代码。
- V41 使用每个物理 slot 一块 backing、`block_stride` 页面步长、long-KV/indexer plane offset 和不同 scheduler group 的叠放。GPU 的通用视图不能直接代替 NPU tuple/scale 视图。
- 上游 `AttentionGroup.create_metadata_builders` 仍会用 MLA `storage_block_size` 重建 builder spec。Ascend V41 builder 消费逻辑 token block，因此现有恢复逻辑 block size 的防护不能直接删除；更规范的长期方向是明确传递两种尺寸。
- `num_actual_reqs` 和 padded request count 的区分仍必要，当前 `_request_counts` 回归修复应保留。
- group 内共享 slot 坐标、跨 group 共享 batch metadata 的边界应保留；不能为减少代码合并为一个全局字典。
- `prepare_source_rope` 只初始化 RoPE，不能为了复用 MRV1 helper 而开启其异步 device-metadata 任务机制。

### Dummy ring 与 eager fallback

上游已有 `prepare_dummy_attn(..., valid_state_slots=True)`，可借鉴其 request-slot 生成。但它对所有 group 设置 slot，且不负责 ring 清零、容量校验和 `skip_gdn_state_update` 的 idle-DP 保护。Ascend 的共享 slot 布局需要保留 group 选择，不能整段替换后声称等价。

`patch_set_forward_context.py` 的 runtime-NONE eager fallback 仍有作用：新上游 `skip_compiled` 主要处理 encoder 输入，不会仅因 graph mode=NONE 就跳过编译包装。保留为模型限定的小补丁，待 Ascend compressor/indexer prefill 的编译路径实测通过后再删。上游 GPU 能编译不是 NPU 路径通过的证据。
> **✅ 后续（2026-09-24 NPU 实测）**：删除条件已达成——移除该 fallback（b9868130a）并保留 O4 后 V4.1 功能精度正常，compressor/indexer prefill 编译路径在 compiled wrapper 下可用，patch 已删且不恢复 runner 内联版。同轮证伪：单独回退 O4 会导致 reduce_scatter padding 形状不对齐（rebase 后上游新 SP 调用方与旧无条件 cat impl 不自洽），O4 必须保留。

## 推荐实施顺序与验收

1. 修复新 cache binding 和 Engram 参数，确保新 main 上 MRV2 能进入真实 forward。
2. 同一轮把 Engram 接到专用 ModelState，复用上游 lookback，删除旧 CPU history；避免先追加一套兼容层再长期保留。
3. 独立收敛 MoE registry 公共 API，确认 target/draft 隔离与其他模型兼容。
4. 补 DSpark 模型契约，复用已有 loader，不扩大 speculator 改动。
5. 用 NPU 证据决定是否进一步精简 SP/dummy/eager fallback；其余必要适配先保留。

最低回归集合：无投机/DSpark/MTP，eager/FULL decode，多 capture bucket、TP 不整除 token、EP/DP idle rank、prefill→decode、chunked prefill、prefix hit、reject/请求重排、PCP 配置。真实权重需校验输出与 MRV1/基线一致；dummy 启动不能证明正确性。

## 本次验证

- 已确认 `vllm-ascend/main` 是最终 HEAD 的祖先；rebase 无遗留冲突，已跟踪工作区干净。
- 22 个变更 Python 文件：AST 解析、Ruff check、Ruff format check、git diff --check 通过。
- 不依赖 torch 的 AST 提取小实验验证了三个局部事实：真实 Engram 函数签名拒绝 history_inputs；新版 binder 直接赋 raw allocation；shape registry 的旧一参数查询不命中已注册对象。这些不是整模型运行测试。
- 尝试运行 `test_engram_state_hooks.py`、`test_dsa_v41_request_counts.py`、`test_sequence_parallel.py`，在 conftest 导入 torch 时失败：`ModuleNotFoundError: No module named 'torch'`。
- 尚未执行完整 UT、NPU 真实权重/性能验证或 `bash format.sh ci`；本次未 push。不能把静态检查通过解释为模型可运行。

本文是分析交付，接口修复和优化尚未落地；上述 P0 问题意味着当前 rebase 后分支还不能据此宣称适配完成。
