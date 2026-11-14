# K3.1 A3 MLA tiling 下沉：接口 mock 与框架接入方案

初版日期：2026-11-13；PR 整理日期：2026-11-14。当前基线：vllm-ascend `76ea9d24f`，包含 #21542。
PR 分支：`codex/a3-mla-tiling-pr`。评审入口：[Draft PR #21876](https://github.com/vllm-project/vllm-ascend/pull/21876)。

面向同事的简版说明见 [当前方案同步说明](a3_mla_tiling_sink_summary.md)。

## 1. 目标与交付

针对 K3.1 的 MLA DSpark 草稿，接通设备长度、设备调度 metadata、注意力计算、DCP 结果合并及图缓冲区复用。算子尚未交付，因此使用显式启用的数值 mock，并提供独立的 external provider 适配点。

交付内容：完整方案、可替换的算子适配层、数值 mock、框架接入、CPU 回归测试、独立数值参考、测试报告和真实算子替换说明。默认配置不改变现有 A3 FIA 或 A5 FlashMLA 行为。

mock 验证接口和框架逻辑，不能证明真实 A3 kernel、ACL graph、HCCL、多机模型精度或性能。硬件验收项必须单独记录，不能用 CPU mock 的结果替代。

## 2. 固定接口依据

依据用户提供的 MinghuasLab/flash-attention-npu，以及 PR #338 的接口版本 `7d3f7433669b524ce71c4e867b90dc3bc6f41264`：

- `flash_attn_npu_3.flash_attn_npu_interface.flash_attn_with_kvcache`
- `flash_attn_npu_3.flash_attn_npu_interface.get_scheduler_metadata`
- Query：TND，`q[...,64]` 为 RoPE，`qv[...,512]` 为内容部分。
- Cache：ND 分页，`k_cache[...,64]`、`v_cache[...,512]`，一个 KV head；只读，框架先完成写入。
- 长度：同设备 contiguous int32 `cache_seqlens[B]`、含起始 0 的 `cu_seqlens_q[B+1]`。
- 显式传入模型的 `softmax_scale`，不能改成按 576 重新计算。
- `return_softmax_lse=True` 返回四项：out、FP32 LSE、out_accum、lse_accum；TND 主输出为 `[T,H,512]`，LSE 为 `[H,T]`。
- metadata 是不透明 int32 Tensor；长度、query 分段、mask 等变化时重新生成。
- 此版本 FP16/BF16 分页容量最多 16384，算子页大小为 16 的倍数且在 16～1024；不能把 128K 长上下文当成已支持。
- 区分管理块和算子页：K3.1 的 1536-token 管理块由现有 block table/cache view 转成 12 个 128-token 算子页。首版允许这种管理块，传给算子的 `page_size` 始终为 128，并检查实际缓存视图。
- 没有 `seqused_q` 入参。图补齐必须通过零 KV 长度、有效 token mask 和输出清零适配，不能假装接口有这个参数。

首版聚焦压缩 MLA decode / 多 token 草稿。192/128 展开 prefill、INT8、滑窗、softcap 作为不支持组合明确拒绝或留在原路径，不伪装实现。

## 3. 架构与调用链

```text
DSpark 设备输入准备（已有代码）
  → 设备实际 KV 长度、Query 分段
  → 共用 A5 的 FlashMLAMetadataBuilder 缓冲区生命周期
  → A3MLAAdapter.build_metadata
  → provider.get_scheduler_metadata
  → MLA 前处理与 KV 写入（保留 A3 原有缓存布局）
  → A3MLAAdapter.attention
  → provider.flash_attn_with_kvcache
  → TND/NTD、LSE 规范化
  → 原有 DCP 通信/合并与 V 投影
```

将 A3 provider 与 A5 FlashMLA 分开选择。A3 不冒充 `use_flashmla=True`，避免误启用 A5 的缓存分配或 PROLOG writer。复用的是设备 metadata 的组织方式和图缓冲区生命周期。

### 3.1 配置与选择

使用集中管理的 additional_config，不增加环境变量：

| 字段 | 默认值 | 用途 |
| --- | --- | --- |
| `a3_mla_sink_mode` | `disabled` | `disabled` / `mock` / `external` |
| `a3_mla_sink_module` | `flash_attn_npu_3.flash_attn_npu_interface` | external 模式导入的模块 |
| `a3_mla_sink_allow_graph` | `false` | 明确允许实验性的 graph 路径；不代表已经完成硬件验收 |

这些是 mock 和真实算子初期联调的临时入口。正式接入完成后，计划把 mock 限定在测试代码中，固定真实算子导入，并按硬件/模型/算子能力选择路径；graph 也纳入正常能力判断。当前代码仍依赖这些配置，不能直接删除配置字段而不调整选择逻辑。

- disabled：默认，使用原后端。
- mock：显式使用内置数值实现，供开发和测试；不可当成性能算子。
- external：动态加载符合上述两个接口的模块；缺少接口或不兼容时明确报错。

选择仅限 A3、K3.1、DSpark 草稿 MLA、V2 runner、支持的浮点与 ND 缓存。A5 和其他模型保持原路径。不支持的静态配置在分配缓存前处理，执行中发生错误不得偷偷回退后端。

### 3.2 A3 适配层

独立模块负责：拆分 576 维 Query、接收原 A3 分离缓存或临时 fused cache、映射参数名、固定容量查询、返回四元组校验、输出布局转换、有效 Query mask、FP32 LSE。

provider 的 metadata 内容不由框架解析。mock 使用自己的记录格式并检测长度/分段变化，防止测试误用旧 schedule。真实算子替换时仅替换 provider，内部记录格式不需要相同。

### 3.3 DCP 与图

沿用设备侧 DCP 局部长度计算及现有输出/LSE 合并。非因果 DSpark 读取本 rank 全部有效 KV；符合条件的因果路径保留 history/current 分拆。测试覆盖 rank 空 KV、跨页、不同长度、分片合并与完整注意力对照。

复用已捕获容量的地址，逐轮刷新长度、页表和 schedule；metadata Tensor 不得在 replay 中被旧长度固定。CPU 上测试地址稳定与连续多轮更新；真实 ACL capture/replay 在算子到位后单独验收。

### 3.4 CPU 长度同步策略

沿用 #21542 的同一谓词同时控制长度 D2H 和下一步 event wait。A3 草稿已使用设备长度只是必要条件。主模型每个消费者必须已经声明/具备设备长度能力，才能跳过同步；未知消费者、PP/PCP、sparse KV offload、自适应验证和不支持的 connector 保持原有复制与等待。

首版检查实际初始化的 model state 和所有目标 attention group，采用 `supports_spec_device_lengths()` 能力声明。目前声明范围为普通 SFA、DCP SFA、SFA indexer、非 PCP 的 Kimi KDA，以及不带 RecoverSSM 的 Ascend hybrid model state；未知子类必须重新声明。SFA 也不再为这些已审核消费者重建缺失的精确 CPU 长度镜像。

不会通过伪造 `device_metadata` 对象或仅检查模型名来移除同步。未知消费者、RecoverSSM 和 PD connector 的整轮同步继续保留；这是明确的首版边界。声明依据是源码消费者审计和隔离测试，完整模型运行仍需 H2 验收。

## 4. 数值 mock

mock 直接用 Torch Tensor 运算实现压缩 MLA：

`score = scale * (q_rope @ k_rope.T + q_content @ v_content.T)`

再应用实际 KV 长度和 bottom-right causal mask，计算 FP32 softmax/LSE，与内容 cache 相乘得到输出。对零 KV、无效 Query 输出 0 和 -inf LSE，KV cache 不可被修改。

热路径不使用 `.cpu()`、`.item()`、`.tolist()` 或按设备数值决定 Python 循环。循环仅允许使用 shape 等静态元信息。mock 有显式工作量上限，避免误用于大模型性能测试。

独立测试参考采用逐请求解页、独立 dense attention 计算，不复用 mock 的 gather/mask 辅助函数。

## 5. 实施顺序

1. 固定基线，建立隔离 worktree，先落此方案。
2. 实现配置、接口 provider、A3 adapter 和 mock。
3. 连接 MLA 选择、metadata builder、普通/DCP attention 和 DSpark 状态。
4. 连接保守的同步策略；保证同一个判断控制复制与等待。
5. 增加接口数值、metadata 生命周期、DCP 合并、框架选择和回退测试。
6. 执行测试、静态检查，检查完整 diff，输出补丁、运行记录及验证报告。

## 6. 验证矩阵

| ID | 检查 | 方法 | 性质 |
| --- | --- | --- | --- |
| V1 | FP16/BF16、不同 Query/head/batch、显式 scale | mock 与独立 FP64 dense reference 比较 out/LSE | 必须 |
| V2 | 非因果/因果、Q>K、零 KV、空/补齐请求 | 边界数值比较、NaN/Inf 检查 | 必须 |
| V3 | 页表重排、跨页、strided cache、只读 cache | 与解页 reference 比较，cache 前后相等 | 必须 |
| V4 | 拒绝数量、请求重排、连续轮次 | 改变设备长度并重建 metadata，检测旧记录误用 | 必须 |
| V5 | 四返回值、参数名/布局/dtype/容量 | 严格 fake external provider 与错误用例 | 必须 |
| V6 | DCP 1/2/8 的逻辑分片、空 rank、LSE 合并 | CPU 模拟多 rank 与完整 attention 对照 | 必须 |
| V7 | 图容量缓冲区稳定、动态长度刷新 | 重复 build/attention 比较地址与结果 | 必须；非真实 ACL graph |
| V8 | disabled/mock/external、A3/A5、能力/配置拒绝 | 真实选择函数或隔离加载方法测试 | 必须 |
| V9 | D2H 和 wait 同时取消/同时保留 | runner 方法回归，未知消费者必须回退 | 必须 |
| V10 | 热路径无 Host 长度读取 | 对 Tensor.item/cpu/tolist 等设禁用陷阱 | 必须 |
| V11 | 原 A5/FIA 行为与构建 | 相关回归、编译、Ruff、diff 检查 | 必须 |
| H1 | A3 真算子 eager/ACL graph | 算子交付后 NPU 对照 | 未具备条件，不冒充通过 |
| H2 | K3.1 完整模型、8机/DCP/PD | 真机模型精度、HCCL 与端到端测试 | 未具备条件，不冒充通过 |
| H3 | 空泡和性能改善 | 同负载交替 benchmark + profiler | 未具备条件，不冒充通过 |

## 7. 真实算子交付后的接入

1. 安装匹配 CANN/torch_npu 的算子包，把 mode 改为 external，填写模块名。
2. 如符号/参数不同，只修改 provider adapter，保持框架标准输入输出不变。
3. 确认 16K 容量、页大小、头数、strides、空 KV、LSE、图模式是否已扩展；按真实能力修改契约及测试，不能只删检查。
4. 重跑本方案的接口矩阵，再执行 H1/H2/H3。主模型消费者能力和同步开关一并验收。

## 8. 实际改动位置与原因

| 文件 | 改动及原因 |
| --- | --- |
| `vllm_ascend/attention/a3_mla_sink.py` | 选择规则、参数/布局适配、provider 加载、主模型设备长度能力检查；真实接口变化集中在这里处理 |
| `vllm_ascend/attention/a3_mla_mock.py` | 两个公开 API 的可计算 mock；不是占位返回值，能发现长度、分页、mask、LSE 错误 |
| `vllm_ascend/ascend_config.py` | 显式开发开关；默认 disabled |
| `vllm_ascend/attention/mla_v1.py` | 选择 A3 adapter；保留原 cache writer/V-up，允许分离缓存进入 external attention |
| `vllm_ascend/attention/flashmla.py` | 复用 A5 的设备长度、schedule、图缓冲区刷新；A3 从真实设备零长度输入初始化 metadata 容量 |
| `vllm_ascend/attention/context_parallel/mla_cp.py` | DCP 接受 A3 分离缓存，复用现有通信、历史/当前分拆与 LSE 合并 |
| `vllm_ascend/worker/v2/spec_decode/dspark/speculator.py` | 识别 A3 provider，停止为该路径读取 draft 精确 CPU 长度 |
| `vllm_ascend/worker/v2/model_runner.py` | 扩展已有同步谓词；同一个条件控制复制和等待，不能只跳过其中一个 |
| `attention/sfa_v1.py`、`attention/context_parallel/sfa_cp.py`、`attention/indexer.py`、`ops/kda_pcp.py`、`worker/v2/model_states/mamba_hybrid.py`（均在 `vllm_ascend/` 下） | 为已审核的消费者声明设备长度能力；未知实现保留同步 |
| `tests/ut/attention/test_a3_mla_sink.py`、`test_flashmla_dcp.py` | 新增接口/数值/生命周期/同步回归，并在真实 DCP 编排方法里替换 provider 运行独立数值对照 |

runner 修改只扩展已有同步策略，未复制 runner 或打 monkey patch。仅在 A3 provider 已被实际选中且目标消费者全部匹配时生效。后续算子交付仍使用同一适配点。

## 9. 使用与替换

在已有的 A3 K3.1 DSpark、V2 runner 启动配置上合并下面的字段；保留原有其他 additional_config 字段。首轮以 eager、小 batch、短序列验证：

```json
{"a3_mla_sink_mode": "mock"}
```

算子包安装并验证两个接口后，改成：

```json
{
  "a3_mla_sink_mode": "external",
  "a3_mla_sink_module": "flash_attn_npu_3.flash_attn_npu_interface"
}
```

默认未开放 graph：模型或 draft 必须 enforce_eager；否则需要显式设置 `a3_mla_sink_allow_graph=true` 才能进入实验验证。此开关不能替代真实 ACL capture/replay 的 H1 测试。

约束：max_model_len 最多 16384、FP16/BF16、未量化、ND cache、64/512、单 KV head、PP=PCP=1。DCP 逻辑支持 1/2/8；PD 保留长度同步。算子运行时逻辑页大小固定 128，框架管理块允许 128 的整数倍（包括 1536）。超出契约明确报错，不把长度截断来假装支持。

mock 的 `T × 逻辑 KV 容量 × 512` 上限为 16M 元素，限制临时内存；只能用于小规模对接验证，不能用于完整 K3.1 的吞吐测试。真实算子若支持更长序列、其他 stride、量化或更多并行组合，需要同步扩展能力检查和测试，不能仅关闭断言。

无需 NPU 的复现命令（从仓库根目录运行，环境需 torch、pytest）：

```bash
python -m pytest --confcutdir=tests/ut/attention \
  tests/ut/attention/test_a3_mla_sink.py \
  tests/ut/attention/test_flashmla_dcp.py \
  tests/ut/attention/test_flashmla_prolog_v3_cpu.py \
  tests/ut/attention/test_mla_prolog_v3_tile_size_cpu.py -q
```

这些测试加载真实模块或从源码提取真实框架方法，隔离 vLLM/NPU 导入依赖。原始 NPU kernel、通信和流通过 mock 代替；因此这不是完整包 import、模型启动或真实 graph 测试。测试报告单独记录已通过与未运行项目。

## 10. 发布与后续验收

当前以 [Draft PR #21876](https://github.com/vllm-project/vllm-ascend/pull/21876) 评审，release note 位于 `release_notes/fragments/21876.yaml`。范围为开发联调接入：

> Added an opt-in A3 K3.1 DSpark MLA provider adapter and numerical mock for device-length integration development. Native operator, graph and end-to-end hardware validation remain required before production use.

提交生产 PR 前需完成 H1/H2/H3，特别是：拒绝率变化、请求换位、chunked prefill 混批、prefix cache、1536 管理块、目标 KDA 状态回滚、8 rank DCP、PD 长度消费者、stream 生命周期、完整模型精度和真实 D2H/event wait trace。当前 mock 无法承诺这些场景已经通过。

## 11. 源码依据

- [vllm-ascend PR #21542](https://github.com/vllm-project/vllm-ascend/pull/21542)
- [社区 PR #338](https://github.com/MinghuasLab/flash-attention-npu/pull/338)
- [flash_attn_with_kvcache 固定版本](https://github.com/MinghuasLab/flash-attention-npu/blob/7d3f7433669b524ce71c4e867b90dc3bc6f41264/flash_attn_npu_3/flash_attn_npu_interface.py#L1413)
- [get_scheduler_metadata 固定版本](https://github.com/MinghuasLab/flash-attention-npu/blob/7d3f7433669b524ce71c4e867b90dc3bc6f41264/flash_attn_npu_3/flash_attn_npu_interface.py#L1897)
