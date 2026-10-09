# 静态 W8A8 MLAPO 实施记录

日期：2026-10-09。分支：`main-dsv3-weekly`。实施基线：`640e9c1a3ae8577c49387e2b324ebdc96eba47f3`。

## 结果与验证边界

已应用保留 MLAPO 的静态 W8A8 分支：符合契约的普通 MLA 层调用仓内旧静态融合算子，
原样传递两段 INT32 quant_bias、静态 input_scale/input_offset 和有效 Q norm beta。
不支持的层或 batch 保留 native 回退。FA、dynamic INT8 和 MXFP8 保留原有路径。

本次没有 NPU 或完整推理依赖，**尚未验证新静态融合分支的真实模型精度、ACLgraph 或性能**。
用户之前提供的 31/32 weekly 是关闭 MLAPO 后的结果，不能作为此次新融合分支通过的证据。

## 当前分支冲突处理

开始时存在 revert `128ecf890`（#14676）未解决的四个文件冲突。
直接完成整次旧提交回退会恢复过期 MLA/A5/设备接口，并撤销后续 MoE/EPLB 逻辑。

处理前已保存四个文件完整内容、combined diff、HEAD/revert 元数据，位于
`evidence/before-static-mlapo/`。随后恢复这些文件的当前 HEAD，退出旧 revert 状态，
在当前分支加入范围明确的静态 MLAPO 修复。
没有切换分支；`device_op.py`、`routed_experts.py` 最终保持与当前 HEAD 相同。

## 实现内容

- `vllm_ascend/attention/mla_static.py`：同步重排 weight、FP32 deq_scale、INT32 quant_bias，
  将 down 的 Q/KV 顺序和各 head 的 RoPE 顺序适配旧内核，再按 16/32 打包；保留源参数。
- `vllm_ascend/attention/mla_v1.py`：加载时检查 static MLAPO 能力和参数，
  准备静态权重；decode 使用静态 asymmetric ABI；mixed/prefill/容量或布局不适配时回退 native。
- `vllm_ascend/device/hardware_profile.py`：新增 `MLAPO_STATIC_W8A8` 能力，仅 A2/A3 启用。
- Q norm 无有效 beta 时传持久零向量；有效 beta 按 `bias_loaded` 判断。
- 保留原投影、norm 和量化参数，不复制旧 SFA 的权重释放逻辑。
- 旧 schema 在 `enable_inner_out=False` 时仍要求 `inner_out`，调用中已提供该输出。
- 新静态路径保留层级 KV 等待，并在 cache 写入后通知 connector。
- 本次不改 C++/CANN 内核，不新增环境变量或模型文件。

## 首期边界

- BF16 激活、FP32 deq_scale、ModelSlim static W8A8、非 FA。
- KV rank=512、RoPE=64、NoPE=128、单 KV head、无需 head padding。
- Q rank 不超过 1536、hidden 不超过 7168，并满足 16/32 打包对齐。
- 两处 epsilon=1e-6；KV 有效 norm beta 全零；Q beta dtype 与激活一致。
- ND BF16 cache，RoPE 启用；排除 context parallelism、LoRA、RL reload、sleep mode。
- 纯 decode，实际输入行数为 1..1024；检查 slot、RoPE、cache dtype/shape 和内层 stride。
- cache 可有不重叠的 dim0 stride；不支持内层非连续或重叠 block。

加载阶段读取标量 scale 和有效 KV beta 以判断资格；decode 热路径不做 NPU 标量读取或设备到 CPU 复制。
原始与派生权重同时存在会增加显存，目标 TP=8 部署的实际余量仍需测量。

## 本地验证

| 检查 | 结果 |
| --- | --- |
| Python 路由/helper 隔离 UT | 83 passed；实际源码方法和仓内 fixture/测试体，外部 vLLM/CANN 接口模拟 |
| Hardware profile 隔离检查 | 11 passed；完整能力矩阵及相关仓内测试函数 |
| Ruff lint/format、Python 语法 | 通过 |
| 正式 UT | conftest 导入失败：本机缺少 vllm；未执行测试体 |
| NPU 测试收集 | e2e conftest 导入失败：隔离环境缺少 huggingface_hub；本机也无 torch_npu/NPU |
| NPU 数值、ACLgraph、weekly、性能 | 未执行 |

选定文件的 manual pre-commit 中，Ruff、codespell、typos、markdownlint、文件名、包初始化、
禁用 import 和 with 语法检查均通过。
完整 hook 集合因 Windows 缺少 `/bin/bash` 和可用 `python3` 入口未全通过；
logger、长函数、symbolic meta 检查已分别用 Git Bash/Python 直接运行通过，Gitleaks 未运行。
完整输出保存在 `evidence/static-mlapo-precommit.log`。

隔离 UT 覆盖两段补偿与权重相同重排、源参数保留、有效/未加载 Q beta、配置关闭、
模型与量化资格回退，以及 1/1024 行、混合 batch、空 batch、超容量、slot/cache stride 等运行时分流。
旧 NZ ownership、FA 分流和动态量化支持检查也在隔离集合中。

证据：`evidence/static-mlapo-cpu-tests.log`、`static-mlapo-hardware-tests.log`、
`static-mlapo-formal-ut.log`、`static-mlapo-npu-collect.log`。
隔离 harness 的限制在源码和日志中明确记录；其结果不能替代正式导入、内核或模型精度测试。

## NPU 验证入口

新增 `test_mla_preprocess_static_bias.py` 共 8 个数值用例，覆盖 1/8 tokens、
dim0 stride 1/2、inner_out 关闭/开启。使用非零 M4 down/up bias、Q beta 和 input offset，
对照 native NPU 静态线性层、RMSNorm、RoPE 和 W_UK，比较 query、KV cache、未写入位置及 padding slot=-1。
三个缺失单项补偿的负对照确保测试数据能发现问题。容差预先固定，未依据 NPU 结果调整。

```bash
pytest -sv tests/ut/attention/test_mla_static.py \
  tests/ut/attention/test_mla_static_dispatch.py \
  tests/ut/attention/test_mla_v1.py tests/ut/device/test_hardware_profile.py

pytest -sv tests/e2e/nightly/single_node/ops/singlecard_ops/test_mla_preprocess_static_bias.py
```

随后固定模型 revision、镜像及依赖运行既有 weekly 配置：
`tests/e2e/cases/models/configs/DeepSeek/DeepSeek-V3.yaml`。
以同环境关闭 MLAPO 的 native 结果为参考，要求新路径至少恢复到既有 31/32，
并通过算子 trace/调试确认目标 D 层实际执行静态 MLAPO；仅加载日志不能证明每个 batch 使用融合。

仍需验证实际 TP=8、P native 历史 cache 与 D 融合新行布局、ACLgraph capture/replay、
1024 行真实内核边界、重算/prefill 回退、显存与 decode 性能。
已有旧内核在 FP32 加 Q beta 后量化，与 native 的 BF16 cast 边界不同，
若真实数值对照显示影响精度，须继续对齐内核 cast/量化顺序。
