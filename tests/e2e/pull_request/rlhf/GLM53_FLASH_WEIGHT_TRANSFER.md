# GLM-5.3-Flash 双卡 IPC unpacked 权重热更新测试用例

参考 [vllm-ascend PR #16959](https://github.com/vllm-project/vllm-ascend/pull/16959)，复用当前仓库的 `FixedRandomWeightSource` 和启动加载参考服务，新增一个 TP=2 的 NPU IPC unpacked 用例。模型配置及 tokenizer 来自 [ModelScope 官方模型](https://www.modelscope.cn/models/ZhipuAI/GLM-5.3-Flash/files)，获取日期为 2026-10-13。

## 测试模型

原始模型为 `Glm5NextForConditionalGeneration`，文本部分共有 45 层、288 个路由专家，采用 KDA / 稀疏 MLA 混合注意力、MoE 和 mHC。发布配置带有 FP8 量化信息。

本测试提取文本配置，使用 `Glm5NextForCausalLM`，保留前 4 层及原始宽度：

| 项目 | 测试配置 |
| --- | --- |
| 层 0、1、2 | KDA 线性注意力 + dense MLP |
| 层 3 | 稀疏 MLA / KPool indexer + MoE |
| hidden_size / vocab_size | 4096 / 154880 |
| 路由专家 / 每 token 专家数 | 8 / 8 |
| mHC | 开启，4 个残差流 |
| MTP、量化、视觉输入 | 关闭 |
| 权重 | 固定随机 BF16，132 个 checkpoint 参数，约 4.64 GiB |
| 推理并行 / graph | TP=2，Model Runner V1，FULL_DECODE_ONLY |
| 最大上下文 / batch / block size | 1024 / 1 / 128 |
| prefix caching | 关闭 |

`glm53flash_assets/config.original.json` 保存官方原始配置；`config.json` 为缩减配置。`parameter_metadata.json` 的名字和形状来自官方 safetensors 文件头，重命名 `model.language_model.` 为 `model.`，去掉 FP8 scales、其余层和专家，同时将 MoE 路由器 `gate.weight`、`e_score_correction_bias` 的专家维度从 288 缩减为 8。获取文件头使用 HTTP Range，只读取元数据，没有下载权重数据。来源、文件哈希和筛选规则记录在 `provenance.json`。

使用 checkpoint 文件头而非运行时融合参数枚举，是因为 GLM-5.3-Flash 在此仓库中提供模型实现，不能直接依赖 Transformers 的 `AutoModelForCausalLM` 枚举参数。合成参数按名称生成固定随机数，并为 KDA 的 `A_log` / `dt_bias`、mHC 和归一化参数设置有界值；正常启动和每次传输均使用同一条生成路径。

本用例验证文本模型的 BF16 权重热更新、融合布局与派生状态恢复。它不验证官方 FP8 权重精度、W8A8/C8、视觉编码器、MTP、专家并行或性能。

## 测试范围

| 用例 | NPU 数量 | 传输方式 | 打包 | 关键状态转换 |
| --- | --- | --- | --- | --- |
| GLM53-IPC-TP2 | 2 | NPU IPC | 否 | pause → sleep(level=2) → wake(weights) → START/LOAD/FINISH → wake(kv_cache) |

推理端使用 TP=2，每张卡上分别放置一个训练进程和一个推理 worker。两个训练 rank 各自在本卡生成相同的完整 checkpoint 参数，IPC 句柄按物理 NPU 标识合并；推理 worker 导入本卡句柄，并由正常的 TP weight loader 加载对应分片。

训练 rank 使用 CPU/Gloo 进程组交换 IPC 句柄元数据和同步，权重仍通过 IPC 传输。仅 rank 0 发起 HTTP 控制请求。每轮先唤醒两张卡的 weights，再由两个训练 rank 共同执行 `send_weights()`；完成后唤醒 KV cache 并比较推理结果。总共执行两轮更新。

Flash 用例位于 `two_card/rlhf/state_transitions/test_glm53flash_npu_ipc_weight_transfer.py`，不加入共享的 `MODEL_CASES`。单卡 IPC 与 HCCL 测试文件均与上游一致，不新增 HCCL 权重传输用例。

## 步骤与断言

1. 从 metadata 构建固定随机权重源，将同一份合成权重写入临时 safetensors checkpoint。
2. 启动 TP=2 的独立参考服务，在正常启动阶段加载该 checkpoint。对两个固定 prompt 采集 greedy 生成文本及逐 token logprobs。
3. 关闭参考服务并等待显存释放，启动 `--load-format dummy` 的待测服务，采集更新前结果。
4. 执行第一轮完整权重更新事务，采集结果。
5. 再次执行同一份权重的完整更新事务，采集结果。
6. 断言生成包含非空 logprobs；更新前结果与参考结果不同；第一轮、第二轮更新的文本和 logprobs 均与参考服务完全一致。

参考服务与待测服务共享 dtype、graph 模式、tokenizer、模型配置、并行配置和显存预算。参考结果独立于热更新路径，避免两次同样错误的更新互相比较而误通过。临时 checkpoint 在测试结束后清理。

## 执行

需要 Linux 昇腾环境、两张空闲 NPU、torch-npu、与当前 vllm-ascend 兼容的 vLLM，以及支持 CPU/Gloo 的 PyTorch 和 pytest、transformers、safetensors 等测试依赖。建议每张 NPU 至少有 64 GiB HBM。每个训练 rank 的完整合成权重约 4.64 GiB，仍需为 TP 推理、运行时状态、graph、缓存及 IPC buffer 预留空间。当前用例不需要下载模型权重，配置和完整 tokenizer 已随用例保存。

在 vllm-ascend 仓库根目录执行：

```bash
# 双卡：TP=2、两个同卡训练 rank、IPC unpacked
pytest -sv tests/e2e/pull_request/two_card/rlhf/state_transitions/test_glm53flash_npu_ipc_weight_transfer.py

# CPU：配置、metadata 和有界初始化检查
pytest -q tests/ut/distributed/weight_transfer/test_glm53flash_weight_fixture.py
pytest -q tests/ut/distributed/weight_transfer/test_glm53flash_tp2_ipc.py
```

默认选择逻辑卡 0、1。`VLLM_RL_TEST_DEVICE_INDEX` 指定起始逻辑卡，使用该卡及下一张逻辑卡。若父进程设置了 `ASCEND_RT_VISIBLE_DEVICES`，会将选中的逻辑卡映射为物理卡后传给服务。例如 `ASCEND_RT_VISIBLE_DEVICES=4,5` 时，两个训练 rank 和两个推理 worker 分别使用物理卡 4、5。

## 验证说明

2026-10-31 在 Windows、CPU 环境完成本地复查，无 torch-npu / 昇腾设备。36 项 CPU 检查通过，包含下述 14 项直接检查与 22 项隔离工具逻辑检查。

- 配置、metadata、有界初始化的 14 项 CPU 检查通过；使用 `--noconftest` 绕过需要完整 vLLM 环境的公共测试配置。
- 另有 22 项隔离工具逻辑检查，覆盖 metadata 枚举、checkpoint 序列化与重复生成一致性、TP=2 参考/热更新配置一致性、两个训练 rank 的启动与清理、物理卡映射、双卡显存门限，以及两个 rank 都发送而仅 rank 0 控制 sleep/wake 的流程。隔离脚本提取 vLLM v0.30.0 的 `ParamMeta` / `WeightSource` 定义，并替代不可用的服务、NPU、网络工具及 Transformers 模型入口；未启动真实服务或模型。
- 本地加载官方 tokenizer 成功。
- 单卡 IPC 和 HCCL 测试文件与上游版本一致。执行 NPU 测试时，请使用目标 vllm-ascend 提交所匹配的 vLLM 版本。
- Ruff、拼写、Markdown 和工作流检查通过。`bash format.sh ci` 的部分入口依赖 Windows 不提供的 `/bin/bash`、`python3` 命令，已分别用 Git Bash、Python 和 Windows 原生扫描工具补跑；PR 差异的空白检查通过。

实际双卡 IPC unpacked 传输、TP=2 graph replay、sleep/wake 和推理一致性仍需在上述 NPU 环境运行，不能据此记录为硬件测试通过。
