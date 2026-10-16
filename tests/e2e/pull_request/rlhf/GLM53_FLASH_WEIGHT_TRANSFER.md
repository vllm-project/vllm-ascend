# GLM-5.3-Flash 权重热更新测试用例

参考 [vllm-ascend PR #16959](https://github.com/vllm-project/vllm-ascend/pull/16959)，复用当前仓库的 `FixedRandomWeightSource`、启动加载参考服务和 IPC/HCCL 事务测试。模型配置及 tokenizer 来自 [ModelScope 官方模型](https://www.modelscope.cn/models/ZhipuAI/GLM-5.3-Flash/files)，获取日期为 2026-10-13。

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
| 推理并行 / graph | TP=1，Model Runner V1，FULL_DECODE_ONLY |
| 最大上下文 / batch / block size | 1024 / 1 / 128 |
| prefix caching | 关闭 |

`glm53flash_assets/config.original.json` 保存官方原始配置；`config.json` 为缩减配置。`parameter_metadata.json` 的名字和形状来自官方 safetensors 文件头，重命名 `model.language_model.` 为 `model.`，去掉 FP8 scales、其余层和专家，同时将 MoE 路由器 `gate.weight`、`e_score_correction_bias` 的专家维度从 288 缩减为 8。获取文件头使用 HTTP Range，只读取元数据，没有下载权重数据。来源、文件哈希和筛选规则记录在 `provenance.json`。

使用 checkpoint 文件头而非运行时融合参数枚举，是因为 GLM-5.3-Flash 在此仓库中提供模型实现，不能直接依赖 Transformers 的 `AutoModelForCausalLM` 枚举参数。合成参数按名称生成固定随机数，并为 KDA 的 `A_log` / `dt_bias`、mHC 和归一化参数设置有界值；正常启动和每次传输均使用同一条生成路径。

本用例验证文本模型的 BF16 权重热更新、融合布局与派生状态恢复。它不验证官方 FP8 权重精度、W8A8/C8、视觉编码器、MTP、专家并行或性能。

## 用例矩阵

| 用例 | NPU 数量 | 传输方式 | 打包 | 关键状态转换 |
| --- | --- | --- | --- | --- |
| GLM53-IPC-01 | 1 | NPU IPC | 否 | pause → sleep(level=2) → wake(weights) → START/LOAD/FINISH → wake(kv_cache) |
| GLM53-IPC-02 | 1 | NPU IPC | 是 | 同上，覆盖 packed 收包 |
| GLM53-HCCL-01 | 2 | HCCL | 否 | pause → START → broadcast → FINISH → resume |
| GLM53-HCCL-02 | 2 | HCCL | 是 | 同上，覆盖 packed broadcast |

IPC 的训练端和推理端共用同一张卡；HCCL 的推理端使用卡 0，训练端使用卡 1。每个用例执行两轮相同权重更新。IPC 显式释放权重和 KV cache 后重新唤醒，覆盖 level-2 sleep 后的恢复；HCCL 保留现有协议握手及并发 RPC 流程。

## 步骤与断言

1. 从 metadata 构建固定随机权重源，将同一份合成权重写入临时 safetensors checkpoint。
2. 启动独立参考服务，在正常启动阶段加载该 checkpoint。对两个固定 prompt 采集 greedy 生成文本及逐 token logprobs。
3. 关闭参考服务并等待显存释放，启动 `--load-format dummy` 的待测服务，采集更新前结果。
4. 执行第一轮完整权重更新事务，采集结果。
5. 再次执行同一份权重的完整更新事务，采集结果。
6. 断言生成包含非空且有限的 logprobs；更新前结果与参考结果不同；第一轮、第二轮更新的文本和 logprobs 均与参考服务完全一致。

参考服务与待测服务共享 dtype、graph 模式、tokenizer、模型配置、并行配置和显存预算。参考结果独立于热更新路径，避免两次同样错误的更新互相比较而误通过。临时 checkpoint 在测试结束后清理。

## 执行

需要 Linux 昇腾环境、torch-npu、与当前 vllm-ascend 兼容的 vLLM，以及 pytest、transformers、safetensors 等测试依赖。建议使用至少 64 GiB HBM 的空闲 NPU；缩减权重约 4.64 GiB，仍需为运行时状态、graph、缓存及 IPC buffer 预留空间。当前用例不需要下载模型权重，配置和完整 tokenizer 已随用例保存。

在 vllm-ascend 仓库根目录执行：

```bash
# 单卡：IPC packed / unpacked 两个组合
pytest -sv tests/e2e/pull_request/one_card/rlhf/state_transitions/test_npu_ipc_weight_transfer.py -k glm-5.3-flash

# 双卡：HCCL packed / unpacked 两个组合
pytest -sv tests/e2e/pull_request/two_card/rlhf/state_transitions/test_hccl_weight_transfer.py -k glm-5.3-flash

# CPU：配置、metadata 和有界初始化检查
pytest -q tests/ut/distributed/weight_transfer/test_glm53flash_weight_fixture.py
```

默认从物理卡 0 开始。共用服务器时可沿用测试已有的 `VLLM_RL_TEST_DEVICE_INDEX` 选择空闲卡；保持 pytest 父进程的 `ASCEND_RT_VISIBLE_DEVICES` 未设置，使逻辑卡号与物理卡号一致。HCCL 还会占用该卡的下一张卡。

## 本机验证记录

当前开发环境为 Windows、CPU，无 torch-npu / 昇腾设备。

- 配置、metadata、有界初始化的 14 项 CPU 检查通过；使用 `--noconftest` 绕过需要完整 vLLM 环境的公共测试配置。
- 另有 5 项工具逻辑检查通过，覆盖参考服务参数一致性、参考结果缓存、离线 metadata 枚举、checkpoint 序列化与重复生成的逐元素一致性。这些检查使用本地隔离脚本，提取 vLLM v0.30.0 的 `ParamMeta` / `WeightSource` 定义，并替代不可用的服务及 Transformers 模型入口；未启动真实服务或模型。
- 本地加载官方 tokenizer 成功。
- 修改的 6 个 Python 文件通过 Ruff lint、format 检查和语法编译；`git diff --check` 通过。

实际 IPC/HCCL 传输、graph replay、sleep/wake 和推理一致性仍需在上述 NPU 环境运行，不能据此记录为硬件测试通过。
