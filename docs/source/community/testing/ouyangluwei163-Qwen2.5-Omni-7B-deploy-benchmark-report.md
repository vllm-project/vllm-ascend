# 欧阳陆伟 - Qwen2.5-Omni-7B 部署与性能测试报告

- 交付人：欧阳陆伟（GitHub: @ouyangluwei163）
- 测试日期：2026-07-22
- 执行环境：Kubernetes 集群 `gen-studio` 命名空间，Pod `infer-97b01f87-23ee-45ac-be14-1329cf305bc7-0`
- 结论：**PASS**（服务健康、文本与多模态推理正常、20/20 压测请求成功、0 失败）

> 说明：本报告参照 PR #12544 的结构组织，但测试对象、硬件与环境均不同（本次为 Qwen2.5-Omni-7B 非量化模型，单张 Ascend 910B4 虚拟化切分卡），测试在自有 Kubernetes 集群环境执行。

## 1. 测试环境

| 项 | 值 |
| --- | --- |
| NPU | 1 × Ascend 910B4（HAMI 虚拟化，配额 32768 MB HBM） |
| npu-smi | 25.5.1 |
| 容器镜像 | `swr.cn-north-4.myhuaweicloud.com/inference-engines/vllm-ascend:v0.18.0rc1` |
| vLLM | 0.18.0 |
| vLLM-Ascend | 0.18.0rc1 |
| torch / torch_npu | 2.9.0+cpu / 2.9.0.post1+gitee7ba04 |
| 模型 | Qwen2.5-Omni-7B（BF16 权重，运行时 cast 为 float16） |
| 最大上下文长度 | 8192 |
| 节点架构 | arm64 |

资源限制（Pod spec）：cpu 8、memory 64Gi、`huawei.com/Ascend910B4: 1`、`huawei.com/Ascend910B4-memory: 32768`，QoS `Guaranteed`。

## 2. 部署配置

模型经 PVC `model` 以只读方式挂载到容器 `/models`，实际路径 `/models/Qwen2.5-Omni-7B`。

服务启动命令（容器 entrypoint）：

    python3 -m vllm.entrypoints.openai.api_server \
      --model /models/Qwen2.5-Omni-7B \
      --served-model-name gen-studio_Qwen2.5-Omni-7B-esru \
      --port 8000 \
      -tp 1 \
      --dtype float16 \
      --max-model-len 8192 \
      --gpu-memory-utilization 0.8

启动关键日志：

- 平台插件激活：`Platform plugin ascend is activated`
- 模型识别：`Qwen2_5OmniModel model (layers: 28)`
- 图模式：`PIECEWISE compilation enabled on NPU`，`CUDAGraphMode.PIECEWISE`
- 权重 dtype 转换：`WARNING [model.py:1920] Casting torch.bfloat16 to torch.float16`
- 分块预填充：`Chunked prefill is enabled with max_num_batched_tokens=2048`
- 异步调度：`Asynchronous scheduling is enabled`

稳定运行时 HBM 占用 27054 / 32768 MB（`VLLMEngineCor` 进程 24247 MB）。

## 3. 功能验证

### 3.1 健康检查

    curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/health
    # 200

### 3.2 模型列表

    curl -s http://127.0.0.1:8000/v1/models

返回 `id=gen-studio_Qwen2.5-Omni-7B-esru`，`root=/models/Qwen2.5-Omni-7B`，`max_model_len=8192`。

### 3.3 文本对话（Chat Completions）

请求参数：`max_tokens=64`，`temperature=0`。

问题：请用一句话说明什么是大型语言模型。

响应：

> 大型语言模型是一种能够理解和生成自然语言的计算机程序，它通过大量的文本数据进行训练，从而能够理解和生成各种类型的文本，包括但不限于对话、文章、故事等。

Token usage：prompt 28，completion 39，total 67。`finish_reason=stop`。

### 3.4 多模态图像理解

Qwen2.5-Omni 为全模态模型，额外验证图像输入路径。构造 224×224 白底红色实心圆 PNG，以 base64 data URI 通过 `image_url` 传入。

提问：这张图片里是什么形状？什么颜色？用一句话回答。

响应：

> 这张图片里是一个红色的圆圈。

Token usage：prompt 98，completion 10，total 108。视觉编码器与图像预处理链路（`Qwen2VLImageProcessor`，fast processor）工作正常。

## 4. 性能测试

命令：

    vllm bench serve --backend vllm \
      --base-url http://127.0.0.1:8000 \
      --model gen-studio_Qwen2.5-Omni-7B-esru \
      --tokenizer /models/Qwen2.5-Omni-7B \
      --dataset-name random \
      --num-prompts 20 \
      --random-input-len 128 \
      --random-output-len 128 \
      --request-rate inf \
      --save-result --result-dir /tmp

原始输出：

    ============ Serving Benchmark Result ============
    Successful requests:                     20
    Failed requests:                         0
    Benchmark duration (s):                  4.42
    Total input tokens:                      2560
    Total generated tokens:                  2560
    Request throughput (req/s):              4.52
    Output token throughput (tok/s):         578.96
    Peak output token throughput (tok/s):    640.00
    Peak concurrent requests:                20.00
    Total token throughput (tok/s):          1157.93
    ---------------Time to First Token----------------
    Mean TTFT (ms):                          363.08
    Median TTFT (ms):                        353.05
    P99 TTFT (ms):                           438.79
    -----Time per Output Token (excl. 1st token)------
    Mean TPOT (ms):                          31.89
    Median TPOT (ms):                        31.99
    P99 TPOT (ms):                           32.50
    ---------------Inter-token Latency----------------
    Mean ITL (ms):                           31.89
    Median ITL (ms):                         31.17
    P99 ITL (ms):                            46.42
    ==================================================

结果文件：`/tmp/vllm-infqps-gen-studio_Qwen2.5-Omni-7B-esru-20260722-090053.json`

### 4.1 数据自洽性校验

对指标做交叉验算，确认结果内部一致：

- 端到端时长：`TTFT 363.08 ms + TPOT 31.89 ms × 127 token = 4.41 s`，与 `Benchmark duration 4.42 s` 吻合
- 输出吞吐：`20 × 128 ÷ 4.42 s = 579.2 tok/s`，与实测 `578.96` 吻合
- 总吞吐：input/output 均为 128，故 `total = 2 × output = 1157.93`，一致
- 单请求 decode 速度 `1 / 0.03189 ≈ 31.4 tok/s`，20 并发下理论上限约 627 tok/s，与 `Peak output token throughput 640.00` 同量级

### 4.2 测试口径说明

以下参数会影响指标解读，如实记录：

- `vllm bench serve` 默认 endpoint 为 `/v1/completions`（非 chat 接口），走的是补全路径
- 工具提示：`vllm bench serve no longer sets temperature==0 by default`，本次未显式指定 `--temperature`，采样参数由服务端决定
- `num_warmups=0`，未做预热；`ready_check_timeout_sec=0`，跳过了 endpoint ready check
- `--request-rate inf` 意味着 20 个请求一次性打入，`Peak concurrent requests 20.00`，测的是小并发批处理峰值而非稳态 QPS

压测后服务状态：Pod `1/1 Running`，`RESTARTS 0`，HBM 27428 / 32768 MB，服务未受损。

## 5. 问题记录与 FAQ

### 5.1 模型路径拼接缺失斜杠导致 CrashLoopBackOff（本次实际排查并修复）

**现象**：Pod 反复重启，状态 `CrashLoopBackOff`。

**根因**：平台下发的 StatefulSet 中，`MODEL_PATH` 环境变量与 `--model` 启动参数均为 `/modelsQwen2.5-Omni-7B` —— PVC 挂载点 `/models` 与模型目录名之间漏了一个 `/`。该本地路径不存在，vLLM 退回将其当作 HuggingFace repo id 解析，报错链为：

    huggingface_hub.errors.HFValidationError: Repo id must use alphanumeric chars ... '/modelsQwen2.5-Omni-7B'
    → OSError: Can't load the configuration of '/modelsQwen2.5-Omni-7B'

调用栈落在 `vllm/transformers_utils/config.py:572 maybe_override_with_speculators` → `PretrainedConfig.get_config_dict`。

**修复**：将 StatefulSet 中 `containers[0].args[1]` 与 `containers[0].env[MODEL_PATH]` 一并改为 `/models/Qwen2.5-Omni-7B`：

    kubectl patch sts infer-97b01f87-23ee-45ac-be14-1329cf305bc7 -n gen-studio --type=json -p '[
      {"op":"replace","path":"/spec/template/spec/containers/0/args/1","value":"/models/Qwen2.5-Omni-7B"},
      {"op":"replace","path":"/spec/template/spec/containers/0/env/1/value","value":"/models/Qwen2.5-Omni-7B"}
    ]'

**注意**：滚动更新不会自动替换处于 CrashLoop 的 Pod，需手动 `kubectl delete pod` 触发按新 revision 重建。

**遗留**：该错误路径由上游 gen-studio 平台在生成部署时拼接产生，本次仅为集群侧止血，平台再次下发同一服务时会复现。建议在平台侧改用 path join 而非字符串直连。

### 5.2 本地模型路径应避免被误判为 HF repo id

当 `--model` 指向本地目录却拼写错误时，报错信息首先抛出的是 `HFValidationError`，容易被误读为网络/鉴权问题。排查时应优先确认路径在容器内真实存在：

    kubectl exec <pod> -n <ns> -- ls /models

### 5.3 benchmark 的 tokenizer 需显式指定本地路径

`--model` 传的是服务别名（`gen-studio_Qwen2.5-Omni-7B-esru`），若不显式指定 `--tokenizer`，客户端会把别名当远程 tokenizer 名去解析。本次显式传入 `--tokenizer /models/Qwen2.5-Omni-7B`，全程未访问外网、未下载任何权重。

### 5.4 BF16 权重在 float16 下运行

启动参数指定 `--dtype float16`，而 Qwen2.5-Omni-7B 原始权重为 bfloat16，vLLM 会打印 cast 警告。本次功能与压测均正常，但 float16 动态范围小于 bfloat16，长上下文或数值敏感场景建议改用 `--dtype bfloat16` 后重新验证精度。

## 6. 验证范围声明

明确区分已验证与未验证项，未执行的部分不给结论：

**已验证**：

- 服务健康检查、`/v1/models` 元数据
- Chat Completions 文本推理（非流式）
- 多模态图像输入推理
- `vllm bench serve` 随机数据集吞吐/时延压测
- 故障排查与修复闭环（5.1）

**未验证**（本次未执行，不作评价）：

- 音频、视频输入模态
- 音频输出（Qwen2.5-Omni 的 Talker 语音生成能力）
- 流式响应（SSE）
- 长上下文（接近 8192）场景
- 精度评测（未跑任何 benchmark 数据集，无精度结论）
- 多卡张量并行（本次固定 `-tp 1`）
- 稳态 QPS 与长时间稳定性（压测仅 4.42 秒、20 请求）

## 7. 复现方式

本次测试全部通过 `kubectl exec` 在运行中的 Pod 内执行，未新建容器。核心命令已在第 2–4 节逐条列出，可直接复现。压测原始结果 JSON 位于容器内 `/tmp/vllm-infqps-*.json`（容器重建后即丢失，如需长期留存应提前拷出）。
