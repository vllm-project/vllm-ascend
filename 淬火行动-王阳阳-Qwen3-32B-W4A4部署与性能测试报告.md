# 淬火行动-王阳阳-Qwen3-32B-W4A4 部署与性能测试报告


- 交付人：王阳阳（GitHub: @wyy52112）
- 跟踪 Issue：https://github.com/vllm-project/vllm-ascend/issues/11759
已在 HiDevLab 提供的单卡环境完成模型部署、功能验证、基础性能测试和资料验证，结论：**PASS**。

### 1. 测试环境

- NPU：1 × Ascend 910B3（64 GB HBM）
- npu-smi：25.5.2
- vLLM：0.18.0
- vLLM-Ascend：0.18.0rc1
- 模型：Qwen3-32B-W4A4（Ascend W4A4）
- 最大上下文长度：4096
- HiDevLab 已提供容器化 WebIDE 环境，因此未重复执行宿主机 docker run；在该环境内完成后续全部部署和测试。

### 2. 模型准备与校验

按实验环境要求直接使用共享目录中的已下载模型，复制到个人目录，全程未重新下载。

命令：

    rsync -a --partial --info=progress2 /workspace/shared_assets/models/Qwen/Qwen3-32B-W4A4/ /workspace/user_data/models/Qwen3-32B-W4A4/
    rsync -ani --delete /workspace/shared_assets/models/Qwen/Qwen3-32B-W4A4/ /workspace/user_data/models/Qwen3-32B-W4A4/

校验结果：源/目标均为 22 个文件、42,769,712,807 字节，rsync dry-run 无差异。

### 3. 部署脚本/关键命令

    export ASCEND_RT_VISIBLE_DEVICES=0
    vllm serve /workspace/user_data/models/Qwen3-32B-W4A4 --served-model-name qwen3-32b-w4a4 --max-model-len 4096 --quantization ascend --host 0.0.0.0 --port 8000

11 个权重分片加载成功，服务启动成功；GET /health 返回 HTTP 200。测试结束时 VLLMEngineCore 正常运行，HBM 占用约 62005 / 65536 MB。

### 4. 功能/推理验证

使用 OpenAI 兼容 Chat Completions 接口，关闭 Qwen3 thinking 模式。请求参数：model=qwen3-32b-w4a4，max_tokens=64，temperature=0，chat_template_kwargs.enable_thinking=false。

测试问题：请用一句话说明什么是大型语言模型。

响应：

> 大型语言模型是一种基于大量文本数据训练而成的深度学习模型，能够理解和生成自然语言，用于回答问题、创作文字、翻译语言等任务。

Token usage：prompt 21，completion 34，total 55。接口与推理结果正常。

### 5. 基础性能测试

    vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 --model qwen3-32b-w4a4 --tokenizer /workspace/user_data/models/Qwen3-32B-W4A4 --dataset-name random --num-prompts 20 --random-input-len 128 --random-output-len 128 --request-rate inf --save-result

压测原始结果摘要：

    Successful requests:                 20
    Failed requests:                      0
    Request throughput (req/s):        2.82
    Output token throughput (tok/s): 360.38
    Total token throughput (tok/s):  720.76
    Mean TTFT (ms):                   541.01
    Median TTFT (ms):                 517.82
    P99 TTFT (ms):                    695.44
    Mean TPOT (ms):                    51.58
    Median TPOT (ms):                  51.79
    P99 TPOT (ms):                     52.87
    Mean ITL (ms):                     51.58
    Median ITL (ms):                   50.43
    P99 ITL (ms):                      78.81

### 6. 问题、规避方案与 FAQ

1. **共享目录直接加载较慢**：将模型复制到个人目录后，11 个权重分片约数秒完成加载。
2. **benchmark 初次尝试访问 Hugging Face**：服务别名被客户端当作远程 tokenizer 名；显式指定本地 --tokenizer /workspace/user_data/models/Qwen3-32B-W4A4 后解决，未下载任何模型。
3. **原始 /v1/completions 是续写接口**：直接使用问句可能得到继续补写问题的文本；补充用 /v1/chat/completions 且设置 chat_template_kwargs.enable_thinking=false 后，得到正常的一句话回答。
4. **WebIDE 文件监听告警**：不影响服务或压测，必要时手动刷新 Explorer。

### 7. 社区文档验证反馈

文档链接：https://docs.vllm.ai/projects/ascend/en/v0.18.0/tutorials/models/Qwen3-32B-W4A4.html

- 文档的容器、量化模型结构、单卡 Online Serving 和 Offline Inference 步骤清晰，部署命令可用。
- 当前 v0.18.0 页面目录中没有 Issue 测试方法所引用的 **Run basic benchmark** 章节，建议补充 vllm bench serve 命令、输入/输出长度、请求数、本地 tokenizer 参数及指标解释。
- Issue 的测试方法第 2、3 个超链接把中文“的”/“的Run”拼进 URL，点击后会形成错误链接，建议把中文说明移到链接外。
- 建议在 Online Serving 中补充 Chat Completions 示例，并说明 Qwen3 的 thinking 开关，避免把原始 completions 的续写结果误判为问答精度异常。

### 8. 交付物

实验目录：/workspace/user_data/qwen3_32b_test

包含 serve.sh、functional_test.sh、benchmark.sh、deployment_report.md、服务/功能/压测日志、功能响应 JSON、压测结果 JSON、模型复制校验日志和 NPU 状态日志。

最终结论：模型复制完整，服务健康检查通过，功能推理正常，20/20 性能请求成功，任务通过。
