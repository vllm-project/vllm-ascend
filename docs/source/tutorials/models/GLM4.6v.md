# GLM-4.6V

## 1 模型介绍

GLM-4.6V 是智谱的视觉语言 MoE 模型，包含约 106B 总参数和 12B 激活参数，支持图像、视频及文档理解等多模态任务。本教程使用完整的 GLM-4.6V 权重。

本文参考 [Ascend-SACT GLM-4.6V 部署指导](https://ai.gitcode.com/Ascend-SACT/GLM-4.6V)，介绍在 A3 上使用 vLLM Ascend 部署在线服务及验证文本、图像输入的方法。

参考指导使用 `v0.22.1rc1-a3` 镜像和 CANN 9.0.0。本文面向 `releases/v0.23.0` 分支，镜像版本随文档构建配置更新；请使用该版本的完整配套软件栈，详见[安装指南](../../installation.md)。参考指导中的运行结果不能作为本文版本的 NPU 验证结果。

## 2 环境与权重准备

### 2.1 硬件要求

准备 Linux A3 主机，确保驱动、固件与所选镜像的 CANN 版本兼容。使用宿主机的 `npu-smi info` 确认设备状态及空闲显存。

本文命令使用 8 个可见 NPU 设备（`/dev/davinci0` 至 `/dev/davinci7`），并设置 TP=8。A3 的物理卡数和可见设备数应以实际硬件配置为准；参考指导中的“单机4卡”不表示可以只映射 4 个设备运行 TP=8。运行前确认这 8 个设备均可用、支持互联，并具备足够显存加载完整权重及 KV cache。

### 2.2 下载模型权重

在宿主机安装 Git LFS 后，将 [GLM-4.6V 权重](https://www.modelscope.cn/models/ZhipuAI/GLM-4.6V)下载到共享目录。确保磁盘空间足够，并检查大文件已实际下载，而非仅有 LFS 指针文件。

```bash
git lfs install
git clone https://www.modelscope.cn/ZhipuAI/GLM-4.6V.git /mnt/share_space/GLM-4.6V
git -C /mnt/share_space/GLM-4.6V lfs pull
git -C /mnt/share_space/GLM-4.6V rev-parse HEAD
```

保留权重提交号，便于复现。本教程将 `/mnt/share_space` 挂载进容器，容器内沿用相同模型路径。

## 3 启动容器

在宿主机执行以下命令。镜像已包含配套的 vLLM 与 vLLM Ascend，无需额外覆盖安装其他版本。

```{code-block} bash
   :substitutions:

export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
docker pull "$IMAGE"
docker run --rm \
    --name vllm-ascend-glm46v \
    --shm-size=8g \
    -p 127.0.0.1:4100:4100 \
    --device /dev/davinci0 \
    --device /dev/davinci1 \
    --device /dev/davinci2 \
    --device /dev/davinci3 \
    --device /dev/davinci4 \
    --device /dev/davinci5 \
    --device /dev/davinci6 \
    --device /dev/davinci7 \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool:ro \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v /mnt/share_space:/mnt/share_space:ro \
    -it "$IMAGE" bash
```

共享内存用于 TP 多进程及多模态处理缓存。这里从 8 GiB 开始配置；高并发或大图像场景需根据实际内存占用增加容量。宿主机示例仅开放本机回环地址端口，客户端从同一主机访问。

进入容器后，检查模型文件和软件版本：

```bash
test -f /mnt/share_space/GLM-4.6V/config.json
npu-smi info
python -c 'from importlib.metadata import version; print({p: version(p) for p in ("vllm", "vllm-ascend", "torch", "torch-npu", "transformers")})'
```

## 4 部署在线推理服务

在容器中运行以下命令，保持该终端运行：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ASCEND_GLOBAL_WORK_SIZE=32
export ASCEND_LAUNCH_BLOCKING=0

vllm serve /mnt/share_space/GLM-4.6V \
    --tensor-parallel-size 8 \
    --mm-encoder-tp-mode data \
    --mm-processor-cache-type shm \
    --enable-expert-parallel \
    --max-model-len 65536 \
    --max-num-seqs 256 \
    --max-num-batched-tokens 8192 \
    --max-cudagraph-capture-size 32 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.95 \
    --enable-auto-tool-choice \
    --tool-call-parser glm45 \
    --reasoning-parser glm45 \
    --served-model-name vllm-glm46v \
    --host 0.0.0.0 \
    --port 4100
```

上述配置沿用参考指导的主要参数，并使用 vLLM 0.23.0 的规范参数名 `--enable-expert-parallel` 和 `--max-num-seqs`。

| 参数 | 说明 |
| --- | --- |
| `--tensor-parallel-size 8` | 语言模型使用 8 路张量并行，与可见设备数一致。 |
| `--enable-expert-parallel` | 为 MoE 专家启用专家并行。 |
| `--mm-encoder-tp-mode data` | 为视觉编码器选择数据并行模式。 |
| `--mm-processor-cache-type shm` | 使用共享内存保存多模态处理缓存。 |
| `--max-model-len 65536` | 本示例上下文上限，输入和输出合计受该值限制。 |
| `--max-num-seqs 256` | 并发序列上限，需结合显存和实际请求调整。 |
| `--gpu-memory-utilization 0.95` | vLLM 通用参数名，同样用于 NPU 显存预算；需给其他占用预留空间。 |
| `--tool-call-parser glm45` | 配合自动工具选择解析工具调用。 |
| `--reasoning-parser glm45` | 解析推理内容；响应可能包含独立的推理字段。 |

这些参数是部署起点，容量和吞吐需在实际 A3 环境测量。若启动时显存不足，先降低 `--max-model-len` 和 `--max-num-seqs`，并确认设备没有其他负载。若出现共享内存空间不足，检查容器内 `/dev/shm` 容量，再调整 `--shm-size`。

## 5 推理验证

### 5.1 检查服务状态

等待服务完成模型加载，然后在宿主机另一个终端执行：

```bash
curl --fail-with-body http://127.0.0.1:4100/health
curl --fail-with-body http://127.0.0.1:4100/v1/models
```

健康检查应返回 HTTP 200，模型列表应包含 `vllm-glm46v`。如请求失败，先检查服务终端中的模型加载或设备通信错误。

### 5.2 文本请求

```bash
curl --fail-with-body http://127.0.0.1:4100/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"vllm-glm46v","messages":[{"role":"user","content":"你好，请介绍一下自己。"}],"max_tokens":256,"temperature":0.0}'
```

检查响应中的 `choices`、`message.content`、推理字段及 `finish_reason`。如果仅返回推理内容且 `finish_reason` 为 `length`，增加 `max_tokens` 后重试。

### 5.3 图像请求

在宿主机安装 OpenAI Python 客户端，并将下面的 `image_path` 改为可读取的本地 JPEG 图片。客户端将图像编码为 data URL，不要求服务端访问外部图片网址。

```bash
python -m pip install openai
```

```python
import base64
from pathlib import Path

from openai import OpenAI

image_path = Path("/mnt/share_space/example.jpg")
image_data = base64.b64encode(image_path.read_bytes()).decode("ascii")
client = OpenAI(base_url="http://127.0.0.1:4100/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="vllm-glm46v",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请描述这张图片中的主要内容。"},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                },
            ],
        }
    ],
    max_tokens=1024,
    temperature=0.0,
)
print(response.model_dump_json(indent=2))
```

检查回答与图片内容是否一致。文本请求通过仅能证明文本路径可用，还需通过图像请求验证视觉编码与多模态输入路径。视频理解、工具调用、精度评测及性能测试需按实际业务另行验证；本教程未提供这些场景的实测结论。
