# Mistral Large 3 675B NVFP4

## 1 模型简介

`mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` 是 Mistral Large 3
的 NVFP4 训练后量化检查点。其文本解码器采用细粒度 MoE 架构，共 675B
参数，每个 token 激活约 41B 参数。Hugging Face 仓库约 403 GB，部署前需
同时规划模型存储、主机内存、NPU 显存和启动时间。

本教程覆盖昇腾平台上的文本生成路径。该检查点同时包含视觉编码器，但当前
`MistralLarge3ForCausalLM` 适配器尚未开放多模态输入。

## 2 功能状态

| 功能 | 状态 | 说明 |
| --- | --- | --- |
| 文本生成 | 已适配，待真实权重验证 | 必须取得 HTTP 200 和非空输出 |
| NVFP4 权重加载 | Python 接口已实现 | 加载 E2M1 打包权重和两级缩放因子 |
| NVFP4 NPU 执行 | 依赖内核 | 需要 `_C_ascend.nvfp4_linear` 和 `_C_ascend.nvfp4_moe` |
| 专家并行 | 启动模板默认开启 | 仅适用于 MoE 模型 |
| FlashComm1 | 启动模板默认开启 | 投产前需以真实流量验证 |
| ACLGraph | 默认验证路径 | Eager 仅用于故障隔离 |
| MTP | 检查点缺失 | 检查点未提供 MTP 权重 |
| 多模态输入 | 当前适配器不支持 | 仅适配文本解码器 |

!!! warning

    Python 适配器可以完成模型构建，但不能证明当前镜像已经包含 NVFP4 NPU
    内核。必须使用真实权重发送至少一次请求。如果缺少任一 `_C_ascend`
    算子，应更换或构建包含对应算子的镜像，不能用启动成功或 dummy 权重替代
    真实权重验证。

## 3 环境依赖准备

### 3.1 硬件与存储

建议从一台包含 16 张 NPU 的 Atlas 800 A3 服务器开始：

- 16 张 NPU 均可见，使用 TP16 和专家并行；
- 模型目录至少预留 500 GB 可用空间；
- 主机内存能容纳权重元数据和加载缓冲区；
- 模型磁盘到主机之间具有稳定的高带宽；
- OpenAI 兼容接口使用的 `8000` 端口未被占用。

模型配置的理论上下文长度超过 256K。本教程先验证 131,072 token 和 16 路
并发。该值是验证目标而非容量承诺；如果服务无法完成启动及首个请求，应先
降低上下文长度和并发数。

### 3.2 启动容器

使用与当前代码版本匹配的 vLLM Ascend 镜像，并将宿主机模型目录挂载到
`/models`，避免容器重建后重复下载。

```shell
export IMAGE=quay.io/ascend/vllm-ascend:{{ vllm_ascend_version }}-a3
export HOST_MODEL_ROOT=/data/models

docker run --rm --name vllm-ascend-mistral-large3-nvfp4 \
    --net=host --shm-size=16g --privileged=true \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v "${HOST_MODEL_ROOT}:/models" \
    -v /root/.cache:/root/.cache \
    -it "${IMAGE}" bash
```

进入容器后检查软件版本和 NPU 数量：

```shell
npu-smi info
pip show vllm vllm-ascend torch torch-npu mistral-common

python - <<'PY'
import torch
import vllm
import vllm_ascend

print("vLLM:", vllm.__file__)
print("vLLM Ascend:", vllm_ascend.__file__)
print("NPU available:", torch.npu.is_available())
print("NPU count:", torch.npu.device_count())
PY
```

不要单独升级 `transformers`。应优先使用版本匹配的 vLLM Ascend 镜像，确保
vLLM、vLLM Ascend、torch、torch-npu、CANN 和 mistral-common 相互兼容。

## 4 下载 Hugging Face 模型

安装 Hugging Face CLI。环境需要鉴权时使用交互式登录，不要将访问令牌写入
脚本、命令历史或本文档。

```shell
python -m pip install --upgrade "huggingface_hub[cli]"
hf auth login

export MODEL_ID=mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4
export MODEL_PATH=/models/Mistral-Large-3-675B-Instruct-2512-NVFP4

mkdir -p "${MODEL_PATH}"
hf download "${MODEL_ID}" \
    --local-dir "${MODEL_PATH}" \
    --max-workers 8
```

Hugging Face CLI 会复用缓存中的完整文件。下载结束后确认 `params.json`、
tokenizer 文件和所有 safetensors 分片均存在：

```shell
test -f "${MODEL_PATH}/params.json"
find "${MODEL_PATH}" -maxdepth 1 -name 'consolidated-*.safetensors' | wc -l
du -sh "${MODEL_PATH}"
```

分片仍在下载时不要启动分布式服务。

## 5 vLLM Ascend 多卡部署

从 `/workspace` 直接启动 vLLM。默认命令开启专家并行、FlashComm1 和
ACLGraph，并显式选择 Ascend NVFP4 后端。

```shell
cd /workspace

export MODEL_PATH=/models/Mistral-Large-3-675B-Instruct-2512-NVFP4
export HCCL_OP_EXPANSION_MODE=AIV
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=1
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve "${MODEL_PATH}" \
    --served-model-name mistral-large-3-nvfp4 \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --quantization nvfp4 \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --max-model-len 131072 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.90 \
    --port 8000
```

首次容量验证保持 `max-model-len=131072` 和 `max-num-seqs=16`。只有在记录
峰值 NPU 显存、TTFT、TPOT、吞吐量及 ACLGraph replay 证据后，才逐步提高
并发数。

### 5.1 Eager 隔离命令

如果图捕获失败，使用相同参数增加 `--enforce-eager` 重试一次。下面同时将
上下文和并发调低，以便区分显存/shape 压力与内核错误。Eager 是隔离手段，
不是优先的生产配置。

```shell
vllm serve "${MODEL_PATH}" \
    --served-model-name mistral-large-3-nvfp4 \
    --tokenizer-mode mistral \
    --config-format mistral \
    --load-format mistral \
    --quantization nvfp4 \
    --dtype bfloat16 \
    --tensor-parallel-size 16 \
    --enable-expert-parallel \
    --max-model-len 32768 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 \
    --enforce-eager \
    --port 8000
```

## 6 功能验证

不能只以 `Application startup complete` 判断成功。先检查 readiness，再以
真实权重发送请求并确认输出非空。

```shell
curl -f http://127.0.0.1:8000/v1/models

curl -f http://127.0.0.1:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "mistral-large-3-nvfp4",
      "messages": [{
        "role": "user",
        "content": "请用中文给出数据库迁移故障的前三项应急措施。"
      }],
      "temperature": 0,
      "max_tokens": 128
    }'
```

响应必须为 HTTP 200，且 `choices[0].message.content` 非空。即使 readiness
已经通过，只要首个请求导致 worker 退出，也应判定为运行时失败。

## 7 常见报错排查模板

### 7.1 日志采集

始终保留完整启动命令、版本、NPU 拓扑、首个错误和最后 200 行日志。诊断时
将完整的 `vllm serve` 参数替换到 `<serve-options>`：

```shell
mkdir -p /workspace/logs
export LOG_FILE=/workspace/logs/mistral-large3-nvfp4-$(date +%Y%m%d-%H%M%S).log

set -o pipefail
vllm serve "${MODEL_PATH}" <serve-options> 2>&1 | tee "${LOG_FILE}"
```

提交问题时使用以下模板：

```text
硬件型号/NPU 数量：
CANN、torch、torch-npu、vLLM、vLLM Ascend 版本：
vLLM 与 vLLM Ascend import 路径：
模型路径、目录大小和分片数量：
完整启动命令及环境变量：
ACLGraph 或 Eager 模式：
readiness 结果：
首个请求内容和 HTTP 状态：
第一个错误签名：
日志文件路径：
```

### 7.2 高频问题

| 现象 | 可能原因 | 处理方法 |
| --- | --- | --- |
| `No space left on device` 或分片数量不足 | 约 403 GB 的检查点未完整下载 | 至少预留 500 GB，续传 `hf download`，启动前核对分片 |
| `No module named ...` 或修改未生效 | 运行时导入了另一套安装 | 打印 `vllm.__file__` 和 `vllm_ascend.__file__`，切换到匹配镜像或 `/vllm-workspace` 安装 |
| 模型架构无法识别 | 运行环境缺少模型适配器 | 确认已注册 `MistralLarge3ForCausalLM` 并核对运行时 commit |
| 缺少 `qscale_weight`、`qscale_weight_2` 或出现未知权重键 | 检查点不完整或 loader 不兼容 | 核对 `params.json`、分片、Mistral load format 和 NVFP4 适配版本 |
| `_C_ascend.nvfp4_linear` 或 `_C_ascend.nvfp4_moe` 不存在 | 镜像未包含 NVFP4 自定义算子 | 安装或构建包含算子的镜像；调整启动参数无法替代缺失内核 |
| 权重加载或图捕获时 NPU OOM | 上下文、并发或 shape 超出容量 | 保持 TP16，将上下文降至 32768、并发降至 4，再用 Eager 隔离 |
| `AclmdlRICaptureEnd ... 507903` | ACLGraph 捕获序列失效 | 保留 `HCCL_OP_EXPANSION_MODE=AIV`，降低 shape 压力，再测试 Eager |
| HCCL `Communication_Error_Bind_IP_Port` | 残留 worker 或通信端口占用 | 停止残留进程，确认 8000 端口空闲，再干净启动所有 rank |
| readiness 正常但首个请求崩溃 | MLA、MoE 或后端引发 false-ready | 保存首个堆栈，用确定性请求在 Eager 重现，修复第一个后端错误 |
| 输出为空或包含模板残片 | endpoint、tokenizer mode 或聊天模板不匹配 | 使用 chat completions 和 Mistral 三种 format，temperature 设为 0 |

常用隔离命令：

```shell
pkill -f "vllm serve|EngineCore" || true
ss -ltnp | grep ':8000' || true
npu-smi info
tail -n 200 "${LOG_FILE}"
```

## 8 精度评测

使用仓库配置 `tests/e2e/models/configs/mistral_large3_675b_nvfp4.yaml`。
其中 GPQA Diamond 数值是检查点参考值，不是昇腾实测结果；只有完成真实权重
评测后才能确认或替换该数值。

```shell
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
    --config tests/e2e/models/configs/mistral_large3_675b_nvfp4.yaml \
    --tp-size 16
```

## 9 性能与投产检查

投产前必须以真实权重记录：

- readiness 与首个请求的 HTTP 状态；
- 复杂推理、多语言和企业场景 prompt 的非空输出；
- TP16+EP rank 健康状态和 FlashComm1 结果；
- ACLGraph 捕获/replay 证据以及 Eager 对照；
- 加载期和稳态的 NPU/主机内存峰值；
- TTFT、TPOT、吞吐量、输入输出长度和并发；
- 模型 revision、镜像 digest、CANN 与 Python 包版本。

官方模型卡提示 NVFP4 在长上下文下可能出现质量或性能下降。必须按照生产
流量的实际上下文分布验证，不能用 32K 工作负载推断理论最大长度的表现。

## 10 参考资料

- [Mistral Large 3 675B NVFP4 模型卡](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4)
- [Hugging Face CLI 文档](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [vLLM Ascend 安装指南](../../installation.md)
- [vLLM Ascend FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)
