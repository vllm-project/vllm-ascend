# Mistral Large 3 675B NVFP4 部署与容量验证

## 1 Introduction / 模型简介

`mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4` 是 Mistral Large 3
的 NVFP4 量化检查点。模型为细粒度 MoE 架构，总参数约 675B，每个 token
激活约 41B 参数。Hugging Face 仓库约 403 GB；部署前必须同时规划模型盘、
主机内存、NPU HBM、分布式拓扑和首次图编译时间。

本仓库提供文本解码器适配、NVFP4 权重接口和部署模板。真实权重通过
`GET /v1/models` 与首个非空推理响应之前，不得宣称 Mistral 检查点已在
Ascend 上验证通过。

### 本次验证边界

2026-07-19 的实际环境只有两张 Ascend 910B2C，每张 HBM 65,536 MiB；
`/data` 总容量 117.56 GiB，预留运行空间后的安全权重上限 100.74 GiB。
该环境无法容纳约 403 GB 的 Mistral 检查点，双卡 HBM 也不足以可靠部署
675B 模型，因此 Mistral 真实权重测试状态为
`blocked_by_hardware_capacity`，不是软件报错，也不是测试通过。

同一环境已使用 `Qwen/Qwen-7B-Chat` 的 8 个真实 safetensors 分片完成
TP=2 验证：两张 NPU Worker 均运行、ACL Graph replay 成功、
`/v1/models` 和 `/v1/chat/completions` 均返回 HTTP 200 且输出非空。
该案例证明 vLLM-Ascend、CANN、双卡通信与 OpenAI API 链路可用，但不能
替代 Mistral NVFP4 的权重加载、MoE、EP、FlashComm1 或量化内核验证。

## 2 Supported Features / 支持状态

| 功能 | 状态 | 说明 |
| --- | --- | --- |
| 文本解码器 | 已实现，真实 Mistral 权重未验证 | 必须通过真实权重门禁 |
| NVFP4 权重加载 | 接口已实现 | 仍需完整检查点验证权重键和 scale |
| NVFP4 NPU 执行 | 依赖镜像内核 | 需要 `_C_ascend.nvfp4_linear` 与 `_C_ascend.nvfp4_moe` |
| TP + EP | Mistral 推荐路径 | 建议 A3 TP16，并启用 EP |
| FlashComm1 | 待 Mistral 实机验证 | 仅 MoE 路径适用 |
| ACLGraph | 默认目标 | Eager 仅用于故障隔离 |
| MTP | checkpoint missing | 检查点未提供 MTP 权重 |
| 多模态输入 | 当前适配器不支持 | 本教程只覆盖文本生成 |
| Qwen TP=2 参考 | 已用真实权重验证 | 仅证明环境和通用 API/TP 链路 |

Dummy 权重只能验证架构构建和部分算子路径，不能验证 NVFP4 分片、权重映射、
量化 scale、真实显存占用或首个真实输出。

## 3 Environment Preparation / 环境准备

### 3.1 资源门禁

部署 Mistral 前至少检查：

```shell
df -h /data
free -h
npu-smi info
python - <<'PY'
import torch
print("visible_npu_count=", torch.npu.device_count())
PY
```

建议从 16 张 NPU 的 Atlas A3 服务器开始，模型盘至少预留 500 GiB。不要把
“磁盘能下载”误当成“HBM 能加载”；还需为 KV cache、图捕获、临时张量和
主机侧加载缓冲区保留空间。

### 3.2 容器

=== "A3 series"

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|-a3
    docker run --rm --name vllm-ascend-mistral-large3 \
        --net=host --shm-size=16g --privileged=true \
        --device /dev/davinci_manager --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /data/models:/models -v /root/.cache:/root/.cache \
        -it "${IMAGE}" bash
    ```

=== "A2 series"

    完整 675B 检查点通常需要多机或更大 NPU 拓扑。先按部署规划配置多机
    TP/EP、HCCL 网卡与共享模型盘，再启动标准 A2 镜像。

    ```shell
    export IMAGE=quay.io/ascend/vllm-ascend:|vllm_ascend_version|
    docker run --rm --name vllm-ascend-mistral-large3 \
        --net=host --shm-size=16g --privileged=true \
        --device /dev/davinci_manager --device /dev/devmm_svm \
        --device /dev/hisi_hdc \
        -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /data/models:/models -it "${IMAGE}" bash
    ```

核对运行时版本与导入路径：

```shell
python -c "import vllm,vllm_ascend;print(vllm.__version__);print(vllm.__file__);print(vllm_ascend.__file__)"
python -c "import torch,torch_npu;print(torch.__version__,torch_npu.__version__);print(torch.npu.device_count())"
```

不要单独升级 `transformers`。vLLM、vLLM-Ascend、torch、torch-npu、CANN、
triton-ascend 必须使用对应版本组合。

## 4 Download and Authorization / 权重下载与授权

先在 Hugging Face 账号取得模型许可，再交互登录；不要把 Token 写进脚本、
文档、日志或聊天记录。

```shell
hf auth login
hf auth whoami

export MODEL_ID=mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4
export MODEL_PATH=/models/Mistral-Large-3-675B-Instruct-2512-NVFP4
mkdir -p "${MODEL_PATH}"
hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}" --max-workers 8
```

常见授权错误：

- `401/403`：账号未接受许可，或 fine-grained token 未启用 gated repository；
- `Not logged in`：运行用户与登录用户不同，或容器未挂载 Hugging Face 缓存；
- `huggingface-cli ... no longer works`：新客户端改用等价的 `hf download`；
- 镜像站 403：确认 `HF_ENDPOINT`，必要时在合规前提下切回官方端点；
- 下载停在 `.incomplete`：保留目录并重跑 `hf download` 续传，不要提前服务。

下载后按索引校验所有分片，并观察至少三个稳定采样周期：

```shell
test -f "${MODEL_PATH}/params.json" || test -f "${MODEL_PATH}/config.json"
find "${MODEL_PATH}" -type f -name '*.incomplete' -o -name '*.part'
find "${MODEL_PATH}" -maxdepth 1 -name '*.safetensors' | wc -l
du -sh "${MODEL_PATH}"
```

## 5 Deployment / 部署

### 5.1 Mistral 目标部署：A3 TP16 + EP

从 `/workspace` 直接启动：

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

该命令是目标配置，不是本次双卡实测结果。只有完整 Mistral 权重和足够硬件
到位后才能执行并验收。

### 5.2 双卡 TP=2 环境参考流程

两张 910B2C 不能部署 675B，但可用较小 Dense 模型验证 TP/HCCL/API 链路。
本次使用 Qwen-7B-Chat 的成功路径如下：

```shell
export MODEL_PATH=/data/models/Qwen-7B-Chat

vllm serve "${MODEL_PATH}" \
    --served-model-name Qwen-7B-Chat \
    --trust-remote-code \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --port 8000
```

真实日志中出现 `Worker_TP0`、`Worker_TP1`、`Graph capturing finished` 与
`Replaying aclgraph`，两个接口均为 HTTP 200。此结果只能标记为
“同框架双卡适配参考通过”。

### 5.3 上下文长度适配

不要从其他模型复制 `--max-model-len`。先读取当前检查点：

```shell
python - <<'PY'
import json
from pathlib import Path

p = json.loads((Path("/path/to/model") / "config.json").read_text())
for key in ("max_position_embeddings", "seq_length", "model_max_length"):
    if p.get(key) is not None:
        print(key, p[key])
PY
```

若 vLLM 报 `User-specified max_model_len ... greater than derived`，将启动值降到
配置声明的长度。不要用 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 强行绕过；RoPE
超界可能产生 NaN，绝对位置编码可能越界。本次 Qwen 从错误的 32768 调整到
原生 8192 后成功。

## 6 Functional Verification / 功能验证

`Application startup complete` 不是通过条件。必须先 readiness，再发送真实
请求并确认非空输出：

```shell
curl -f http://127.0.0.1:8000/v1/models

curl -f http://127.0.0.1:8000/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "mistral-large-3-nvfp4",
      "messages": [{"role": "user", "content": "请给出数据库迁移故障的前三项应急措施。"}],
      "temperature": 0,
      "max_tokens": 128
    }'
```

通过门禁：两个请求均 HTTP 200，且 `choices[0].message.content` 非空。首个
请求导致 worker 退出属于 false-ready，必须判失败。

## 7 Troubleshooting / 故障排查

### 7.1 NPU 被占用或残留进程

```shell
npu-smi info
ss -ltnp | grep ':8000' || true
ps -eo pid,ppid,stat,cmd | grep -E 'vllm serve|EngineCore|Worker_TP'
```

先核验 PID 命令行，再终止精确服务 PID；不要对未知进程使用宽泛强杀：

```shell
tr '\0' ' ' </proc/<PID>/cmdline
kill <PID>
for i in 1 2 3 4 5; do kill -0 <PID> 2>/dev/null || break; sleep 1; done
```

如果主进程退出但 Worker/EngineCore 残留，保存进程树和日志后再清理。确认
8000 端口空闲、两卡没有旧 Worker，才重新启动。

### 7.2 典型故障矩阵

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `No space left on device` | 403 GB 检查点无法落盘 | 扩容至至少 500 GiB，续传并复核分片 |
| NPU HBM 高占用或 OOM | 残留 Worker、上下文/并发过大 | 核验 PID；降低长度和并发；保留 TP 拓扑 |
| HCCL bind/通信错误 | 端口或 rank 残留、网卡配置错误 | 清理旧进程，核对 HCCL 网卡和所有 rank |
| NVFP4 算子不存在 | 镜像缺少自定义内核 | 更换/构建匹配镜像，不能靠参数绕过 |
| 权重键或 scale 缺失 | 分片不完整或 loader 不匹配 | 核对索引、模型 revision 与适配 commit |
| ACLGraph `507903` | 图捕获序列失效或 shape 压力 | 使用 AIV，降低长度/并发，再以 Eager 隔离 |
| readiness 200 后请求崩溃 | 后端运行时 false-ready | 保存首请求堆栈，使用确定性请求复现 |
| 输出含模板残片 | chat template/tokenizer 不匹配 | 核对 endpoint、模板和 tokenizer mode |

Eager 隔离示例：保持 TP16 和 EP，先降低 shape 压力，并增加
`--enforce-eager`。它是诊断路径，不是默认生产结论。

## 8 Accuracy Evaluation / 精度评测

使用：

```shell
pytest -sv tests/e2e/models/test_lm_eval_correctness.py \
    --config tests/e2e/models/configs/mistral_large3_675b_nvfp4.yaml \
    --tp-size 16
```

YAML 中 GPQA Diamond 的 `0.6717` 是参考目标，不是本次 Ascend 实测结果。
当前 Mistral 用例因硬件容量阻塞，必须在完整真实权重环境中重新评测后，才能
确认或替换该数值。

## 9 Performance / 性能与投产

目标基线为 `max-model-len=131072`、`max-num-seqs=16`、TP16+EP。真实部署
需记录加载时间、TTFT、TPOT、吞吐量、输入输出长度、峰值 HBM/主机内存、
ACLGraph replay、FlashComm1 和所有 rank 健康状态。

本次双卡实例没有执行 Mistral 性能测试；原因是磁盘安全容量 100.74 GiB 小于
约 403 GB 权重，且约 128 GiB 总 HBM 不具备可靠部署余量。不得用 Qwen 的
性能数字外推 Mistral。

## 10 Evidence / 证据与归档

本次 Qwen 参考案例的日志、接口响应和报告已打包并通过 SHA256 校验：

```text
archive: vllm-ascend_run_logs_20260719.tar.gz
sha256: ddd591ecaec016f99ff55f244fd549efbd209646c9f8d42c64f3206052807edb
files: 91
```

关键文件包括 `FINAL_ACCEPTANCE.zh-CN.md`、`MIGRATION_REPORT.zh-CN.md`、
`qwen7b_tp2_service_startup.log`、`qwen7b_models_response.json` 与
`qwen7b_chat_response.json`。

## 11 References / 参考资料

- [Mistral Large 3 675B NVFP4 模型卡](https://huggingface.co/mistralai/Mistral-Large-3-675B-Instruct-2512-NVFP4)
- [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli)
- [vLLM Ascend 安装指南](../../installation.md)
- [vLLM Ascend FAQ](https://docs.vllm.ai/projects/ascend/en/latest/faqs.html)
