---
name: vllm-ascend
description: Adapt, deploy, troubleshoot, validate, and document large MoE or quantized checkpoints on vLLM Ascend. Use for Mistral Large 3 NVFP4 model configuration, Ascend NPU capacity gating, TP/EP service bring-up, real-weight OpenAI API acceptance, SSH migration, evidence archiving, and AI-assisted delivery reports.
---

# vLLM Ascend 模型适配与验收

## 核心原则

1. 先确认仓库、运行时导入路径、模型配置、权重索引和硬件事实，再改代码。
2. 绝不把架构适配 Python 文件当作模型权重；`--model` 必须指向完整检查点
   目录或受支持的模型 ID。
3. 不单独升级 `transformers`。保持 vLLM、vLLM-Ascend、torch、torch-npu、
   triton-ascend 和 CANN 的版本组合一致。
4. Dummy 只能作为快速隔离手段，不能替代真实权重。
5. 真实权重门禁必须同时满足：服务就绪、`/v1/models` HTTP 200、至少一个
   推理请求 HTTP 200 且输出非空。
6. `Application startup complete` 不是通过；首请求崩溃属于 false-ready。
7. 对 EP、FlashComm1、MTP、多模态和 ACLGraph 分别记录“已验证、不支持、
   checkpoint missing、N/A 或未验证”，不要混写。
8. 资源不足时明确标记 `blocked_by_hardware_capacity`，不能用小模型成功冒充
   目标模型成功，也不能把容量阻塞写成软件失败。
9. 操作云实例时先归档并校验交付物，再停服务，最后关机。

## 标准工作流

### 1. 收集事实

```shell
git status --short
git branch --show-current
python -c "import vllm,vllm_ascend;print(vllm.__version__,vllm.__file__);print(vllm_ascend.__file__)"
python -c "import torch,torch_npu;print(torch.__version__,torch_npu.__version__);print(torch.npu.device_count())"
npu-smi info
df -h /data
```

检查模型：

```shell
find "${MODEL_PATH}" -maxdepth 1 -type f -printf '%f %s bytes\n' | sort
python - <<'PY'
import json, os
p = json.load(open(os.path.join(os.environ["MODEL_PATH"], "config.json")))
for k in ("architectures", "model_type", "quantization_config",
          "max_position_embeddings", "seq_length", "num_experts"):
    print(k, p.get(k))
PY
```

### 2. 做资源门禁

比较四类容量：模型仓库总大小、模型盘安全可用空间、主机加载内存、NPU HBM
及 KV cache/图捕获余量。检查点无法落盘时立即停止真实权重部署尝试，记录：

- 文件系统总量、原始可用量和预留日志后的安全上限；
- NPU 数量、型号、单卡 HBM；
- 目标权重预计大小；
- 推荐硬件拓扑；
- 可用的小模型同框架参考，但明确非等价。

本次 Mistral 事实：检查点约 403 GB；测试实例 `/data` 117.56 GiB，安全权重
上限 100.74 GiB；两张 910B2C 每卡 65,536 MiB。结论是磁盘和 HBM 均不足。

### 3. 下载和稳定性校验

```shell
hf auth login
hf download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
```

遇到 401/403 时检查仓库许可、fine-grained token 的 gated repository 权限、
运行用户和 `HF_ENDPOINT`。不要记录 Token。若 `huggingface-cli` 提示废弃，
使用客户端给出的等价 `hf download`，不要为此升级 transformers。

启动前要求 config、索引和索引引用的所有分片存在；`.incomplete`、`.part`、
`.tmp` 或 `.lock` 必须为零。连续多个采样周期确认大小和 mtime 不再变化。

### 4. 启动与上下文适配

从 `/workspace` 直接启动。先使用模型配置声明的上下文长度，不复制其他模型
的值。若 vLLM 报用户长度超过 derived length，应降低到 config 声明值；不要
使用 `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` 强行越界。

双卡 Dense 参考模板：

```shell
vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len "${MODEL_MAX_LEN}" \
  --max-num-seqs 4 \
  --port 8000
```

Mistral 675B NVFP4 目标模板使用 A3 TP16+EP、NVFP4、FlashComm1 和 ACLGraph；
两张 910B2C 只用于较小模型的环境参考验证。

### 5. API 和功能门禁

```shell
curl -f http://127.0.0.1:8000/v1/models
curl -f http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"MODEL","messages":[{"role":"user","content":"say hi"}],"temperature":0,"max_tokens":32}'
```

验证日志中的 Worker rank、权重加载、KV cache、图捕获和首请求。MoE 才检查
EP/FlashComm1；Dense 模型标记 N/A。配置/索引没有 MTP 时标记 checkpoint
missing；文本模型的多模态标记 N/A。

### 6. 故障排查阶梯

1. 以相同参数复现一次并保留第一个错误。
2. 检查端口、残留 APIServer、EngineCore 和 Worker_TP。
3. 检查 config derived max length、模型架构、权重索引和缺失键。
4. 图捕获失败时保持其他参数不变，使用 `--enforce-eager` 隔离一次。
5. OOM 时先降低上下文和并发，不要破坏必要 TP/EP 拓扑。
6. HCCL 错误先清残留 rank、核对网卡和通信端口。
7. NVFP4 自定义算子缺失时更换或构建匹配镜像，不做巨型 CPU/NPU 反量化
   fallback。
8. 首请求失败时按运行时失败处理，不接受 readiness 假阳性。

精确停进程：

```shell
tr '\0' ' ' </proc/<PID>/cmdline
kill <PID>
for i in 1 2 3 4 5; do kill -0 <PID> 2>/dev/null || break; sleep 1; done
```

### 7. 交付和关机顺序

1. 更新 e2e YAML、模型教程、SKILL.md 和提示词归档。
2. 校验 YAML、Markdown、链接、`git diff --check` 和相关测试。
3. 生成一个带 sign-off 的 Conventional Commit。
4. 在服务器归档完整 `run_logs`，生成 SHA256。
5. 下载到本地并对比 SHA256，确认最终报告和接口响应存在。
6. 精确终止推理服务，记录停服时间、PID、端口与 NPU 释放状态。
7. 再次下载停服后的增量日志并校验。
8. 最后执行云实例关机；操作系统关机不一定等于云控制台停止计费，应在控制台
   复核实例状态和计费策略。

## 本次验证结论

### Mistral Large 3 675B NVFP4

- 测试配置和部署教程已提供。
- 真实权重没有下载或加载，因为约 403 GB 权重大于 100.74 GiB 安全磁盘容量。
- 两卡总 HBM 约 128 GiB，不具备可靠部署余量。
- 状态：`blocked_by_hardware_capacity`。
- GPQA `0.6717` 仅为参考目标，不是 Ascend 实测值。

### Qwen-7B-Chat 同框架参考

- 真实权重 8/8 分片，总计 15,442,677,288 字节。
- 2 × Ascend 910B2C，TP=2，`gpu-memory-utilization=0.90`。
- Qwen config `seq_length=8192`；从错误的 32768 调整到 8192 后启动成功。
- `/v1/models` 与 chat 均 HTTP 200，输出非空。
- ACLGraph 捕获和 replay 有日志证据。
- 该结果证明环境、TP/HCCL、NPU 图和 OpenAI API 链路，不证明 Mistral
  NVFP4/EP/FlashComm1。

## 关键提示词归档

以下内容按实际任务阶段归档。所有密码、Token、完整 SSH 用户标识和实例敏感
信息均以 `[REDACTED]` 替换；复用提示词时通过安全凭据渠道注入。

### A. 文件角色与启动参数

```text
区分 vllm_ascend/models 下的 Python 架构适配代码、需要额外下载的模型权重，
说明运行时自动调用关系，并明确 vllm serve --model 应填写完整权重目录或模型 ID。
```

```text
基于 vLLM-Ascend、Ascend NPU 和 Mistral Large 3，校验启动命令，补充显存比例、
最大上下文长度、张量并行参数，并给出 OpenAI 格式 /v1/models 与 chat 示例。
```

### B. 仓库版本、安装与算子适配

```text
查阅当前 vllm-ascend README，确认要求的 vLLM 基准版本；在目标服务器执行
VLLM_TARGET_DEVICE=empty python3 -m pip install -e .，完整保存日志并校验 import。
```

```text
定位 build_aclnn.sh、LightningIndexer 和 SparseFlashAttention 的 socVersion
校验，在支持列表加入 ascend910b2c；保存三处 diff，切换 CANN 8.5.1 对应发布
分支，核对 torch、torch-npu、triton-ascend 后重装并运行模型测试。
```

### C. 双卡实例迁移与环境验证

```text
通过已配置 SSH 接入双卡实例，核验两张 Ascend 910B、CANN、vLLM-Ascend 环境；
迁移完整项目、算子改动、run_logs 和文档，执行 editable install，准备 TP=2
脚本与模型目录，全程记录操作日志。
```

```text
实例重启后记录 npu-smi、torch_npu.device_count、两卡最小张量计算和 vLLM import；
识别两卡则校验 TP=2，识别失败则把故障证据写入迁移报告。
```

### D. 容量门禁与自动监控

```text
把双卡 NPU 验证日志归档到迁移报告，统计 /data 原始与安全可用空间；等待完整
config、索引和权重分片稳定后自动启动 TP=2，保存服务、接口和最终验收日志。
```

```text
Mistral 675B 因模型盘与双卡 HBM 不足无法部署时，明确写为客观硬件限制，不得
声称真实权重测试通过；使用可部署小模型验证同一 vLLM-Ascend 双卡链路。
```

### E. Llama 权重授权隔离

```text
创建 Llama-2-7B 目录并切换 TP=2 脚本；从官方 gated repository 下载配置与
权重。若没有 Meta 许可或 Token 返回 403，记录授权阻塞，不绕过许可，不使用
dummy 冒充真实权重。
```

### F. Qwen 双卡真实权重验收

```text
创建 /data/models/Qwen-7B-Chat；修改现有启动脚本指向该目录，保留 TP_SIZE=2
和 gpu-memory-utilization=0.90；下载 Qwen/Qwen-7B-Chat，等待权重稳定后启动，
调用 /v1/models 和 /v1/chat/completions，补齐 FINAL_ACCEPTANCE。
```

```text
如果 huggingface-cli 已废弃，执行工具建议的 hf download 等价命令，不升级
transformers；若 max_model_len 超过 config derived length，降到模型原生值后
重试，保留首次失败和最终成功日志。
```

### G. 证据归档与下线

```text
将服务器 run_logs 整体压缩，生成 SHA256，下载到本地桌面并复核哈希、文件数、
FINAL_ACCEPTANCE 和 MIGRATION_REPORT；云平台计费截图可选，但日志归档必须完成。
```

```text
完成 Mistral YAML、部署教程和 SKILL.md 后归档全部文档；精确核验并终止推理
PID，确认端口和 NPU 释放，下载最终增量日志，最后下发云实例关机指令。
```

## 可复用交付检查表

- [ ] YAML 包含 `model_name`、`hardware`、`tasks.metrics` 和 `num_fewshot`。
- [ ] 参考精度明确标注来源，不冒充 Ascend 实测。
- [ ] 教程包含环境、下载、部署、接口、精度和性能章节。
- [ ] 理论上下文与实际成功上下文分别记录。
- [ ] 目标模型和小模型参考案例的结论严格分离。
- [ ] ACLGraph、EP、FlashComm1、MTP、多模态逐项给出状态。
- [ ] 完整日志、报告和接口响应已做 SHA256 本地备份。
- [ ] 停服和关机发生在最终备份之后。
