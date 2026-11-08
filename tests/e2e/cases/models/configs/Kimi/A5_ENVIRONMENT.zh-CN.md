# Kimi-K3 A5 历史精度测试环境、镜像与修改记录

本文记录此前单节点 A5 8 卡、Kimi-K3 W4A8 256 专家 GSM8K 精度测试的环境调整。
这些是历史开发环境的修改记录，不代表当前分支或正式镜像已完成验证。
历史结果见[源测试报告](https://github.com/CuiZ2007/vllm-ascend/blob/2f1679281e53450a907490c042118568185b3bc4/tools/kimi_k3/TEST_REPORT.md)。

整理日期：2026-11-05。目标分支：`test/kimi-k3-ep256-yaml`。

资料固定于源分支 `test/kimi-k3-pruned-experts-accuracy` 的提交
`2f1679281e53450a907490c042118568185b3bc4`，来源为
[环境报告](https://github.com/CuiZ2007/vllm-ascend/blob/2f1679281e53450a907490c042118568185b3bc4/tools/kimi_k3/TEST_REPORT.md)和
[容器修改记录](https://github.com/CuiZ2007/vllm-ascend/blob/2f1679281e53450a907490c042118568185b3bc4/tools/kimi_k3/A5_CONTAINER_CHANGES.zh-CN.md)。
本文依据仓库记录整理，未重新连接 A5 容器验证，也未执行任何依赖修改。

## 1. 基础镜像与容器分工

基础镜像为 ARM64 / aarch64 版本：

```text
quay.io/ascend/vllm-ascend:nightly-releases-v0.30.0-a5
```

服务器镜像 ID：

```text
sha256:4e34d514c73d903e1860a9034275dd855c14511bf35a01f75601ae25980610e6
```

归档路径：

```text
/mnt/share/c30087248/vllm-ascend_images/vllm-ascend-nightly-releases-v0.30.0-a5-arm64.oci.tar
```

镜像层比对确认：上述镜像的 18 个基础层完整构成本地快照
`c30087248/kimi-k3-slim:before-components-20261031`（23 层）的前缀。
后者 ID 为 `sha256:a8fc248b64bf4031d508b764bbb4bd7808bd0687980d4be9341157f2dbbc0c2e`。
后续依赖安装和源码修改发生在容器内，不能将修改后的软件版本视为原始 nightly 镜像版本。
历史测试没有换用另行下载的正式 `v0.30.0-a5` 镜像。

| 位置 | 职责与调整 |
| --- | --- |
| 宿主机 | 沿用 Docker、Ascend 驱动与 npu-smi；未在全局 Python 安装评测依赖 |
| `c30087248-kimi-k3-slim-container` | 推理服务、NPU 运行栈及依赖兼容修复 |
| `c30087248-kimi-k3-eval-container` | 独立 AISBench 环境，通过 host 网络调用推理 API，不映射 NPU |

权重与数据集采用只读挂载；评测容器中的权重目录用于配置和分词信息访问。

### 宿主机与挂载记录

| 项目 | 配置 |
| --- | --- |
| 服务器 | `141.61.41.213` |
| 硬件 | 单节点 A5，8 张 NPU |
| 架构 | ARM64 / aarch64 |
| Ascend 驱动 | `25.6.rc3` |
| 裁剪权重 | `/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot-slim_ep_256xps` |
| 当前 GSM8K 数据目录 | `/mnt/share/c30087248/datasets/gsm8k` |

环境快照确认容器内可访问 `/dev/davinci0` 至 `/dev/davinci7`，以及
`/usr/local/Ascend/driver`。容器运行还需要可读的权重和数据集挂载。
快照中的历史数据挂载路径为 `/mnt/share/c30087248/gsm8k`。
宿主机操作系统、Docker 版本和完整容器创建参数尚未从本次证据核实。

### 历史推理容器软件版本

| 组件 | 版本 |
| --- | --- |
| Python | `3.12.13` |
| CANN Toolkit | `9.2.0` |
| PyTorch | `2.10.0+cpu` |
| torch_npu | `2.10.0.post7` |
| vLLM | `0.30.0+empty` |
| vllm-ascend | `0.30.0rc2.dev304+geac9c0cf3.d20261031` |
| Transformers | `5.14.1` |
| Triton | `3.6.0` |
| triton-ascend | `3.6.0.post2` |
| safetensors | `0.8.0` |

PyTorch 的 `+cpu` 是发行版本标签，NPU 支持由 `torch_npu` 提供。
vLLM 与 vllm-ascend 分别以可编辑方式安装自 `/vllm-workspace/vllm`
和 `/vllm-workspace/vllm-ascend`。

## 2. V2 model runner 与服务启动调整

V2 通过环境变量选择，不是为了切换 runner 而修改其源码：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

保存的 V1/V2 对比记录确认：软件包、模型元数据、Git 提交和源码 diff 哈希相同，
该次对比记录的环境差异仅为 `VLLM_USE_V2_MODEL_RUNNER` 从 `0` 改为 `1`。
这不表示基础镜像从未做过其他依赖修改。

| 项目 | 最终精度测试使用的配置 |
| --- | --- |
| 并行 | `--tensor-parallel-size 8 --enable-expert-parallel` |
| 执行模式 | `--enforce-eager` |
| Prefix caching | `--no-enable-prefix-caching` |
| 上下文长度 | `--max-model-len 6144` |
| 服务并发上限 | `--max-num-seqs 100` |
| Prefill token 预算 | `--max-num-batched-tokens 2048 --enable-chunked-prefill` |
| 显存利用率 | `--gpu-memory-utilization 0.85` |
| 数据类型与量化 | `--dtype bfloat16 --quantization ascend` |
| 分词器与推理解析器 | `--tokenizer-mode kimi_k3 --reasoning-parser kimi_k3` |
| 生成配置 | `--generation-config vllm` |
| 文本模式 | `--mm-processor-cache-gb 0 --limit-mm-per-prompt '{"vision_chunk":0}'` |
| 通信优化 | `--additional-config '{"enable_flashcomm1":true}'` |
| 服务随机种子 | `--seed 42` |

客户端并发为 60、每题输出上限 1024 token、temperature 为 0，thinking effort 为 low。
这些是请求配置，不能全部当作服务启动参数。新单文件 YAML 的模板默认参数写法仍待目标环境验证。
历史较早的预检配置曾使用并发上限 30，不能与最终 100 的配置混为一谈。

## 3. 通信、内存与超时环境变量

以下是历史服务器上的配置，其中 IP 和网卡名需要按部署机器核实：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_IF_IP=141.61.41.213
export HCCL_SOCKET_IFNAME=eth2
export GLOO_SOCKET_IFNAME=eth2
export HCCL_BUFFSIZE=800
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export VLLM_ENGINE_READY_TIMEOUT_S=7200
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000
unset HCCL_OP_EXPANSION_MODE
```

服务脚本还将临时目录、Hugging Face、vLLM、Triton、TorchInductor 缓存及日志转到共享盘，
设置 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`。
`VLLM_RPC_BASE_PATH=/tmp/k3ipc` 使用较短路径保存本地进程通信文件。
这些设置属于运行环境调整，不是容器镜像代码补丁。

## 4. 主推理容器依赖修改

| 组件 | 原状态 | 调整后的版本 |
| --- | --- | --- |
| FastAPI | `0.123.10`，不满足当时 vLLM 要求 | `0.136.3` |
| Starlette | `0.50.0`，存在依赖冲突 | `1.0.1` |
| memcache-hybrid / memfabric-hybrid | 均为 `1.3.0` | 均为 `1.2.0` |
| NumPy | `1.26.4` | 保持 `1.26.4` |
| OpenCV headless | `5.0.0.93` 要求 NumPy 2 | `4.11.0.86` |
| contourpy | 安装时解析到 `1.4.0`，要求 NumPy 2 | `1.3.3` |
| Triton-Ascend | 已有 `3.6.0.post2` 与源码约束不一致 | 保留 `3.6.0.post2`，调整源码约束 |

补齐 colorlog `6.12.0`、Plotly `7.1.0`、Matplotlib `3.11.2`、msguard `0.0.8`、
expecttest `0.3.0` 以及 FastAPI standard 所需缺失依赖。
这些是历史环境补齐内容，并非已证明每一项都是纯精度测试的最小必需依赖。

PyTorch 保持 `2.10.0+cpu`，torch_npu 保持 `2.10.0.post7`，Transformers 保持 `5.14.1`，
FLA NPU 按组件修复报告保持 `26.9.1`。PyTorch 的 `+cpu` 为发行标签，NPU 支持由 torch_npu 提供。

## 5. 实际修改的源码依赖声明

| 容器内文件 | 修改 |
| --- | --- |
| `/vllm-workspace/vllm-ascend/requirements.txt`、`pyproject.toml` | FastAPI 约束由 `<0.124.0` 调整为 `>=0.133.0,<0.137.0`；Triton-Ascend 固定版本按实际环境调整为 `3.6.0.post2` |
| `/vllm-workspace/vllm/requirements/common.txt` | OpenCV 依赖改为 `opencv-python-headless==4.11.0.86` |

修改后通过 editable 安装重新生成包元数据，没有手工修改 METADATA 或屏蔽 pip check。
OpenCV 调整用于兼容 NumPy 1.26.4，不满足当时上游原始 `>=4.13.0` 声明，
属于本地文本推理兼容配置，未验证完整图像或视频推理。

安装时关闭 Ascend 自定义算子编译；前后已有 vLLM、vllm-ascend `.so` 文件的 SHA256 一致。
vLLM 仍为 `0.30.0+empty`，vllm-ascend 为 `0.30.0rc2.dev304+geac9c0cf3.d20261031`。

## 6. 独立评测环境

评测源码位于容器 `/opt/aisbench`，使用 `/opt/aisbench-venv` 独立虚拟环境。
来源为服务器已有 `/home/jiguang/inference/benchmark`，提交
`31ee7e65b312ecaa5f91b7d89dc6d36a7e5924e2`，保留并记录了已有配置修改。

| 组件 | 最终版本 |
| --- | --- |
| AISBench | `3.1.0` |
| datasets / evaluate | `3.6.0` / `0.4.6` |
| NumPy / OpenCV headless | `1.26.4` / `4.11.0.86` |
| PyArrow | `20.0.0` |
| PyTorch | `2.10.0`，ARM64 CPU wheel |
| Transformers / tiktoken | `5.14.1` / `0.14.0` |
| OpenAI SDK / NLTK | `3.42.0` / `3.8` |

PyArrow `26.0.0` 实际导入要求 NumPy 2，因此固定为 `20.0.0`。
评测调用虚拟环境 Python，不使用基础镜像全局依赖集合。

## 7. 不应混入纯精度测试的临时采集补丁

此前为 logits、路由及专家范数等中间数据采集做过临时源码埋点与实机验证。
这些采集补丁在相关验证结束后已回退，不属于此次仅统计最终答案正确率的必需修改。
不得把采集候选文件存在误认为三轮精度测试启用了这些补丁。

## 8. 验证边界与证据

组件修复阶段记录了两个环境最终 pip check 通过、主要模块导入通过、8 张 NPU 可枚举、
HAL/HCCL/ACL 动态库加载通过，已有算子二进制哈希保持一致。
这些安装检查本身不等同于真实模型精度验证。
历史三轮精度结果与新单文件 YAML 使用的默认评分口径需要区分，不能直接共用基线。

本地看护工作区保存的依据如下，不随本文重复上传：

- `outputs/组件补齐与依赖修复报告.md`
- `outputs/component-repair-evidence/reports/ascend-dependencies.patch`
- `outputs/component-repair-evidence/reports/vllm-numpy-compat.patch`
- `outputs/component-repair-evidence/reports/main-readiness/environment.json`
- `outputs/component-repair-evidence/reports/final-state.json`
- `outputs/v2-eager-retest/ep256-2aeb5dcaaf3f/execution-comparison.json`
- `outputs/ep256-watch-plan/start_service.sh`

本文是历史环境审计记录，不是完整 Dockerfile，也未证明只设置 V2 就能在干净镜像中复现。

## 9. 当前 YAML 的使用边界

当前分支的 A5 用例是 `Kimi-K3-W4A8-EP256-A5.yaml`，位于本文同目录。
它采用原生 AISBench 4-shot 评分；上述历史三轮使用独立评测容器与自定义最终答案评分。
历史 AISBench 3.1.0 的存在已由独立评测环境记录补全，不能因推理环境快照未安装
AISBench 就认定历史测试没有评测环境。

当前 YAML 的模型路径为
`/mnt/weight/modelscope/hub/models/CuiZ2007/Kimi-K3-w4a8-mxfp-QuaRot-experts256`；数据集路径为
`/mnt/weight/modelscope/hub/datasets/CuiZ2007/gsm8k`（无末尾斜杠）。
运行前应将模型和数据集挂载到 YAML 指定的位置，数据集在容器内保持相同绝对路径。
该独立目录包含原始测试集前 500 题的 `test.jsonl` 和用于 4-shot 示例的 `train.jsonl`，
保留原有完整 GSM8K 与 gsm8k500 目录，不修改 `tools/aisbench.py`。
当前基线 60.0%、容差 3.0 个百分点为用户选定值，尚未通过原生评分校准。
nightly 注册仅声明调度入口，不代表运行环境和模型已验证。

镜像标签、镜像 ID 和仓库 manifest digest 的含义不同；本文记录的是历史服务器镜像 ID，
没有将其当作可直接 `docker pull` 的 registry digest。历史 nightly 标签可能更新，
严格复现需要原归档或经确认的同一镜像及容器修改记录。
