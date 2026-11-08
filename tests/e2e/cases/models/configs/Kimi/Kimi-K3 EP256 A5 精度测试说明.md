# Kimi-K3 EP256 A5 精度测试说明

对应 PR：[#21117](https://github.com/vllm-project/vllm-ascend/pull/21117)。
测试目标：在单节点 A5 8 卡环境中，评测裁剪至 256 个路由专家的 Kimi-K3 W4A8 MXFP/QuaRot 模型，
使用 GSM8K 测试集前 500 题统计答案准确率。

## 1. PR 改动文件

| 文件 | 改动内容 |
| --- | --- |
| `.github/workflows/configs/nightly_config.yaml` | 注册 A5 单节点测试任务，指定 ARM64 A5 8 卡 runner 和测试 YAML |
| `tests/e2e/cases/models/configs/Kimi/Kimi-K3-W4A8-EP256-A5.yaml` | 新增模型启动参数、数据集配置、请求参数和精度验收值 |
| `tests/e2e/cases/models/configs/Kimi/Kimi-K3 EP256 A5 精度测试说明.md` | 说明改动范围、测试环境、测试参数和验证结果 |

用例名称：`Kimi-K3-W4A8-EP256-A5-GSM8K-First500`。
测试通过项目共享入口 `tests/e2e/common/single_node/test_single_node.py` 执行，
使用现有 AISBench 集成。模型裁剪及权重准备在运行测试前完成。

## 2. 环境配置

### 2.1 硬件与软件

以下版本取自历史 A5 精度测试环境记录。

| 项目 | 配置 |
| --- | --- |
| 硬件 / 架构 | 单节点 A5，8 张 NPU；ARM64 / aarch64 |
| 基础镜像 | `quay.io/ascend/vllm-ascend:nightly-releases-v0.30.0-a5` |
| Ascend 驱动 / CANN Toolkit | `25.6.rc3` / `9.2.0` |
| Python / PyTorch / torch_npu | `3.12.13` / `2.10.0+cpu` / `2.10.0.post7` |
| vLLM / vllm-ascend | `0.30.0+empty` / `0.30.0rc2.dev304+geac9c0cf3.d20261031` |
| Transformers / safetensors | `5.14.1` / `0.8.0` |
| Triton / triton-ascend | `3.6.0` / `3.6.0.post2` |
| AISBench / datasets / evaluate | `3.1.0` / `3.6.0` / `0.4.6` |
| NumPy / PyArrow | `1.26.4` / `20.0.0` |
| 推理容器 / 评测容器 | `c30087248-kimi-k3-slim-container` / `c30087248-kimi-k3-eval-container` |

历史环境将推理和评测分别放在独立容器中，评测使用 `/opt/aisbench-venv`，
通过 host 网络调用推理 API；权重与数据集只读挂载。

### 2.2 历史环境依赖调整

| 组件 | 调整 |
| --- | --- |
| FastAPI / Starlette | `0.123.10 → 0.136.3` / `0.50.0 → 1.0.1` |
| memcache-hybrid / memfabric-hybrid | 均由 `1.3.0` 调整为 `1.2.0` |
| OpenCV headless / contourpy | 调整为 `4.11.0.86` / `1.3.3`，保持 NumPy `1.26.4` |
| FastAPI 源码依赖声明 | 在 vllm-ascend 的 `requirements.txt`、`pyproject.toml` 中对齐 `>=0.133.0,<0.137.0` |
| triton-ascend 源码约束 | 对齐已安装版本 `3.6.0.post2` |
| OpenCV 源码依赖声明 | 在 vLLM 的 `requirements/common.txt` 中固定为 `opencv-python-headless==4.11.0.86` |
| 其他依赖 | 补齐 colorlog、Plotly、Matplotlib、msguard、expecttest 和 FastAPI standard 依赖 |

### 2.3 权重、数据与环境变量

| 项目 | 当前 YAML 配置 |
| --- | --- |
| 模型 | Kimi-K3 W4A8 MXFP/QuaRot，256 个路由专家 |
| 模型路径 | `/mnt/weight/modelscope/hub/models/CuiZ2007/Kimi-K3-w4a8-mxfp-QuaRot-experts256` |
| 数据集路径 | `/mnt/weight/modelscope/hub/datasets/CuiZ2007/gsm8k` |
| 数据文件 | `test.jsonl`：前 500 题；`train.jsonl`：4-shot 示例 |
| 服务端口 / Runner | `SERVER_PORT=8080` / `VLLM_USE_V2_MODEL_RUNNER=1` |
| 通信缓冲 / CPU 线程 | `HCCL_BUFFSIZE=800` / `OMP_NUM_THREADS=1` |
| 内存分配 | `PYTORCH_NPU_ALLOC_CONF=expandable_segments:True` |
| 就绪 / 执行超时 | `VLLM_ENGINE_READY_TIMEOUT_S=7200` / `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3000` |

运行前将权重和数据集挂载到 YAML 指定路径，配置目标机器的通信 IP、网卡和 8 个 NPU 设备，
并取消设置 `HCCL_OP_EXPANSION_MODE`。数据集目录末尾不加斜杠。

## 3. 测试参数与执行方式

| 类型 | 参数 | 配置 |
| --- | --- | --- |
| 服务 | 并行 / 执行模式 | TP8 + EP / eager |
| 服务 | 数据类型 / 量化 | `bfloat16` / `ascend` |
| 服务 | 上下文 / 并发上限 / 批量 token 上限 | `6144` / `100` / `2048` |
| 服务 | 显存利用率 / Chunked prefill / Prefix caching | `0.85` / 开启 / 关闭 |
| 服务 | 分词器 / reasoning parser | `kimi_k3` / `kimi_k3` |
| 服务 | 生成配置 / FlashComm1 | `vllm` / 开启 |
| 请求 | 测试题数 / 客户端并发 / 输出上限 | `500` / `60` / `1024 tokens` |
| 请求 | temperature / top_p / seed / thinking_effort | `0` / `1.0` / `42` / `low` |
| 评分 | 当前 YAML 评分方式 | AISBench 原生 GSM8K 4-shot CoT |
| 验收 | 基线 / 容差 / 通过范围 | `60.0%` / `±3.0` 个百分点 / `57.0%～63.0%` |

在已准备依赖的测试环境中，从项目根目录执行：

```bash
export CONFIG_BASE_PATH=tests/e2e/cases/models/configs/Kimi
export CONFIG_YAML_PATH=Kimi-K3-W4A8-EP256-A5.yaml
unset HCCL_OP_EXPANSION_MODE
python -m pytest -sv tests/e2e/common/single_node/test_single_node.py
```

## 4. 测试结果

### 4.1 历史 A5 三轮精度结果

历史测试使用独立 AISBench 环境和自定义最终答案评分。
每轮覆盖相同的 500 题；最终答案无效或截断时计为错误。

| 轮次 | 正确题数 | 准确率 | 无效最终答案 | 请求失败 |
| --- | --- | --- | --- | --- |
| 第 1 轮 | 312 / 500 | 62.4% | 106 | 0 |
| 第 2 轮 | 289 / 500 | 57.8% | 141 | 0 |
| 第 3 轮 | 313 / 500 | 62.6% | 104 | 0 |
| 汇总 | 平均 304.67 / 500 | 平均 60.93% | — | 合计 0 |

准确率范围为 57.8%～62.6%，极差为 4.8 个百分点。
第 1、2 轮共用服务，第 3 轮重启服务；每轮均核对了完整题号、原始问题和独立评分。

### 4.2 当前 PR 验证进度

| 检查项 | 结果 |
| --- | --- |
| YAML 解析、启动参数类型和内嵌 JSON 检查 | 通过 |
| nightly 任务注册及配置路径检查 | 通过 |
| Git 空白检查 | 通过 |
| 数据准备 | 测试集前 500 题比对通过，训练集 7473 条 |
| 完整格式检查 `bash format.sh ci` | 待完成，环境缺少 `pre-commit` |
| 当前 YAML 的 A5 服务启动、推理及 500 题评测 | 待执行 |
| nightly CI 运行 | 待执行 |
| 原生评分验收值校准 | 待执行；当前采用用户选定的 60.0% ±3.0 个百分点 |
