# Kimi-K3 减至 256 专家的 GSM8K 精度回归测试

[English](README.md)

更新日期：2026-11-03。分支：`test/kimi-k3-pruned-experts-accuracy`。
本文合并运行说明、改动清单和测试报告，与英文版采用相同章节结构。

## 1. 用例范围与交付状态

**单节点 A5 8 卡，Kimi-K3 W4A8 减至 256 专家，GSM8K 前 500 题精度回归测试。**
单次执行只运行一轮；历史三轮用于观察波动，不要求每次 nightly 重复三轮。

pytest 入口、裁剪及评分工具、nightly 注册已在源码中实现。
18 项 CPU 单元测试、Ruff 检查和矩阵解析通过；尚未在目标 NPU/AISBench 环境验证新入口。
历史远端实测结果不能当作当前分支验证通过的证据。

## 2. 文件清单

以下路径相对于本文。工具位于 `tools/kimi_k3`，E2E 配置和入口位于
`tests/e2e/cases/models`，单元测试位于 `tests/ut/tools/kimi_k3`。

| 文件 | 用途 |
| --- | --- |
| [prune_experts.py](prune_experts.py) | 保留原始字节的裁剪、计划生成与分片校验 |
| [test_prune_experts.py](../../tests/ut/tools/kimi_k3/test_prune_experts.py) | 8 项裁剪工具测试 |
| [serve_ep256.sh](serve_ep256.sh) | 8 卡服务启动参数 |
| [gsm8k_ep256_first500.py](../../tests/e2e/cases/models/configs/Kimi/gsm8k_ep256_first500.py) | AISBench 前 500 题配置及原始 4-shot 提示词 |
| [accuracy_api.py](accuracy_api.py) | 非流式最终答案通道适配 |
| [final_answer.py](final_answer.py) | 最终通道恢复与严格数值提取 |
| [test_final_answer.py](../../tests/ut/tools/kimi_k3/test_final_answer.py) | 4 项最终答案处理测试 |
| [accuracy_case.py](accuracy_case.py) | 模型、数据校验与独立结果审计 |
| [test_accuracy_case.py](../../tests/ut/tools/kimi_k3/test_accuracy_case.py) | 6 项输入、结果约定测试 |
| [test_kimi_k3_ep256_accuracy.py](../../tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py) | 可执行 pytest 精度回归入口 |
| [nightly_config.yaml](../../.github/workflows/configs/nightly_config.yaml) | A5 nightly 矩阵注册 |
| [nightly-plan.yaml](nightly-plan.yaml) | 用例约定元数据，并非可执行模型 YAML |
| [summarize_accuracy.py](summarize_accuracy.py) | 三轮均值、标准差及极差统计 |
| [**init**.py](__init__.py) | 工具包导入入口 |

## 3. 权重裁剪及校验

### 本次使用的原始权重版本

裁剪输入为 **Kimi-K3 W4A8 MXFP QuaRot 权重版本**，本地标识为
`Kimi-K3-w4a8-mxfp-QuaRot`，保留完整 93 层、每个 MoE 层 896 个路由专家。
原始权重目录为 `/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot`；
裁剪发布目录为 `/mnt/share/weights/Kimi-K3-w4a8-mxfp-QuaRot-slim_ep_256xps`。

本次实验使用原始权重的复制副本，也可以直接使用原始权重目录进行裁剪。
裁剪清单记录的源文件身份校验值如下：

| 源文件 | SHA256 |
| --- | --- |
| `config.json` | `07677f14bc468c16a788fc6517f7fa715701fa1c3ff74b3f7ef3449a5bc8c735` |
| `model.safetensors.index.json` | `b21a53e0b92c0bb6474be2da74ceb9261ba888f8880a8b8f84fb8b8d6ee6e4e1` |

现有记录未提供该本地权重对应的上游仓库 revision 或发布 tag。
以上目录标识和校验值用于定位本次实验输入；校验值覆盖配置、索引文件，
并非全部源权重数据的哈希。第 8 节的 vLLM、vllm-ascend 版本属于运行环境版本，
不属于原始权重版本。

输入原始量化权重；每个 MoE 层从 896 个路由专家中均匀选取 256 个：
`keep_ids[j] = floor(j * 896 / 256)`，其中 `j=0..255`。
92 个 MoE 层均采用此规则；专家编号重映射为 0～255，Router 权重和校正 bias
按同一清单切片。量化权重及 scale 保留原始字节，不反量化或重新量化。
完整保留 93 层、每 token 激活 16 个专家和两个共享专家。
这是均匀容量裁剪，不是基于重要性校准的选择，也不保证精度保持不变。

```bash
python -m tools.kimi_k3.prune_experts \
  --src /models/source --dst /models/ep256.partial --work /work/pruning \
  --keep-experts 256 --mode plan
python -m tools.kimi_k3.prune_experts \
  --src /models/source --dst /models/ep256.partial --work /work/pruning \
  --keep-experts 256 --mode run --workers 4
```

命令中的路径需要替换。保留专家映射、复制计划和分片校验记录，确认全部完成后再发布
临时输出为最终权重。个人服务器 Docker 发布及挂载脚本未迁入工具。
历史裁剪报告记录：358 个输出分片、144683 个张量，保留张量通过逐字节校验。
复制计划摘要为：
`6ed94726e172e3b67c344901a70383451308446f6f346cd64213bd0094533f1e`。
这不是完整权重哈希；整理后的工具尚未重新处理全量权重。

## 4. 固定配置与评分规则

| 配置项 | 取值 |
| --- | --- |
| 硬件 | 单节点 A5，8 张可用 NPU |
| 模型 | W4A8 MXFP QuaRot，93 层，256 个路由专家 |
| 并行及执行模式 | TP8＋EP、MRV2、eager |
| 上下文／最大序列数／批处理 token | 6144／100／2048 |
| 显存利用率 | 0.85 |
| prefix cache／flashcomm1 | 关闭／启用 |
| 数据与提示词 | 原始 GSM8K 测试集 `[0:500]`，保留原始 4-shot 提示词 |
| 客户端并发／输出上限 | 60／1024 token |
| 生成参数 | temperature=0、top_p=1、seed=42、thinking_effort=low |
| API | 非流式，模型别名 `kimi-k3-ep256` |

原始 test.jsonl 的 SHA256：
`3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14`。
数据加载器同时需要 train.jsonl；原始 GSM8K 数据未修改。

仅评分显式最终数值答案或纯数值最终响应。输出截断、缺失或无效最终答案计错，
不从 500 题分母中排除。不采用推理中的最后一个数字作为答案。
可恢复完整的 XTML response 通道，不能将任意 reasoning 当作最终答案。
适配器直接从仓库导入，无需覆盖 site-packages。通用 GSM8K 预设不等价于本次评分流程。

## 5. 执行精度回归用例

在兼容的 Linux A5 环境预装 vLLM、vllm-ascend、AISBench 和 Kimi tokenizer 依赖，
确保 8 张 NPU 可用；本工具不执行依赖安装。切换到仓库根目录，配置四项必需变量：

| 变量 | 要求 |
| --- | --- |
| `KIMI_EP256_MODEL` | 已准备的 93 层、256 专家权重绝对路径 |
| `KIMI_GSM8K_DATASET` | 原始 train.jsonl、test.jsonl 所在目录的绝对路径 |
| `KIMI_EP256_BASELINE` | 经确认的准确率参考值，以百分数表示，范围 [0,100] |
| `KIMI_EP256_TOLERANCE` | 经确认的绝对容差，单位为百分点，范围 [0,100] |

```bash
python -m pytest -sv tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py
```

入口先校验输入，通过共享 `RemoteOpenAIServer` 分配空闲端口并启动自己的服务，
执行 AISBench，核对 500 个唯一题号、原题对应关系，独立重算准确率并检查请求成功情况。
验收规则为 `abs(accuracy - baseline) <= tolerance`，采用双向偏差比较。
缺少输入或验收参数会明确失败；服务由共享框架清理。
历史均值和极差仅供参考，不自动作为验收标准。

每次运行的日志、逐题输出、结果和 `accuracy-summary.json` 位于
`benchmark_results/Kimi-K3-W4A8-EP256-A5/` 下的独立目录。
将该目录归档为 CI 或 PR 证据。请求失败使本轮验证不成立；无有效最终答案仍按错题计分。

## 6. 独立复现与三轮重复测试

手动运行时选择空闲端口，在仓库根目录执行：

```bash
export KIMI_EP256_MODEL=/models/ep256
export KIMI_GSM8K_DATASET=/datasets/gsm8k
export KIMI_API_PORT=2000
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
bash tools/kimi_k3/serve_ep256.sh
```

```bash
ais_bench tests/e2e/cases/models/configs/Kimi/gsm8k_ep256_first500.py \
  --mode all --num-warmups 0 --max-num-workers 1 --dump-eval-details \
  --work-dir /results/round1
```

AISBench 命令在另一个终端执行，导出相同变量。三轮测试按顺序执行，
后两轮分别使用新的 `/results/round2`、`/results/round3` 目录，分析重复性时保持同一服务。
可选 `KIMI_API_HOST` 默认为 127.0.0.1，`KIMI_API_PORT` 默认为 2000。
不要用 `--num-prompts` 替代明确的 `[0:500]` 选择。
下列 TIMESTAMP、MODEL 需要替换为实际路径；统计工具要求输入三份完整结果：

```bash
python -m tools.kimi_k3.summarize_accuracy \
  /results/round1/TIMESTAMP/results/MODEL/gsm8k.json \
  /results/round2/TIMESTAMP/results/MODEL/gsm8k.json \
  /results/round3/TIMESTAMP/results/MODEL/gsm8k.json \
  --output /results/three-round-summary.json
```

## 7. nightly 注册及本地检查

注册文件：`.github/workflows/configs/nightly_config.yaml`；位置：`a5.single_node.test_config`。

```yaml
- name: Kimi-K3-W4A8-EP256-A5
  os: linux-aarch64-a5-8
  tests: tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py
```

选择命令为 `/nightly Kimi-K3-W4A8-EP256-A5`，取决于 PR 工作流和 runner 可用性。
CI 测试容器内需要配置上述四项必需变量。`nightly-plan.yaml` 仅为用例元数据，
并非执行用的模型 YAML。原有四节点完整 K3 用例保持不变；源码注册不代表 CI 资源已准备好。

```bash
python -m unittest discover -s tests/ut/tools/kimi_k3 -v
python -m ruff check tools/kimi_k3 tests/ut/tools/kimi_k3 \
  tests/e2e/cases/models/test_kimi_k3_ep256_accuracy.py \
  tests/e2e/cases/models/configs/Kimi/gsm8k_ep256_first500.py
```

已通过：8 项裁剪测试、4 项最终答案测试、6 项精度约定测试，以及 Ruff 检查和矩阵入口解析。
这些 CPU 验证不代表 NPU 集成验证通过。

## 8. 历史三轮测试结果与总结

历史远端测试使用旧开发容器：vLLM `0.30.0+empty`、vllm-ascend
`0.30.0rc2.dev304+geac9c0cf3.d20261031`、torch `2.10.0+cpu`、torch_npu
`2.10.0.post7`、transformers `5.14.1`。
该结果未验证当前分支 HEAD，也未验证另行下载的正式 v0.30.0-a5 镜像。

三轮均独立核验了 500 个唯一题号 0～499、原题对应及评分一致性。
第 501 题及之后的 819 题续跑数据、此前零记录的第三轮尝试均未计入。

| 轮次 | 正确数 / 500 | 正确率 | 无有效最终答案，计错 | 请求失败 |
| --- | --- | --- | --- | --- |
| 第一轮 | 312 / 500 | 62.4% | 106 | 0 |
| 第二轮 | 289 / 500 | 57.8% | 141 | 0 |
| 第三轮 | 313 / 500 | 62.6% | 104 | 0 |

平均正确率：**60.93%**（914/1500 次重复作答正确）。
波动范围：**57.8%～62.6%**；极差：**4.8 个百分点**。
总体标准差：**2.22 个百分点**；样本标准差：**2.72 个百分点**。
三轮是同一组 500 题的重复评测，不是 1500 道不同题目。

证据路径如下：后续行相对于第一行根目录；换行显示的路径需要拼接后使用。

```text
/mnt/share/c30087248/kimi-k3-slim_lp/component-repair-20261031-134927/watch_ep256/runs/
  ep256-19195a7468a7/accuracy_full/20261103_003449/results/
    kimi-k3-ep256-v2-eager-greedy-low-first500-o1024-c60-final/gsm8k.json
  ep256-19195a7468a7/first500_repeats/round2/20261103_011356/results/
    kimi-k3-ep256-v2-eager-greedy-low-first500-o1024-c60-final/gsm8k.json
  ep256-44f5b8764afc/accuracy_full/20261103_054335/results/
    kimi-k3-ep256-v2-eager-greedy-low-first500-o1024-c60-final/gsm8k.json
  ep256-19195a7468a7/first500_repeats/summary.json
```

第一、二轮复用同一服务，第三轮重新启动了服务。固定 seed、采用贪心生成仍出现波动，
其根因尚未确定。无有效最终答案是错误的重要组成部分，应继续区分截断、格式及生成行为，
不能全部算作请求失败。缺少完整原始权重对照，无法量化裁剪导致的精度损失。
已观察极差不是确认后的验收容差；以上证据仅支持历史实验，不能宣称当前分支或 nightly 已通过。
