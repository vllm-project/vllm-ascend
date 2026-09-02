# Qwen3.6 35B GDN Phase6 A2/A3 部署与验证操作指南

## 1. 目的和适用范围

本文用于在已经具备昇腾驱动和 CANN 的 Docker 容器中，从源码部署并
验证以下组合：

- vLLM；
- vLLM-Ascend A2/A3/A5 通用 GDN 接入分支；
- `flash-linear-attention-npu`（下文简称 FLA）；
- 本地模型 `/home/weights/Qwen3.6-35B-A3B`；
- A2 或 A3 单机环境；
- eager 模式、普通 prefill 和普通 decode；
- 单 DP、单 TP，即 `DP=1`、`TP=1`。

当前首要目标是证明：

1. FLA wheel 与实际 SoC 匹配；
2. FLA Phase6 融合 GDN 算子和 recurrent 算子可用；
3. vLLM-Ascend 能选择 FLA 后端；
4. Qwen3.6 35B 可以用 `vllm serve` 完成 prefill 和 decode。

当前不验证 MTP、speculative decode 或 ACL Graph。启动服务时必须使用
`--enforce-eager`。

## 2. 固定仓库版本

为了保证三个仓库彼此匹配，首次验证不要直接使用各仓库最新 `main`。

| 仓库 | 分支或提交 | 用途 |
| --- | --- | --- |
| vLLM | `ba07e4a48fc951300d97eb506217dd530583dea3` | vLLM-Ascend 记录的 main verified commit |
| vLLM-Ascend | `a2-a3-gdn-phase6-validation` | A2/A3/A5 通用 GDN 接入 |
| vLLM-Ascend 最低实现提交 | `72a4aed4f997e47e438841999fade8b2a862c3e0` | Phase6 融合路径和跨 SoC 路由 |
| FLA | `chw_new_cumsum_kkt_solve_tri_simple` | 包含 `gdn_core_fwd_phase6` |
| FLA 提交 | `19cccd26186fc2e386b4d7c386de5539006d215a` | 当前验证基线 |

注意：本次检查时，FLA `main` 的 `9ed9351` 不包含
`gdn_core_fwd_phase6`。如果误用该版本，vLLM-Ascend 无法加载当前融合
算子。

## 3. 容器和硬件前置条件

### 3.1 必须由宿主机或容器镜像提供

- 可用的昇腾驱动；
- 与 A2/A3 匹配的 CANN toolkit 和 kernels/ops 包；
- Python 3.11；
- 与 CANN 匹配的 `torch`、`torch-npu`；
- `triton-ascend`；
- `gcc`、`g++`、CMake、Ninja、Git；
- 至少一张容器内可见的 A2 或 A3 NPU；
- `/home/weights/Qwen3.6-35B-A3B` 中完整的模型配置、tokenizer 和
  safetensors 权重。

注意：不同机器的权重目录可能不同。如果 `/home/weights` 不存在，可以先
查找本机可用的权重路径，例如 `/mnt/weight/Qwen3.6-35B-A3B`，然后把
第 10.2 节的 `MODEL_PATH` 指向该实际路径即可。

本文不使用 Conda，所有安装都发生在 Docker 当前 Python 环境中。

### 3.2 加载 CANN 环境

根据容器内的实际安装位置，执行其中一条：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

或者：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
```

确认环境变量已经加载：

```bash
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "ASCEND_OPP_PATH=${ASCEND_OPP_PATH}"
which bisheng
```

### 3.3 选择一张物理卡

例如只使用宿主机物理 3 号卡：

```bash
export ASCEND_RT_VISIBLE_DEVICES=3
```

此后 Python 进程通常把这张可见卡映射为逻辑 `npu:0`。因此本文的
FLA 测试统一使用 `--device 0`。不要同时把它写成 `--device 3`，否则
可能访问不存在的逻辑卡。

先确认实际映射：

```bash
npu-smi info

python3 - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("visible device count:", torch.npu.device_count())
torch.npu.set_device(0)
print("current logical device:", torch.npu.current_device())
print("device name:", torch.npu.get_device_name(0))
PY
```

只有 `device_count` 大于等于 1，且 `set_device(0)` 成功，才能继续。

## 4. Clone 三个仓库

统一放到 `/home/z00886386`：

```bash
cd /home/z00886386

git clone https://github.com/vllm-project/vllm.git

git clone \
  --branch a2-a3-gdn-phase6-validation \
  --single-branch \
  git@github.com:pantszhang/vllm-ascend.git

git clone \
  --branch chw_new_cumsum_kkt_solve_tri_simple \
  --single-branch \
  https://github.com/yjmyl/flash-linear-attention-npu.git
```

如果 A2/A3 环境没有配置 GitHub SSH key，可以把 vLLM-Ascend 地址改为：

```text
https://github.com/pantszhang/vllm-ascend.git
```

固定 vLLM 和 FLA 提交：

```bash
cd /home/z00886386/vllm
git checkout ba07e4a48fc951300d97eb506217dd530583dea3

cd /home/z00886386/flash-linear-attention-npu
git checkout 19cccd26186fc2e386b4d7c386de5539006d215a
```

确认三个版本：

```bash
git -C /home/z00886386/vllm rev-parse HEAD
git -C /home/z00886386/vllm-ascend rev-parse HEAD
git -C /home/z00886386/flash-linear-attention-npu rev-parse HEAD
```

如果仓库已经存在，不要重新 clone。先检查本地改动：

```bash
git -C /home/z00886386/vllm status --short
git -C /home/z00886386/vllm-ascend status --short
git -C /home/z00886386/flash-linear-attention-npu status --short
```

确认没有需要保留的未提交改动后，再使用 `git fetch` 和 `git switch`
更新对应仓库。

## 5. 安装 vLLM

当前 vLLM-Ascend 分支使用其仓库中
`.github/vllm-main-verified.commit` 记录的 vLLM 提交。

先确认容器已有基础构建工具和 PyTorch 环境：

```bash
python3 -m pip show \
  torch torch-npu triton-ascend \
  cmake ninja setuptools setuptools-scm wheel
```

然后以 editable、无依赖覆盖方式安装 vLLM，避免 pip 替换容器中已经
与 CANN 匹配的 `torch` 和 `torch-npu`：

```bash
cd /home/z00886386/vllm

export VLLM_TARGET_DEVICE=empty
python3 -m pip install \
  --no-build-isolation \
  --no-deps \
  -e .
unset VLLM_TARGET_DEVICE
```

验证安装来源：

```bash
python3 - <<'PY'
import vllm

print("vllm version:", vllm.__version__)
print("vllm path:", vllm.__file__)
PY
```

`vllm path` 应指向 `/home/z00886386/vllm`。

## 6. 安装 vLLM-Ascend

### 6.1 设置 vLLM 兼容版本

当前 editable vLLM 来自主干提交 `ba07e4a4`。不要设置 `VLLM_VERSION`
环境变量，让 vLLM-Ascend 按已安装 vLLM 的真实版本号选择分支：

```bash
unset VLLM_VERSION
```

注意：不要设置 `VLLM_VERSION=0.27.1`。本指南固定的 vLLM 提交
`ba07e4a4`（`v0.26.1rc0-1046`）在 v0.27.1 之后，其 pcp 模块已迁移到
`vllm/v1/attention/ops/pcp`。强制 `VLLM_VERSION=0.27.1` 会让
`attention_v1.py` 走旧路径导入，EngineCore 启动时报
`ModuleNotFoundError: No module named 'vllm.model_executor.layers.attention.pcp'`。

### 6.2 设置 vLLM-Ascend 编译 SoC

A2：

```bash
export SOC_VERSION=ascend910b1
```

A3：

```bash
export SOC_VERSION=ascend910_9391
```

这里的 `SOC_VERSION` 是 vLLM-Ascend 编译参数；它和后面 FLA 使用的
`FLA_NPU_SOC` 名字不同。

### 6.3 编译并安装

```bash
cd /home/z00886386/vllm-ascend

git submodule update --init --recursive

python3 -m pip install \
  --no-build-isolation \
  --no-deps \
  -e .
```

验证 plugin 加载位置：

```bash
python3 - <<'PY'
import vllm_ascend
from vllm_ascend.device.device_config import get_fla_gdn_soc

print("vllm_ascend path:", vllm_ascend.__file__)
print("detected FLA GDN SoC:", get_fla_gdn_soc())
PY
```

预期结果：

- A2：`ascend910b`；
- A3：`ascend910_93`。

如果返回 `None`，不要继续启动模型，应先排查设备识别和
`SOC_VERSION`。

## 7. 编译并安装 FLA wheel

### 7.1 安装 FLA Python 构建依赖

```bash
cd /home/z00886386/flash-linear-attention-npu
python3 -m pip install -r requirements.txt
```

这一步不应替换容器已有的 `torch`、`torch-npu` 或 CANN。

### 7.2 设置正确的 FLA SoC

A2：

```bash
export FLA_NPU_SOC=ascend910b
```

A3：

```bash
export FLA_NPU_SOC=ascend910_93
```

干净构建不要设置 `FLA_NPU_SKIP_RUN_BUILD=TRUE`。显式清除可能遗留的
调试变量：

```bash
unset FLA_NPU_SKIP_RUN_BUILD
unset FLA_NPU_SKIP_RUN_INSTALL
unset FLA_NPU_INCREMENTAL_BUILD
unset FLA_NPU_OPS
```

### 7.3 环境预检

```bash
cd /home/z00886386/flash-linear-attention-npu

python3 scripts/check_npu_env.py --build-only
```

预检失败时先解决缺失的 CANN、编译器或 Python 构建依赖，不要用
`FLA_NPU_SKIP_RUN_BUILD=TRUE` 绕过首次构建。

### 7.4 完整编译 wheel

推荐使用仓库脚本：

```bash
cd /home/z00886386/flash-linear-attention-npu

set -o pipefail
python3 scripts/build_wheel.py 2>&1 | tee /tmp/fla-wheel-build.log
```

该脚本内部执行等价的 wheel 构建：

```bash
FLA_NPU_SOC=${FLA_NPU_SOC} \
python3 -m pip wheel \
  --no-build-isolation \
  --no-deps \
  . \
  -w dist
```

成功后，构建日志会打印本轮 wheel 的绝对路径和准确安装命令。

### 7.5 安装准确的 wheel 文件

先列出本轮产物：

```bash
ls -lt /home/z00886386/flash-linear-attention-npu/dist/*.whl
```

从本轮构建日志自动提取准确文件名，不要直接使用可能匹配多个旧产物的
`dist/*.whl`：

```bash
export FLA_WHEEL="$(
  sed -n 's/^\[fla-npu build\] Wheel: //p' /tmp/fla-wheel-build.log |
    tail -n 1
)"

test -n "${FLA_WHEEL}"
test -f "${FLA_WHEEL}"

python3 -m pip install \
  --force-reinstall \
  --no-cache-dir \
  --no-deps \
  "${FLA_WHEEL}"
```

### 7.6 验证 wheel API 和内嵌 OPP

```bash
cd /home/z00886386/flash-linear-attention-npu

python3 scripts/check_packaged_wheel_api.py

python3 - <<'PY'
from pathlib import Path

import fla_npu
from fla_npu.ops import ascendc

package_dir = Path(fla_npu.__file__).resolve().parent
vendor_dir = package_dir / "opp" / "vendors" / "fla_npu_transformer"

required = (
    "gdn_core_fwd_phase6",
    "recurrent_gated_delta_rule",
    "causal_conv1d",
)

print("fla_npu path:", fla_npu.__file__)
print("embedded vendor dir:", vendor_dir)
print("embedded vendor exists:", vendor_dir.is_dir())

for name in required:
    print(name, hasattr(ascendc, name))
PY
```

三个 API 均应为 `True`，并且 `embedded vendor exists` 应为 `True`。
正常完整 wheel 不需要手工设置 `FLA_NPU_OPP_PATH`。

## 8. FLA 单算子和融合算子精度测试

以下所有测试都假设：

```bash
export ASCEND_RT_VISIBLE_DEVICES=3
```

物理 3 号卡映射为逻辑 `device 0`。如果选择其他物理卡，只修改
`ASCEND_RT_VISIBLE_DEVICES`，测试参数仍使用 `0`。

### 8.1 查看 FLA 基础算子测试计划

```bash
cd /home/z00886386/flash-linear-attention-npu/torch_custom/fla_npu/test

bash test.sh --device 0 --mode dry-run
```

### 8.2 运行当前推理相关的基础算子

```bash
cd /home/z00886386/flash-linear-attention-npu/torch_custom/fla_npu/test

bash test.sh --device 0 --op causal_conv1d
bash test.sh --device 0 --op chunk_local_cumsum
bash test.sh --device 0 --op chunk_scaled_dot_kkt
bash test.sh --device 0 --op recompute_w_u_fwd
bash test.sh --device 0 --op gdn_fwd_h
bash test.sh --device 0 --op gdn_fwd_o
```

这些测试用于诊断内部或兼容路径。当前 vLLM-Ascend 正常 prefill 的
首选路径是一个 `gdn_core_fwd_phase6` 调用，不是依次调用六个小算子。

测试日志位于：

```text
/home/z00886386/flash-linear-attention-npu/torch_custom/fla_npu/test/test_output
```

### 8.3 Phase6 融合算子精度和调用次数测试

下面的脚本会比较旧六算子路径和 Phase6 融合路径，要求结果 bit-exact，
并检查旧路径为 6 次 ACLNN 调用、Phase6 为 1 次 ACLNN 调用：

```bash
cd /home/z00886386/flash-linear-attention-npu/torch_custom/fla_npu/test

python3 benchmark_gdn_core_ablation.py \
  --device 0 \
  --batch 1 \
  --key-heads 2 \
  --value-heads 8 \
  --value-dim 128 \
  --tokens 256 \
  --chunk-size 64 \
  --dtype bf16 \
  --warmup 2 \
  --iterations 2 \
  --output-final-state \
  --output /tmp/gdn-phase6-a2-a3.json
```

重点检查 JSON：

- `accuracy.phase6_vs_legacy.bit_exact` 为 `true`；
- `finiteness` 全部为 `all_finite: true`；
- `expected_aclnn_call_count.phase6_one_aclnn_fused_core` 为 `1`；
- 实际 `variants.phase6_one_aclnn_fused_core.aclnn_call_count` 为 `1`。

### 8.4 Phase6 原生 GVA 和非对齐长度测试

```bash
cd /home/z00886386/flash-linear-attention-npu/torch_custom/fla_npu/test

python3 validate_gdn_phase6_gva_dense_t.py \
  --device 0 \
  --key-heads 2 \
  --value-heads 8 \
  --tokens 130 \
  --chunk-size 64
```

重点检查：

- `native_vs_expanded_bit_exact` 全部为 `true`；
- `native_finite` 全部为 `true`。

### 8.5 recurrent 普通 decode 精度测试

FLA 自带 `test_accuracy.py` 的 `main()` 固定使用 `npu:2`。以下命令
直接复用该文件的 golden 和测试函数，但显式选择逻辑 `npu:0`，无需修改
FLA 源码：

```bash
cd /home/z00886386/flash-linear-attention-npu/fla/ops/ascendc/gdn/recurrent_gdn/recurrent_gated_delta_rule/tests/pta

python3 - <<'PY'
import torch
import torch_npu

from test_accuracy import run_test_case

device = torch.device("npu:0")
torch_npu.npu.set_device(device)

passed = run_test_case(
    "ordinary_decode_bs1_mtp1",
    bs=1,
    mtp=1,
    nk=2,
    nv=8,
    dk=128,
    dv=128,
    device=device,
    use_g=True,
    use_gk=False,
    use_accepted_tokens=False,
)

raise SystemExit(0 if passed else 1)
PY
```

这是普通 decode 测试，不代表当前已经支持 MTP。

## 9. vLLM-Ascend GDN 测试

### 9.1 单元测试

```bash
cd /home/z00886386/vllm-ascend

pytest -q \
  tests/ut/device/test_device_config.py \
  tests/ut/ops/test_gdn_fla.py
```

该测试主要验证：

- A2/A3/A5 到 FLA SoC 的映射；
- backend 配置解析；
- Phase6 参数和布局转换；
- recurrent 和状态更新；
- 兼容别名和错误日志。

### 9.2 真实 NPU operator smoke

验证阶段使用严格 FLA 后端，避免算子失败后静默使用 native：

```bash
export VLLM_ASCEND_GDN_BACKEND=fla_npu
unset VLLM_ASCEND_GDN_OP_BACKENDS

cd /home/z00886386/vllm-ascend

pytest -s -q \
  -o log_cli=true \
  --log-cli-level=INFO \
  tests/e2e/nightly/single_node/ops/singlecard_ops/test_gdn_fla.py \
  2>&1 | tee /tmp/vllm-ascend-gdn-fla-operator.log
```

重点检查：

- 测试没有 skip；
- A2 日志显示 `soc=ascend910b`；
- A3 日志显示 `soc=ascend910_93`；
- `gdn_core_fwd_phase6` 被成功解析；
- 没有 `backend=native` 或 unexpected fallback。

## 10. 启动 Qwen3.6 35B：单 DP、单 TP

### 10.1 显存前置条件

`Qwen3.6-35B-A3B` 虽然每个 token 只激活部分专家，但单 TP 仍需要在
一张卡上加载全部模型权重。开始前确认单卡显存能够容纳：

- 全部模型权重；
- vLLM runtime；
- KV cache；
- GDN/FLA workspace。

如果加载阶段 OOM，它首先是 DP1/TP1 资源问题，不能直接判定为 GDN
算子错误。为了降低首次启动的额外显存占用，下面限制上下文长度和并发，
但不会减少模型权重本身的占用。

### 10.2 设置环境

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_RT_VISIBLE_DEVICES=3
unset VLLM_VERSION
export VLLM_LOGGING_LEVEL=INFO

export VLLM_ASCEND_GDN_BACKEND=fla_npu
unset VLLM_ASCEND_GDN_OP_BACKENDS

export MODEL_PATH=/home/weights/Qwen3.6-35B-A3B
export SERVED_MODEL_NAME=qwen36-35b-gdn
```

如果本机权重在其他位置，例如 `/mnt/weight/Qwen3.6-35B-A3B`，直接把
`MODEL_PATH` 改为该路径（或建立软链接），其余命令不变。

如果 CANN 的实际路径带 `latest`，使用第 3.2 节中的对应 source 命令。

### 10.3 启动服务

```bash
cd /home/z00886386/vllm-ascend

set -o pipefail

vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --dtype bfloat16 \
  --data-parallel-size 1 \
  --tensor-parallel-size 1 \
  --max-model-len 2048 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.95 \
  --enforce-eager \
  --trust-remote-code \
  --port 8000 \
  2>&1 | tee /tmp/qwen36-35b-gdn-dp1-tp1.log
```

验证阶段使用 `VLLM_ASCEND_GDN_BACKEND=fla_npu`，目的是让 FLA 缺失、
符号不匹配或 probe 失败直接暴露。完成验收后，如果希望生产运行允许
解析阶段回落，可改为：

```bash
export VLLM_ASCEND_GDN_BACKEND=auto
```

注意：当前 Phase6 首次真实形状 runtime probe 失败时，不会自动重新执行
六小算子链路；该情况仍会报错退出，不能视为成功 fallback。

### 10.4 发送请求

在另一个容器终端执行：

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen36-35b-gdn",
    "prompt": "请用三句话介绍昇腾上的线性注意力。",
    "max_tokens": 64,
    "temperature": 0
  }'
```

至少连续请求两次，以覆盖首次 prefill、连续 decode 和选择缓存：

```bash
curl -s http://127.0.0.1:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen36-35b-gdn",
    "prompt": "GDN 融合算子的主要作用是什么？",
    "max_tokens": 64,
    "temperature": 0
  }'
```

## 11. 服务日志验收

注意：vLLM 默认只给 `vllm` 命名空间的 logger 配置 INFO 级别，
`vllm_ascend` 命名空间的算子选择日志（`GDN FLA operator selected` 等）
默认不会出现在控制台日志里。启动服务前建议额外设置
`VLLM_LOGGING_CONFIG_PATH`，指向一个把 `vllm_ascend` logger 配置为
INFO 的自定义日志配置（JSON，可在 vLLM 默认配置上追加一条
`"vllm_ascend": {"handlers": ["vllm"], "level": "INFO", "propagate": false}`），
否则下面的验收 grep 会一无所获。

```bash
grep -E \
  'GDN FLA|gdn_core_fwd_phase6|recurrent_gated_delta_rule|fallback|ERROR|FAILED' \
  /tmp/qwen36-35b-gdn-dp1-tp1.log
```

必须确认：

1. A2 为 `soc=ascend910b`，A3 为 `soc=ascend910_93`；
2. prefill 选择 `op=gdn_core_fwd`；
3. 具体 symbol 包含 `gdn_core_fwd_phase6`；
4. ordinary decode 选择 FLA `recurrent_gated_delta_rule`；
5. backend 为 `fla_npu`；
6. 请求返回非空 token；
7. 日志中没有 fallback、`161001`、HCCL、AICPU 或非有限值错误。

只看到六个 standalone stage 的解析或初始化日志，不能证明 prefill 使用了
融合核。必须同时看到 `gdn_core_fwd_phase6`，必要时再通过 profiler 确认
设备 kernel 名称为 `ChunkGdnCoreFwd`。

## 12. 验证结果记录表

建议在 A2 和 A3 分别保存以下信息：

| 项目 | A2 | A3 |
| --- | --- | --- |
| 物理 NPU 型号和设备 ID |  |  |
| CANN 版本 |  |  |
| `torch` / `torch-npu` 版本 |  |  |
| vLLM commit |  |  |
| vLLM-Ascend commit |  |  |
| FLA commit |  |  |
| FLA wheel 完整文件名 |  |  |
| `check_packaged_wheel_api.py` |  |  |
| Phase6 bit-exact |  |  |
| Phase6 ACLNN 调用次数为 1 |  |  |
| GVA T=130 |  |  |
| recurrent ordinary decode |  |  |
| vLLM-Ascend unit tests |  |  |
| vLLM-Ascend operator smoke |  |  |
| Qwen3.6 35B DP1/TP1 启动 |  |  |
| 两次 API 请求 |  |  |
| prefill/decode backend 日志 |  |  |

## 13. 常见问题

### 13.1 找不到 `gdn_core_fwd_phase6`

先检查 FLA 分支和提交：

```bash
git -C /home/z00886386/flash-linear-attention-npu \
  log -1 --oneline --decorate
```

然后检查 wheel 安装来源：

```bash
python3 -m pip show flash-linear-attention-npu
```

常见原因是误装了 FLA `main`、安装了旧 wheel，或使用通配符选中了
`dist/` 中上一次构建的文件。

### 13.2 `Unable to find FLA NPU custom OPP`

完整 wheel 应包含：

```text
fla_npu/opp/vendors/fla_npu_transformer
```

优先重新执行完整 wheel 构建和准确文件安装。只有在明确使用外部 OPP
调试时才设置 `FLA_NPU_OPP_PATH`；正常完整 wheel 不依赖该变量。

### 13.3 `libopapi.so` 或 `libcust_opapi.so` 冲突

FLA 自定义算子应使用其 wheel 内嵌 OPP 中的 `libcust_opapi.so`。不要把
FLA 自定义库重命名成 `libopapi.so`，也不要用它覆盖 CANN 自带
`libopapi.so`。

先只读检查：

```bash
python3 - <<'PY'
from pathlib import Path
import fla_npu

root = Path(fla_npu.__file__).resolve().parent / "opp"
for path in root.rglob("lib*opapi.so"):
    print(path)
PY
```

如果发现旧 wheel 遗留的自定义 `libopapi.so`，停止服务进程，重新安装
本指南固定提交生成的完整 wheel。动态库已经被 Python 进程加载后，不能
在同一进程内热替换。

### 13.4 单算子通过，但 vLLM serve 失败

单算子测试通常覆盖固定形状；模型会带入真实 Hk、Hv、K、V、chunk、
varlen metadata 和 state layout。检查服务日志中的真实 runtime signature，
并优先复现 `gdn_core_fwd_phase6`，不要只反复运行 `solve_tri` 等内部阶段。

### 13.5 35B 在 DP1/TP1 加载时 OOM

先检查错误发生在权重加载、KV cache 分配还是首次 GDN 调用：

- 权重加载阶段 OOM：单卡无法容纳完整 35B 权重；
- KV cache 分配阶段 OOM：继续降低 `--max-model-len`、
  `--max-num-seqs` 或 `--gpu-memory-utilization`；
- 首次 GDN 调用 OOM：记录 Phase6 runtime signature 和 workspace 日志。

如果单卡物理容量不足，必须改用更大显存的单卡、量化权重或 TP 大于 1。
这不是通过修改 GDN backend 可以解决的问题。

### 13.6 日志显示 native 或 fallback

验收时确认：

```bash
export VLLM_ASCEND_GDN_BACKEND=fla_npu
unset VLLM_ASCEND_GDN_OP_BACKENDS
```

严格模式仍回落或启动失败时，日志必须包含逻辑算子、具体 FLA symbol、
SoC、失败 stage 和异常首行。保存完整日志，不要只截取最后一行。

## 14. 最终通过标准

某一硬件家族只有同时满足以下条件，才能标记为通过：

- 安装了为该 SoC 编译的 FLA wheel；
- Phase6 与六算子参考路径 bit-exact；
- profiler/调用统计证明 Phase6 是一次 ACLNN 融合调用；
- native GVA 和非对齐长度测试通过；
- recurrent 普通 decode 精度测试通过；
- vLLM-Ascend unit 和 NPU operator smoke 通过；
- Qwen3.6 35B 在 DP1/TP1 eager 模式成功启动；
- 至少两次请求返回非空输出；
- 日志证明 prefill 使用 `gdn_core_fwd_phase6`；
- 日志证明 ordinary decode 使用 FLA recurrent；
- 没有无法解释的 fallback、OOM、HCCL、AICPU 或精度错误。

A2 和 A3 必须分别记录结果。一种 SoC 通过不能自动代表另一种 SoC
通过。
