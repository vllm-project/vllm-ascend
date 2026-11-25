# Gemma4 E2B / E4B

## 模型与范围

`google/gemma-4-E2B-it` 和 `google/gemma-4-E4B-it` 是支持文本、图片、
音频输入的轻量级 BF16 模型，不是 Gemma4 26B MoE 或 31B Dense。
本文针对 Atlas A3；其他变体见 [Gemma4](Gemma4.md)。

模型使用 256 维滑窗注意力、512 维全局注意力以及跨层 KV 共享。
共享层首次 prefill 没有自己的 K/V，必须读取生产层已经写入的分页缓存；
不能因为状态是 `PrefillNoCache` 就改读当前输入 K/V。

| 特性 | 本文验证范围 |
| ---- | ------------ |
| 权重 | E2B/E4B 完整 BF16 权重；非 dummy、非裁剪层数 |
| 输入 | 文本、单图片、单音频、图片与音频混合；顺序与 batch 请求 |
| 执行 | Eager、`FULL_DECODE_ONLY`；不扩大到其他 graph 模式 |
| 并行 | A3 TP1、TP2；未验证其他硬件或 TP 规模 |
| Chunked prefill / prefix cache | 使用默认启用路径；共享缓存另有独立数值测试 |
| 容量配置 | TP2 启动 `max-model-len=131072`、`max-num-seqs=16`，完成短请求并发 |
| 长输入 | 上述容量配置不代表实际 128K 输入或 16 个满长请求已验证 |
| EP / flashcomm1 | 不适用：这两个 checkpoint 没有 MoE 专家 |
| MTP | checkpoint 未提供 nextn/MTP 权重；不宣称 MTP 验证通过 |

## 环境准备

按[安装指南](../../getting_started/installation.md)选择对应 CANN、TorchNPU
和 vLLM Ascend。源码安装必须使用仓库 `.github/vllm-main-verified.commit`
指定的 vLLM revision，不能独立更新 vLLM 或升级 Transformers。

=== "Atlas A3"

    使用安装指南中的 A3 容器启动命令，并将完整权重目录挂载为 `/models`。
    `ASCEND_RT_VISIBLE_DEVICES` 中填写运行时逻辑设备 ID；A3 双 die
    环境中的 TP2 表示两个运行时设备，不等同于两张物理卡。

=== "Atlas A2"

    使用安装指南中的 A2 镜像和驱动挂载配置。本文没有 A2 实测证据，
    不能直接沿用 A3 的性能或容量结论。

源码安装不能跳过原生算子编译。`aclnnAddRmsNormBias` 随项目源码提供，
不能仅凭 CANN 中存在同名接口判断可用。进入匹配环境后执行：

```shell
cd /vllm-workspace/vllm-ascend
MAX_JOBS=32 COMPILE_CUSTOM_KERNELS=1 python setup.py build_ext --inplace
COMPILE_CUSTOM_KERNELS=0 python -m pip install -e . --no-deps --no-build-isolation
```

保留构建日志、源码提交、原生库 SHA256 以及 `pip freeze`。
下载权重后校验全部 safetensors 分片、index、processor、tokenizer 文件；
不能只下载 config 作为模型验证。

## 部署

以下为 TP2 graph 路径，两种模型只需替换 `MODEL_PATH`。运行前确认所选
设备空闲、端口 8000 未被其他服务占用，并配置本机可用的 HCCL/Gloo 网卡。
共享机器不得停止其他任务来释放设备。

```shell
export ASCEND_RT_VISIBLE_DEVICES=0,1
export HCCL_OP_EXPANSION_MODE=AIV
export MODEL_PATH=/models/gemma-4-E2B-it
cd /workspace
vllm serve "$MODEL_PATH" \
  --host 127.0.0.1 --port 8000 --served-model-name gemma4 \
  --tensor-parallel-size 2 --distributed-executor-backend mp \
  --max-model-len 131072 --max-num-seqs 16 \
  --max-num-batched-tokens 2048 --kv-cache-memory-bytes 8589934592 \
  --limit-mm-per-prompt '{"image":1,"audio":1}' \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8,16]}'
```

需要隔离 graph 问题时，以 `--enforce-eager` 替换 compilation config，
其他参数保持不变。初次精度回归可用 TP1、4096 长度、2 GiB KV cache；
不要把精度配置和长上下文容量配置混用。

## 功能验证

必须先检查模型身份，再发送真实生成请求。仅启动成功不算通过。

```shell
curl --fail http://127.0.0.1:8000/v1/models
curl --fail http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"gemma4","messages":[{"role":"user","content":"What is 2 + 2? Answer with just one digit."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

图片、音频分别使用 `image_url` 和 `audio_url` 内容项。可以传入
`data:image/jpeg;base64,...` 或 `data:audio/ogg;base64,...`，混合请求将两项
和文本放入同一个 `content` 数组。仓库真实权重测试使用固定公共资产
`ImageAsset("stop_sign")` 与 `AudioAsset("mary_had_lamb")`，并实际覆盖四种输入：

```shell
python -m pytest -s tests/e2e/models/test_gemma4_mm_correctness.py \
  --config tests/e2e/models/configs/gemma-4-E2B-it.yaml
python -m pytest -s tests/e2e/models/test_gemma4_mm_correctness.py \
  --config tests/e2e/models/configs/gemma-4-E4B-it.yaml
```

本地 checkpoint 可制作只替换 `model_name` 的 YAML 副本，不修改其余测试参数。
测试对比同一 checkpoint 的 Transformers 参考：文本和视觉使用 eager，
音频使用原生布尔可见性 mask 构造。不能给手写音频 attention 传入全局
eager 路径生成的加性浮点 mask。首个分歧 token 使用匹配 vLLM 的双方
top-5 包含规则，不意味着音频输出逐 token 一致或 ASR WER 已达标。

## 精度回归

以下为 GSM8K 前 100 个测试样本的固定回归基线，不是完整数据集成绩。
环境为 A3、BF16、TP1、Eager，lm-eval 0.4.12，5-shot、chat template、
fewshot-as-multiturn、batch 8、关闭 thinking，默认最多生成 256 token。

| 模型 | strict-match | flexible-extract |
| ---- | ------------ | ---------------- |
| E2B | 0.54 | 0.66 |
| E4B | 0.73 | 0.80 |

```shell
python -m pytest -s tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/gemma-4-E2B-it.yaml
python -m pytest -s tests/e2e/models/test_lm_eval_correctness.py \
  --config tests/e2e/models/configs/gemma-4-E4B-it.yaml
```

文本分数不能代替图片或音频精度验收；多模态参考测试与以上回归同时保留。

## 性能与排查

按 [vLLM 性能评估](https://docs.vllm.ai/en/latest/benchmarking/)记录 TTFT、
端到端延迟和输出 token 吞吐，明确模型、TP、输入/输出长度、并发、缓存命中
与 warmup。固定输入重复请求会命中 prefix/多模态缓存，不能当作冷请求吞吐。
连续多轮采集设备 HBM 和进程显存，不能仅凭一次峰值判断无内存增长。

共享注意力回归包含 256/512 维、非连续物理块、不同序列长度、有无缓存前缀，
以独立 SDPA 参考比较。修改该公共后端后还应对已有多模态模型使用同配置
baseline/candidate/candidate/baseline 对照，不能拿两个不同模型互相比性能。

显存启动检查失败时先检查其他进程与当前可用 HBM，不应直接归因模型不支持。
先保留错误日志，再选择空闲设备重跑；不要为通过测试静默降低精度判据。
