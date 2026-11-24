# MiniMax-M3.1-Flash BF16 LoRA 验证

本目录连接 MindSpeed-Agent_9400 的 LoRA P0–P9 runner，提供 M3 的配置生成和版本兼容采集器。设备验证需要真实本地 BF16 模型及 NPU；没有 adapter 时由参考生成器生成确定性 A/B/zero。生成的 adapter 只用于正确性诊断。

## 版本与范围

- 起点：个人 `m31-lora` 分支 `6657d587b218e67e809c5123f28de0e0f22f3817`。
- vLLM：仓库配套 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`。
- 参考：MindSpeed-Agent_9400 `d6b369ed8298326ec01119f73a5adbd948651448`，LoRA skill v1.0.2；其 LoRA engine、工作流、生成器和测试与 `e1c0bc9` 快照逐字节一致。
- 本轮资产：真实 BF16 前4层，hidden6144、Q头64、KV头4、head128，保留128专家/top4；3个dense层和1个MoE层。Q/K/V/O共16个逻辑目标、32个A/B张量。未训练随机因子不代表模型质量验证。
- 本轮执行：A3、单卡、TP/DP/PP均1、eager、单请求、一个驻留adapter；关闭prefix cache和chunked prefill。
- 冻结探针：rank8、alpha16、seed20261124、amplitude1.5；磁盘FP32因子，运行BF16。prompt为`[1,3,4]`，生成3个token，覆盖一次prefill和两步decode。amplitude0.5的信号不足来自迁移前诊断；本次新资产独立冻结1.5并重新校验全部实际信号，旧记录不计本次通过数。
- 冻结误差：`atol=rtol=1/64`，逐元素要求`|actual-reference| <= atol + rtol * |reference|`；结构和索引精确匹配，非零信号须超过误差裕量10倍。不得因结果失败修改容差或减小权重。

实测依赖为Python3.12.13、torch2.10.0、torch-npu2.10.0.post7、Transformers5.19.0、safetensors0.8.0。镜像内安装元数据的vLLM/Ascend版本号与检出源码不同；配套身份以以上commit和[实际源码哈希](validation-summary.json)为准。

## 生产改动

M3的稀疏注意力层把Q/K/V和两个indexer投影融合为五段物理输出：`[8192,512,512,512,128]`。普通LoRA选择器不能包装此专用类型；直接把五段都当作QKV的三段也会错位。

`vllm_ascend/lora/utils.py`的`MiniMaxM3QKVParallelLinearWithLoRA`仅匹配该精确类型，继承已有Ascend packed QKV包装器，复用加载、分片、打包和reset逻辑，只分配三个adapter片段。执行时取前三段的连续副本，应用标准低秩分支，再写回前三段；两个indexer片段保持原值。O投影继续使用现有row-parallel包装器。没有引入merged缓存模式。

注册保持原有六个Ascend包装器的优先级；没有修改已有注册顺序测试。

## 采集器与独立参考

`m31_lora_driver.py`使用具名worker RPC和typed engine LoRA API，适配此版本的M3模块名称及HF多模态入口。它不启用任意callable序列化，不替换生产推理结果。

模型级参考为独立HF CPU BF16，使用相同token历史进行teacher forcing，比较采样位置logprob并检查greedy token。M3原始checkpoint的MoE路由权重和修正bias为FP32；参考使用Transformers的`_keep_in_fp32_modules_strict`按精确张量名保留这两项，避免全模型BF16加载改变专家选择，dense MLP和QKV仍为BF16。LoRA通过HF原生linear的独立低秩forward hook实现，不使用vLLM算子或已加载buffer构造参考。原FP32预检结果保留为差异记录。

CPU参考按本轮A3路径保留融合运算的存储边界：batch-invariant残差先BF16相加，norm用FP32中间值乘已BF16舍入的`1+weight`；clipped SwiGLU与RoPE使用FP32中间值、BF16输出。注意力采用标准HF SDPA，并分别执行一次prefill和两步带cache的teacher forcing。独立NPU小算例验证这些边界；不把原生HF的逐点BF16中间舍入当作融合算子的存储边界。RoPE仅在同步CPU调用期间替换，退出或异常后恢复。本参考要求`VLLM_BATCH_INVARIANT=1`，不适用于其他路径。

层级参考从原始磁盘A/B和实际X计算CPU FP64乘法，显式加入BF16存储边界：因子按BF16加载、alpha/r折入B、shrink结果舍入BF16、expand结果舍入BF16。完整输出用独立的FP64 XW舍入到BF16再加参考delta。另以FP64 XW核对真实基座，加载buffer单独核对。

原版observer以`Y_BF16-XW_FP64`充当增量。无adapter对照中，基座舍入也产生最大0.321的非零差值。`m31_observation.py`因此补采真实`y.add_(delta)`操作的base和delta：

1. 分支delta与磁盘因子计算的独立BF16执行参考比较，容差不变。
2. 完整层输出与独立的BF16基座加分支结果比较，并保留实际base与FP64基座对照。
3. 精确核对最终输出等于捕获的BF16加法，indexer尾部不变；融合未活动片段检查实际分支。
4. 保存原始X/Y/W、加载buffer及新增观测，可复算原版差值。缺少观测或进入其他执行路径直接报错。

该补采只用于诊断单adapter eager matmul路径，不用于性能测量，不外推到图模式、自定义多adapter kernel或A5。

理想FP64数学与BF16实际执行分开报告：每条层记录保留`ideal_fp64_difference`，包含原始FP32磁盘因子在FP64数学下的delta/完整输出误差及超限元素数。它不被删除或改成通过。用户已明确选择按独立BF16舍入参考判断实现正确性；门限仍为1/64。参考变化须建立独立run，或登记repair使源码关联用例失效并全部重测，保留旧失败记录。

## 重跑

先检出上述配套版本，并准备参考仓库和真实本地模型；运行前确认指定设备空闲。以下路径变量由实际环境填写，不依赖本轮私有目录。

```bash
cd /path/to/vllm-ascend
export MODEL=/path/to/real-bf16-model
export VLLM_SRC=/path/to/vllm
export ENGINE_ROOT=/path/to/MindSpeed-Agent_9400/plugins/inference-model-optimize
export CASE_ROOT=/path/to/new-validation
export ASCEND_RT_VISIBLE_DEVICES=2
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export VLLM_VERSION=0.33.0.dev0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_BATCH_INVARIANT=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

python tests/e2e/models/minimax_m31_lora/prepare.py \
  --model "$MODEL" --vllm "$VLLM_SRC" --engine-root "$ENGINE_ROOT" \
  --directory "$CASE_ROOT/config"
python "$ENGINE_ROOT/engine/lora/runner.py" --workspace "$CASE_ROOT/run" \
  prepare --config "$CASE_ROOT/config/config.json"
python "$ENGINE_ROOT/engine/lora/runner.py" --workspace "$CASE_ROOT/run" \
  run --target device --max-cases 100
python "$ENGINE_ROOT/engine/lora/runner.py" --workspace "$CASE_ROOT/run" finalize
```

生成器自动读取实际层数和维度，不能把减层目录称为完整60层验收。`prepare`固定全部模型分片、索引、关键源码、采集器和运行依赖；源码变化会失效关联阶段，须登记修复并重测，不能复用旧绿灯。

```bash
VLLM_VERSION=0.33.0.dev0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q tests/ut/lora
bash format.sh ci
```

## 验证清单

| 阶段 | 验证内容 |
| --- | --- |
| P0/P2 | 环境身份、16目标及32张量映射 |
| P3 | 原始A/B/zero、低秩链与物化矩阵的独立数学对照 |
| P1 | 无adapter基座与独立HF参考 |
| P4 | 逐目标真实加载buffer、scale、rank尾部和切片布局 |
| P5 | 逐目标prefill/decode增量、完整输出、融合切片隔离和信号裕量 |
| P6 | base/zero/A模型级及所有目标的层级数值 |
| P7 | 先按原合同执行A→base→A，再从该A继续base→B→A；分别保存原始load/remove/load与附加切换操作，逐次比较HF |
| P8 | 超限rank实际拒绝、恢复base/zero，以及缺delta/错scale/错slice的测量负例 |
| P9 | 审计当前代证据、完整性与失败修复历史 |

混批、多adapter同时驻留、TP/DP/EP多卡、ACLGraph、flashcomm1、MTP、多模态输入、A5和完整60层均不在本轮单请求合同内，未验证。词表、embedding、LM head和MoE专家本身也未作为LoRA目标。实际max-model-len为64；未验证理论上下文长度和128k/bs16容量。

本轮P0–P9为VERIFIED，当前代68项用例及额外普通推理对照通过。范围和限制见[验证报告](VALIDATION.md)，逐项结果与失败修复历史见[标量证据](validation-summary.json)；不能把该结论外推为完整60层、多卡或混批验收。
