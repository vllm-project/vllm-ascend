# m31-lora 独立验证与修复计划

**目标：** 从个人分支 `m31-lora` 的 `6657d587b218e67e809c5123f28de0e0f22f3817` 重新验证 MiniMax-M3.1-Flash，不带入旧适配提交。参考 MindSpeed-Agent_9400 最新 `d6b369ed8298326ec01119f73a5adbd948651448` 的 LoRA 单请求流程；该提交的执行引擎、生成器与P0–P9合同和已固定的 `e1c0bc9` 快照逐字节一致，仅新增统一Agent入口说明。

**约束：** vLLM 固定配套 `c410543d9ffff03f5f6c4a8c0ae1d3ad082c4a15`，不升级 Transformers。真实 BF16 前4层、原宽度和专家数；Q/K/V/O共16个逻辑目标。单设备、eager、单请求、单驻留 adapter，prefill和两步decode。合成权重只证明对应范围，不代表60层和训练效果。

## 冻结合同

- 使用参考流程生成归一化 FP32 A/B/zero，rank8、alpha16、seed20261124；真实加载为BF16。最初amplitude0.5；基座真实输入的信号诊断不足后，以amplitude1.5新建独立run，保留原run记录。
- 本次设备误差预算在首轮实测前冻结为 `atol=rtol=1/64`，即两倍BF16 epsilon；索引、布局、生命周期、单请求数量等结构条件精确匹配。信号仍须超过误差预算十倍裕量，不按结果放宽。
- 原始磁盘因子和实际层输入构造CPU FP64理想delta并保留差异报告；BF16执行参考使用独立FP64乘法和显式BF16存储边界。模型参考为独立HF CPU BF16同历史teacher forcing，原FP32预检记录保留。
- 覆盖base、zero、A、B以及A→base→B→A串行切换、同adapter load/remove/load和非法rank拒绝后恢复。混批一致性不属于本轮验收，明确记录未测试。

## 执行步骤

1. 保留基线启动与加载失败。记录源文件、配套commit、设备和权重身份；生成原始模块/磁盘A/B/运行wrapper映射。
2. 将参考runner原样固定到测试目录，使用它的P0–P9合同、生成器、磁盘FP64参考和修复事件。M3专用采集器适配HF入口、具名worker RPC、五段QKV/indexer尾部。
3. 发现问题先报告具体文件、触发和影响；依照用户本轮明确授权修复。核对安全、资源、边界、时序，并尝试反驳缺陷避免误报。
4. 只改必要的生产路径，补充对应回归；登记修复与依赖失效，重跑原失败及关联阶段，保留失败历史。
5. 相关单测、真实NPU阶段和参考runner审计全部完成后，打包不含权重、凭证与内网信息的证据；仅推送验证通过的修改到个人分支。

## 初始限制

当前已提供的设备为A3。没有A5设备实测则不宣称A5通过。当前使用真实减层资产，不以dummy或随机小模型替代M3实机结论。未经验证的结果保留PARTIAL，不承诺或伪造VERIFIED。

## 基座对照揭示的采集修正

原版observer将`Y_BF16 - XW_FP64`全部当作LoRA增量。无adapter基座的O投影也出现最大0.321的差值，QKV的未活动列亦包含基座舍入误差，因此不能用这个差值判定LoRA分支。

补充采集真实eager路径`y.add_(delta)`的输入和delta，不重算、不替换模型输出。独立BF16参考只使用原始磁盘因子、实际输入和基座W；精确核对实际BF16加法及indexer尾部。原始记录同时保留，可复算原版observer的差异。该修正仅适用本轮单adapter eager matmul路径，其他路径缺观测直接报错。

## 探针信号诊断与新run

使用已保存的真实基座X，amplitude0.5的Q/K/V在部分decode步低于十倍误差裕量，最小peak为0.036479；这是探针无效的诊断，不是模型缺陷，不算通过。另建`run-004`固定amplitude1.5，A/B各增大3倍，使理论delta增大9倍；容差与十倍信号标准均不变。适配代码同时直接继承已有Ascend packed QKV包装器，复用打包、加载和清理，避免另造执行分支。设备P5仍须根据本次实际输入逐步重新检验信号。

## 用户确认的BF16执行参考

正常CPU BF16运算在强探针上也触发原始FP64门限，具体反例和归因见`strong-cpu-counterexamples.json`与`bf16-rounding-attribution.json`。用户选择“增加独立BF16舍入参考来判断实现正确性，保留理想FP64差异报告，容差不变”。据此新建run固定参考变化：独立FP64计算明确模拟BF16因子、shrink和expand的舍入；模型用CPU HF BF16独立低秩分支；逐层理想FP64误差另存报告。不得复用旧run的最终验收，也不改成宽松门限。

## 本轮参考精度与回执修正

新设备的真实checkpoint保留MoE路由权重和修正bias为FP32；HF全模型BF16加载改变了这两项和专家选择。仅恢复原始FP32路由即可使非零A的token与logprob对齐。使用当前Transformers已有的`_keep_in_fp32_modules_strict`按精确张量名保留，实测路由权重与磁盘完全一致，QKV及dense MLP仍为BF16。参考精度修正以独立run固定，未改变模型权重或误差预算。

原P7验收器严格要求三个操作：load/remove/load。采集器先执行A→base→A，保留原始操作回执，再从该A继续base→B→A；附加操作与全部HF比较独立记录并参与检查。参考状态机不变，原始生命周期及要求的A→base→B→A均被真实执行。正式`run-007`冻结上述源码，先前预检仅作诊断，不计入最终验收。

## P7失败后的参考边界修复

正式P7保留一次失败：B的15个logprob有4项超限，第二步greedy与原生CPU HF不同。实际48次B增量与独立BF16层级参考逐元素一致，新的无观测NPU engine先加载B，也与切换后的B输出完全一致。不能据此删除整模独立参考检查。

第一性原理：BF16张量相同不代表融合算子的中间值也以BF16存储。可检验的约束是本轮实际执行路径和已冻结的dtype、权重及容差；“原生HF逐点计算与NPU融合执行有相同舍入”是未经验证的假设。实测开启batch-invariant时，AddRMSNorm先BF16相加，再以FP32归一化并乘BF16的`1+weight`；73,728个输出与独立CPU公式精确一致。Triton RoPE使用FP32中间值，104,448个输出精确一致；clipped SwiGLU也使用FP32中间值，196,608个输出均满足原门限。未经batch-invariant替换的融合AddRMSNorm小算例不能作为本轮路径依据。

最小修复仅作用于测试进程内的CPU HF参考：使用标准SDPA和相同prefill/decode历史，按实际融合边界计算残差norm、clipped SwiGLU、RoPE，继续以独立磁盘权重和因子计算。RoPE替换在每次同步CPU调用后恢复，异常也恢复；不改变NPU实现、权重、容差、目标或判定门。四组预检的token全部匹配，原门限下均无logprob超限。先保留这些独立算例，再登记reference repair，重新执行全部被源码指纹变化失效的用例，保留原P7失败及修复记录。

## 执行结果

2026-11-24完成：当前代68项用例全部PASS，P0–P9为VERIFIED，原P7问题经修复重测关闭。额外无观测hook的A→base→B→A普通推理对照中，token及logprob与正式P7记录完全一致；LoRA单测94 passed，本地`format.sh ci`通过。历史失败及6次权重加载前的显存阻塞均保留，未计入本次通过数。

当前代设备用例均使用同一张A3卡，单请求、单驻留adapter、BF16/eager及真实前4层；A5、完整60层、TP/DP/EP多卡与混批仍未验证。具体结果和证据见[验证报告](../../tests/e2e/models/minimax_m31_lora/VALIDATION.md)。
