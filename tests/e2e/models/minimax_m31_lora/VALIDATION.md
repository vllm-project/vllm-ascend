# 本轮验证报告

日期：2026-11-24。版本、冻结参数、参考方法和重跑命令见[README](README.md)。从个人 `m31-lora` 分支的 `6657d587` 重新开发，未沿用旧适配分支的通过结论。

结论：本轮声明范围为 **VERIFIED**。当前代68项用例全部通过，其中32项CPU映射/探针对照、36项真实NPU用例；P0环境检查和P9最终审计也通过。逐用例指标、源码哈希、原失败及修复记录见[标量证据](validation-summary.json)。

## 验证范围

真实60层checkpoint直接保留decoder 0–3，原始张量值和dtype不变，保留全部宽度、128专家/top4、词表及视觉权重，关闭本轮未执行的MTP。减层资产共960个张量、23,485,394,944字节，含957个BF16和3个FP32张量；源配置、源索引、生成脚本和输出分片均留存哈希。

实机执行为A3单卡、BF16、eager、单请求、一个驻留adapter，TP/DP/PP均1；关闭prefix cache和chunked prefill。16个Q/K/V/O逻辑目标、32个A/B张量；rank8、alpha16、seed20261124、amplitude1.5，prompt `[1,3,4]`，每次生成3个token。

当前代全部有效设备用例在同一张A3卡上完成。历史记录还使用过另一张同型号卡；合计6次调用在权重加载前因显存不足被阻塞，原始日志保留，未作为数值失败或通过数计入。等待空闲并核对实际可用显存后，补跑全部缺项，未降低显存配置或修改验收参数。

容差始终为 `atol=rtol=1/64`；布局和索引精确匹配，每个非零探针调用须满足十倍信号裕量。层级采用用户确认的独立BF16舍入参考，另外保留理想FP64差异。

## 已定位和修复

| 问题 | 根因与处理 | 验证依据 |
| --- | --- | --- |
| 启用LoRA后M3专用QKV无法初始化 | 专用类型融合Q/K/V与两个indexer投影。新增精确类型包装器，复用已有packed QKV加载、分片和reset，只更新前三段 | 原始失败日志、五段输出UT、实机加载和层级比较 |
| 初次注册改变已有包装器优先级 | 将新类型放在原有六个Ascend包装器之后 | 原顺序UT保持不变；完整LoRA UT通过 |
| 非M3选择路径提前导入M3依赖 | 先检查五段布局，再延迟导入精确类型 | 非M3导入隔离回归通过 |
| 基座BF16舍入混入LoRA增量 | 原observer用最终BF16输出减FP64基座。补采实际加法前delta；独立检查delta、完整输出、基座及精确BF16加法 | 无adapter反例、跨custom-op采集UT、独立BF16参考UT；容差未改 |
| 正常BF16计算与理想FP64有超限差异 | 用户确认使用独立BF16存储边界参考；保留原始FP64误差及超限数，不删除失败历史 | 独立FP64乘法显式模拟因子、shrink和expand舍入；新run固定参考 |
| HF全模型BF16加载改变MoE路由 | 原checkpoint路由权重和修正bias为FP32；HF默认加载丢失原精度。使用标准严格FP32保留接口及精确张量名 | 原始路由权重逐元素一致，QKV和dense MLP仍BF16；仅恢复这两项即使非零A的token及logprob对齐 |
| P7串行B切换与原回执合同冲突 | 原状态机严格要求load/remove/load。先完成A→base→A，再继续base→B→A；原操作与附加操作分别记录，逐次比较HF | 验收器未修改，原合同与附加切换均需真实执行 |
| 原生CPU HF的BF16逐点舍入与A3融合执行不同 | P7中B的4/15个logprob超限。独立验证实际batch-invariant norm、clipped SwiGLU和RoPE的存储边界；CPU参考按这些边界计算，使用标准SDPA和相同prefill/decode历史 | norm 73,728值和RoPE 104,448值精确一致，SwiGLU 196,608值满足原门限；保留失败并登记reference repair，全部源码关联用例重新执行 |

迁移前amplitude0.5在部分真实输入上信号不足。保留原记录；本次新资产以独立run固定amplitude1.5，并重新验证全部实际信号。相同输入与随机方向下，理论delta增强9倍；没有降低信号标准、放宽容差或减小权重。不同资产的记录不合并为本次通过数。

## 验证结果

- 完整LoRA单测：94 passed，17 warnings，1.74秒。涵盖M3选择、2D/3D五段隔离、部分adapter重载、reset、非M3导入隔离、真实custom-op边界观测和独立BF16参考。
- 本地Linux执行`bash format.sh ci`全部通过；此处是本地格式和静态检查结果，不是GitHub CI结果。
- 参考engine既有CPU回归在本轮早期执行：79 passed，31 subtests passed。参考生成器、状态机及判定门未修改；该结果独立列示，不计入本次NPU验收通过数。
- 正式整模及切换对照：base/zero/A/B的greedy token均匹配独立CPU BF16参考；每组15个logprob满足原容差。最大绝对差为base/zero 0.03126001、A 0.06146288、B 0.06556129。
- 正式P6层级6,168,960个比较值、加载buffer 18,677,760个比较值均满足原容差。理想FP64误差和超限数另存，未将其改成通过。
- 原P7失败记录保留。登记reference repair后，原67项尝试因源码指纹变化全部失效；重新执行当前代全部68项，P7成功重测关闭`issue-1`。历史135项尝试中的134 PASS和1 FAIL不混算为本次通过数。
- 额外普通推理对照未安装观测hook：A→base→B→A四次请求的token和logprob与当前代正式P7记录完全一致。

| 阶段 | 当前代结果 | 内容 |
| --- | --- | --- |
| P0/P9 | PASS / PASS | 环境身份、最终证据完整性与修复历史审计 |
| P2/P3 | 16/16 + 16/16 PASS | 映射、非零A/B及zero、独立低秩数学对照 |
| P1 | 1/1 PASS | 无adapter基座与独立模型 |
| P4 | 16/16 PASS | 实际buffer、缩放、rank尾部和物理slice |
| P5 | 16/16 PASS | prefill及两步decode、delta、完整输出、切片隔离和信号裕量 |
| P6 | 1/1 PASS | base/zero/A整模及全部目标层级数值 |
| P7 | 1/1 PASS | A→base→A，再执行base→B→A，全部独立模型对照及加载操作通过 |
| P8 | 1/1 PASS | rank1032被max-rank8实际拒绝，base/zero恢复；漏delta、错误scale和错slice均被检测 |

| Adapter | 正式NPU及独立CPU参考的greedy token ID |
| --- | --- |
| base / zero | `[3051,3051,3051]` |
| A | `[164528,42904,6670]` |
| B | `[4232,488,58720]` |

## 对抗式检查

| 视角 | 依据与结论 |
| --- | --- |
| 安全 | `m31_lora_driver.py:43`只映射具名诊断RPC和typed LoRA API，未开启任意callable序列化；公开资料不含权重、层级X/Y/W快照、账密及内网地址 |
| 性能 | `vllm_ascend/lora/utils.py:164`仅复制目标QKV前三段以满足连续行要求；无全参数扫描或设备标量读取。诊断快照仅用于测试，本轮无性能结论 |
| 边界 | `vllm_ascend/lora/utils.py:150`精确匹配类型、三目标和五段布局，并拒绝fully-sharded。`tests/ut/lora/test_minimax_m3.py:71`覆盖形状、非活动片段、部分重载和reset；同文件第52行覆盖非M3导入边界 |
| 时序 | `m31_lora_driver.py:103`限制单请求和单驻留adapter，第32行按请求清空采集状态，第300行退出时关闭本用例engine。第99行CPU RoPE替换在同步调用后恢复，异常同样恢复。正式P7/P8及普通推理对照均通过 |

反驳检查：初始化错误发生在adapter加载前，故不能归因于合成权重；无adapter基座同样触发旧增量门限，证明旧观测混入基座舍入；仅恢复原始FP32路由即可消除非零A的旧分歧，且精确名称方案未提高dense MLP精度。B另由独立小算例证实的融合存储边界修复，整模独立参考检查保留。

## 未验证范围

A5、完整60层、真实训练adapter、混批、多adapter同时驻留、TP/DP/EP多卡、fully-sharded、ACLGraph、flashcomm1、MTP、多模态输入、长上下文和吞吐均未验证。专家、embedding和LM head本身未作为LoRA目标。当前结果不能替代原始全层、多卡或混批验收。
