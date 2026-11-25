---
name: gemma4-mm-validation
description: "在 Ascend 上复现 Gemma4 E2B/E4B 共享 KV 与真实多模态回归。"
disable-model-invocation: true
---

# Gemma4 E2B/E4B 验证

适用于任务 #10663。通用适配流程见相邻 `vllm-ascend-model-adapter/SKILL.md`；
部署与测试命令见 `docs/source/tutorials/models/Gemma4-E2B-E4B.md`。

1. 查询官方 main 并记录 SHA，使用 `.github/vllm-main-verified.commit`
   指定的 vLLM pin。保留现有用户改动；独立工作树迭代，不升级 Transformers。
2. 用 `npu-smi info` 核对物理卡、die、空闲 HBM、占用进程。共享环境只使用
   空闲设备；不要终止其他任务。记录设备、驱动、CANN 和 Python 包版本。
3. 下载两种完整 BF16 checkpoint，校验全部分片 SHA256、index 与处理器文件。
   禁止用 dummy、config 存在或裁剪层数代替真实权重验收。
4. 先编译项目原生算子再安装，保留 `aclnnAddRmsNormBias` opapi、tiling、
   A3 kernel 产物与日志。确认运行时导入当前源码与构建的扩展。
5. 运行 attention UT 和 `test_shared_kv_attention.py`。共享 reader 在首次
   prefill 中必须使用 producer 分页 KV；不能修改共享 metadata 来伪造状态。
6. 对 E2B/E4B 分别执行 `test_gemma4_mm_correctness.py --config ...`，覆盖
   eager/graph、文本/图片/音频/混合、单请求/batch。参考实现保留音频原生
   布尔 mask；首分歧 mutual top-5 对齐不得描述成逐 token 相同或完整 ASR 精度。
7. 执行两份 YAML 的 GSM8K 回归；100 样本仅作回归基线，不能描述为完整榜单。
   不得因为结果不通过就降低阈值，先检查模板、generation 参数和真实代码路径。
8. 从 `/workspace` 直接运行真实 `vllm serve`，默认端口 8000；核对模型身份、
   真实 HTTP 生成和 graph capture/replay。测 TP1/TP2，容量配置 128K+bs16
   与实际输入长度分开报告。EP/flashcomm1 为非 MoE 不适用；MTP 缺 checkpoint。
9. 对已有多模态模型作同配置 ABBA 回归，记录每轮输出、TTFT、E2E、吞吐、
   HBM 与缓存状态。共享机器波动、不同输出长度不能包装成确定的性能提升。
10. 更新教程、配置与证据索引，以一个带 Signed-off-by 的提交交付；最终源码
    重新构建并实机重测。公共 PR、CI 和维护者验收与本地通过分别标注。

复盘材料至少包含源码 SHA、vLLM pin、命令、退出码、原始输出、权重校验、
构建日志和适用边界。任何失败保留原始记录，不以初始化成功覆盖失败。

## 可复用提示词

- “基于当前官方 main 与匹配 vLLM pin，追踪 Gemma4 共享 KV 生产层和读取层的首次 prefill 数据路径。先核实完整权重和实际导入源码，不把初始化成功当成验收。”
- “对 256/512 维注意力、不同序列长度、非连续物理块和有无缓存前缀建立独立 SDPA 参考。在真实 NPU 上验证，不修改共享 metadata 来绕过失败。”
- “音频对齐失败时，逐项核对处理器、输入特征和 attention mask 语义。区分参考脚本问题与 NPU 实现缺陷，不升级 Transformers，也不放宽精度规则。”
- “使用相同模型、硬件、配置和定长输出做 ABBA 性能对照，保留所有慢轮及 HBM 采样。编译项目算子后再测试，公开 CI、维护者验收与本地结果分别报告。”
