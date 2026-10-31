# Qwen3.8-Next PLE CPU offload

本功能用于Qwen4Exp架构的BF16 ngram表，权重常驻CANN注册的CPU内存，NPU在第二层按索引读取。投影和卷积状态继续驻留NPU。默认关闭，不改变checkpoint格式。

## 启用

在原有MRV2启动命令中增加：

```bash
--engram-config '{"cpu_offload":true,"dp_shared_memory":false}'
```

关闭并建立对照：

```bash
--engram-config '{"cpu_offload":false}'
```

当前限定DP=PP=PCP=DCP=1，不支持共享表、THP、elastic EP和ubatching。需要支持host注册映射的CANN运行时及充足CPU内存。TP4每rank表约23.84 GiB，具体以启动日志为准。

## 实现与释放

保留上游PLE词表行分片、权重加载、ngram上下文与SP通信语义。160维kernel使用256宽masked向量。CPU表在构造失败时清理；正常MRV2 shutdown先同步并释放图，再调用host owner关闭。注销失败保留所有者与参数供重试；注销成功但释放失败时不会重复注销；runner保留待释放owner直到成功。成功后移除参数别名。进程被强制终止时由操作系统回收资源，不能等同于测试过优雅退出。

## 验证

```bash
pytest -q tests/ut/models/test_ple_offload.py
pytest -q tests/e2e/pull_request/one_card/test_ple_host_lookup.py
```

NPU测试覆盖160维、无效索引、分片及分段边界、多次修改输入的FULL graph replay。模型级MTP/SP、真实显存与性能结果以独立验证报告为准，不能只凭小表测试宣称通过。

## 性能比较

显存收益应比较model loading和KV可用预算；服务通常会把释放空间重新分给KV，因此npu-smi总占用不一定下降。固定相同实际batch比较延迟，再扩大并发比较容量收益。随机host读取可能增加延迟，不能按连续H2D带宽预测TPOT。

## 实测结果（2026-10-29）

Ascend 950、TP4/EP、真实Qwen3.8-Flash-Next-w8a8-mxfp8权重、MRV2、MTP3、FULL_DECODE_ONLY：

| 每卡预算 | 驻留表 | CPU offload |
| --- | ---: | ---: |
| 模型加载NPU占用 | 58.07 GiB | 34.22 GiB |
| 可用KV预算 | 20.36 GiB | 44.20 GiB |
| CPU表 | 0 | 23.842 GiB |

4K输入/256输出/并发16两轮端到端吞吐约下降1.8%；128K输入/128输出/并发16单轮下降0.45%。三条自然语言请求和66条合成请求的输出与驻留表逐条一致。这是功能与性能回归，不是正式数据集精度评分。测试包含prefill和排队，不能当作PD纯D稳态结果。

这些A/B数据关闭FlashComm1。FlashComm1在驻留基线也出现HC形状错误，当前不支持以该组合交付。DP/PP/PCP/DCP扩展与CPU共享表需另行验证。

## 原生HC SP

在上述配置上追加 `--enable-hc-sp`。已验证TP4/EP、MTP3、FULL_DECODE_ONLY组合的文本、图片及并发8请求；目标与MTP图均捕获成功。PLE和MTP的分片使用动态补齐算子，Ascend MoE的已归约输出由HC直接分片，避免重复求和。该验证不包含FlashComm1或MoE SP。

SP自然语言输出语义正确，但可能因归约顺序而与非SP措辞不同；本次未进行SP驻留/offload的匹配性能A/B。四卡查表通信可通过以下命令回归：

```bash
torchrun --nnodes=1 --nproc-per-node=4 --master-addr=127.0.0.1 --master-port=29628 tests/e2e/ple_host_sp.py
```
