# GLM-5.3-Flash 裁层 CI：作用、实现与使用

## 作用和当前接入状态

这个 CI 在 A3 四卡上运行九层固定合成权重模型，用来发现 Flash 推理实现的功能失败、数值回归和性能退化。真实权重任务准确率、完整模型深度和 TP16 通信仍需 nightly。

当前提交提供 opt-in 四卡冒烟、完整边界采集器、离线门禁、MTP 探针，以及单图和 TP16 服务启动工具。尚未修改社区 workflow 自动启用硬门禁。未提供 checkpoint 参数时，四卡测试显示 SKIP；这不代表模型验证通过。

| 层次 | 实际检查 | 能发现的问题 |
| --- | --- | --- |
| 四卡冒烟 | 初始化、W8A8 MoE 实例、层结构、有限 logits、每 rank 图 replay | 配置或运行路径失败 |
| 完整精度 | 固定输入和续写、完整词表 logits、三次 graph 与一次 eager | 数值回归及执行路径不一致 |
| 性能门禁 | 四并发输入128输出64的吞吐、TTFT p95、TPOT p95 | 匹配环境下的性能退化 |
| MTP 探针 | draft/verify、混合请求复用、固定续写、拒绝恢复 | speculative 状态或输出回归 |

模块实例和量化方法检查证明模型构造采用对应实现，不等同于逐个算子的 profiler 证据。每个 TP worker 的 NPUGraph.replay 计数证明图实际回放。

## 裁层和权重构造

tools/ci/glm53flash_config.py 读取 W8A8 checkpoint 的 config.json 和 quant_model_description.json，生成裁层配置，不读取完整权重分片。

默认保留原始0–8层：0–2为KDA+Dense，第3和7层为稀疏注意力+MoE，第4–6和8层为KDA+MoE。隐藏维度、专家数、路由及mHC保持原配置。已知按层列表和注意力层索引同步裁剪，未知按层列表直接拒绝。支持5层配置，但本文完整基线和性能结果针对9层。

启用MTP时保留一个物理预测层，将原始末尾的量化层号映射到裁层末尾；预测3 token不代表保留3个物理MTP层。文本配置删除视觉部分，单图配置保留视觉编码器。

FlashDummyLoader 按参数名SHA-256派生种子，初始化小范围INT8权重、正scale及专门处理的KDA/mHC参数，保留结构buffer。每个rank在权重处理完成后计算参数名、dtype、shape和原始字节的SHA-256，作为实际执行权重的指纹。

## 精度门禁实现

1. 使用固定token-ID输入，关闭tokenizer，续写8次token 42以固定上下文。
2. hook compute_logits，在采样器allowed-token mask之前保存完整词表logits。比较的仍是全部154880个token，强制续写不会把比较数据压缩为一个token。
3. 单请求保存float32的[8,154880]数组，拒绝形状错误和NaN/Inf。
4. 分别独立启动graph-0、graph-1、graph-2、eager-0。
5. analyzer检查权重、配置、图回放和数据完整性；gate校验基线文件哈希，比较新采集与基线的最大绝对误差和top-1，并检查新采集的eager/graph差异。

边界包括3/4/5、127/128/129、511/512/513、2047/2048/2049、2051/2052/2053、2177，再加入实际block size前后边界。已校准配置的逻辑block size为1152，共19个长度。analyzer要求至少19个case；--smoke只有128长度，不能用作完整基线。

精度采集包含CPU拷贝和强制续写。性能采集恢复原始logits函数并正常生成。MTP探针中的batch-invariant是单独诊断配置，不能混入普通性能基线。

## 性能协议与阈值

TP4+EP、MTP关闭、prefix cache关闭、batch=4、输入128、输出64。每次graph启动预热3批、测量15批，三次共45批。吞吐为256/批次耗时；TTFT来自每请求首token指标；TPOT为首末token时间差除以63。抢占、指标损坏和输出长度不符均失败。这是离线引擎指标，HTTP服务需独立基线。

| 指标 | 历史A3 day20加修复源码实测 | 候选阈值 |
| --- | --- | --- |
| graph重复、eager/graph最大误差 | 0 | 0 |
| 45批吞吐中位数 | 179.582 token/s | ≥161.070 token/s |
| TTFT p95 | 1329.772 ms | ≤1529.237 ms |
| TPOT p95 | 22.166 ms | ≤25.491 ms |

吞吐取三个run中最低中位数的90%，延迟取p95的115%。这是工程余量，不是统计置信界。公共policy是未审批示例，基线哈希为空，不能直接启用。新机器、源码或镜像必须重新校准；历史A3结果不能作为当前PR HEAD通过NPU验证的证明。

## 运行方式

先在兼容vLLM/Ascend环境给进程分配四张空闲NPU，从仓库根目录执行。checkpoint由runner提供；不发布或下载完整权重。

```bash
python -m pytest -q tests/e2e/pull_request/four_card/test_glm5_3_flash.py \
  --glm53flash-checkpoint /models/GLM-5.3-Flash-W8A8

python -m tools.ci.glm53flash_config \
  --checkpoint /models/GLM-5.3-Flash-W8A8 --output /artifacts/flash9

# 在已确认正确版本采集，输出目录必须是新目录。
for run in 0 1 2; do
  python -m tools.ci.glm53flash_collect --model /artifacts/flash9 \
    --mode graph --output /artifacts/baseline/graph-$run
done
python -m tools.ci.glm53flash_collect --model /artifacts/flash9 \
  --mode eager --output /artifacts/baseline/eager-0
python -m tools.ci.glm53flash_analyze --root /artifacts/baseline \
  --output /artifacts/baseline-summary.json
python -m tools.ci.glm53flash_make_policy \
  --summary /artifacts/baseline-summary.json --output /artifacts/policy.json \
  --scope 'runner:A3:TP4:image-digest:source-commit:flash9'
```

待测版本用相同四次采集命令生成独立/artifacts/current。基线与待测目录不得相同或互相嵌套。runner核验机器、镜像、源码和卡独占状态后，生成与policy scope一致的environment JSON。示例中的false不能未经核验改为true；scope字符串本身不是自动硬件鉴别机制。

```bash
python -m tools.ci.run_glm53flash_gates \
  --baseline /artifacts/baseline --current /artifacts/current \
  --policy /artifacts/policy.json --environment /artifacts/environment.json \
  --output /artifacts/gate-result.json
```

默认回归报告WARN、退出码0；参数/数据无效为ERROR、退出码2。完成基线正确性审核、独立回放与审批记录后加--enforce，回归为FAIL、退出码1。精度和性能分别输出判定，当前同一个--enforce控制两者。

## 修复内容、验证和剩余工作

PR包含KeyPool历史地址、NoPE初始化、私有环缓存协调器、KDA非连续卷积状态暂存、custom-op可变别名及batch-invariant reduction兼容修复。二维speculative cache索引按请求和状态列全部暂存；schema声明output/conv_state原地写；prefill与mixed verify/prefill在必要边界同步。新增同步影响性能，需NPU复测和评审。

此前隔离A3环境相关回归106 passed、门禁单测11 passed。原始512-token budget混合复用场景在eager两次、graph三次启动均通过。MTP记录150 drafts、450 draft tokens、0 accepted，因此接受分支尚未覆盖。当前PR基于更新后的主线整理，必须用匹配运行时重新验证。

单图与TP16工具尚不等于完整nightly已接入。真实权重任务评分、MTP接受率和加速比、TP16全模型性能仍待完成。该CI提供基础实现回归看护，不能认证整网任务精度。

本次主线PR整理后，Flash工具测试41项通过，覆盖配置、采集布局、分析、policy、门禁失败路径和服务参数映射。Ruff检查与格式检查通过；历史NPU结果与当前分支测试应分别记录。
