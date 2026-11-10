# Qwen PLE 的 MemFabric GVA offload

## 范围

在 Model Runner V2 上，将 Qwen4Exp / Qwen3.8-Next 的 BF16 PLE 表放入
MemFabric Offload 管理的 Host DRAM，按原 ETP vocabulary range 查表。
不改变 n-gram ID、PLE gate/投影/norm、short-conv、ETP reduce 或外层
sequence-parallel 行为；不修改 DeepSeek Engram backend。

基于 [UVA PR #20560](https://github.com/vllm-project/vllm-ascend/pull/20560)
移植模型接入与生命周期逻辑。GVA 使用独立 `offload.Offload()` 实例，
初始化、分配、copy 和释放始终通过同一实例；不使用模块级默认池。
该接口对应 [#20393](https://github.com/vllm-project/vllm-ascend/pull/20393)
及 MemFabric 当前源码的实例方案，替代早期设计中的显式 `pool_id` 方案。

## 依赖与限制

- MemFabric 必须提供 `offload.Offload()` 和 `sparse_copy`，并已为当前
  torch/torch_npu、CANN 和硬件构建 Offload kernel。仅有模块级 API 的
  1.3.0 会明确报错，不回退到全局单例或 UVA。
- 当前验证设备为 A3；A2/A5 未验证。原生库对各平台的支持仍是前提。
- 仅支持 BF16 PLE、ETP=TP、Model Runner V2，不支持 THP、异步 prefetch。
- BF16 每行字节数须为 32 的倍数且不超过 32768；每池容量按 GiB 向上对齐。
- `dp_shared_memory=false`：各 rank 使用独立 LOCAL 实例和 backing。
- `dp_shared_memory=true` 且同节点有多个 Engram DP/PCP peers：使用 SHARED
  实例，组内 rank 0 贡献 backing 并加载权重，其他 rank 不重复分配。
  跨节点或没有本地 peers 时沿用原 PR 的 private-table fallback。
  本次不实现跨节点 owner-backed 或新增 row-sharding。
- 同一 SHARED 池的参与 rank 必须按相同顺序初始化实例；部署应保证其他
  Offload 用户的初始化顺序一致。MemFabric 原生 entity ID 预算为每进程
  64，包含其他 Offload 实例和失败的初始化尝试；不能无限重建模型。

## 配置

包含 PLE 层的 Qwen4Exp 默认开启 GVA offload。未配置共享端点时使用私有表。可通过以下参数显式关闭 offload：

```bash
--engram-config '{"cpu_offload":false}'
```

共享表使用 `dp_shared_memory=true`，并通过 additional-config 指定每个
逻辑表/ETP shard 独占的 TCP store endpoint：

```json
{
  "ple_gva": {
    "store_urls": {
      "<实际 PLE embedding prefix>/etp0": "tcp://<owner-IP>:31592",
      "<实际 PLE embedding prefix>/etp1": "tcp://<owner-IP>:31593"
    }
  }
}
```

使用模型实际 prefix，不要直接复制占位符。缺失绑定时错误信息会包含完整
key；不同池不能复用 endpoint。所有参与 ranks 必须收到相同映射，端口
不得与 DP RPC、KV connector、API server 或其他作业冲突。

## 数据与关闭顺序

owner 保留可由 upstream checkpoint loader 写入的 CPU tensor；consumer
仅持有 meta parameter 记录形状和 dtype。权重分片仍由 upstream loader
根据 checkpoint_start 与 ETP range 求交，组内传播加载错误。

lookup 在设备上生成源地址、目标地址和 copy 长度，然后调用 MemFabric
`sparse_copy`。无效或非本分片 ID 的 copy 长度为零，输出保持零。
奇数个 ID 补零长度条目，兼容 native KV copy 对两半配对的要求。
native API 被封装在 custom op 内，避免 Dynamo 跟踪 `data_ptr()`；
ACLGraph replay 使用设备上的 ID 和 copy metadata。

runner 先停止并释放 graph，再关闭表。close 同步设备和共享 peers，
解除 parameter alias，释放所属 allocation，最后关闭本实例。
close 失败后保留 owner 以便重试，进入关闭状态后拒绝新的 lookup。
不要在其他 rank 仍 replay 时单独关闭 owner。

## 验证命令

在已安装匹配 vLLM、vLLM Ascend 和 MemFabric 的开发环境执行：

```bash
python -m pytest --confcutdir=tests/ut \
  tests/ut/distributed/test_ple_gva.py \
  tests/ut/models/test_ple_gva_embedding.py \
  tests/ut/test_ascend_config.py \
  tests/ut/worker/test_model_runner_v2.py -q

python tests/manual/check_ple_gva.py --device 0 --adapter
python tests/manual/check_ple_gva.py --shared-devices 0 1 \
  --port 31591 --store-port 31592
```

夜间测试入口：
`tests/e2e/nightly/single_node/ops/singlecard_ops/test_ple_gva.py`。
未安装独立 Offload 实例 API 时该测试跳过。

## 当前验证状态

2026-11-09，Ascend main `37854b83e`，配套 vLLM `24420b7a`，
MemFabric 源码 `0d7bb27b` 构建为 1.5.0，torch/torch_npu 2.10，CANN 9.2.1：

| 检查 | 结果 |
| --- | --- |
| 相关四个 UT 文件 | 419 passed，1 skipped |
| A3 eager lookup，空/奇偶输入、无效/越界/重复 ID | 通过 |
| 修改 ID 后 ACLGraph replay | 通过 |
| adapter `torch.compile(fullgraph=True, backend="eager")` + ACLGraph | 通过 |
| 同进程两个池，关闭其中一个后继续访问另一个 | 通过 |
| 同节点两 rank，owner backing + consumer 无重复 backing | 通过 |
| 完整真实权重服务、文本/图文、MTP、EP、128K 容量 | 尚未验证 |
| TTFT/TPOT/吞吐及与 UVA 的性能对比 | 尚未测量 |

上述小表与编译测试不能替代完整模型验证，也不证明性能收益。
切换服务前须在独立环境构建匹配的 Ascend 算子，并完成真实权重服务测试。
