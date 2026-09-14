# FlashMLA BBND Draft：移植范围与调用逻辑

本 Draft 在 #16456 的 token-fused cache 上接入外部 FlashMLA 两个算子。
首次功能提交为 `7fef0589`；后续标注提交只增加代码注释与本说明，不改变执行逻辑。

## 固定对照版本

- 缓存基线：[ #16456 的 fe01d1df](https://github.com/HackClawMxw/vllm-ascend-fork/commit/fe01d1dfcc2c0ce4a27971b9c62904b26ebcfa86)。
- Python 接入参考：[ #16468 的 e36a3d05](https://github.com/maoxx241/vllm-ascend/tree/e36a3d05e9c869007439f36a5111992008a990c4)。
- 本地参考提交 `9204b0ba3770d7959c6f897c8908b6b51e53a8b2` 的 `mla_v1.py`、
  `attention_v1.py`、`model_runner_v1.py`、`platform.py` 与上述 `e36a3d05` 内容相同。
- 配套 vLLM：`a97dacb7106ee49f39f3d1fc6ae1800ff724e01d`。

代码内可以搜索 `FLASHMLA[` 定位标记：

| 标记 | 含义 |
| --- | --- |
| `FLASHMLA[REF-16468]` | 从 #16468 移植的逻辑，或接入其同样使用的既有框架机制 |
| `FLASHMLA[ADAPT-16456]` | 配合 #16456 的 fused BBND cache 保留或适配的位置 |
| `FLASHMLA[EXTERNAL]` | 使用外部包替代仓内 FlashMLA dispatcher 的位置 |
| `FLASHMLA[TODO]` | 对照发现的尚未移植或需要补齐的相关边界 |

## 已实现的修改

| 位置 | 修改和原因 |
| --- | --- |
| `envs.py` / `platform.py` | 增加 opt-in 开关；启用时选择 dense MLA backend，限定 A5、PCP=DCP=1、无量化 KV、无 KV transfer。 |
| `AscendMLABackend` | 支持 16 至 1024 的 16 倍数 kernel block size；缓存 shape 保持 #16456 的 `[P,S,N,D]`。 |
| `AscendMLAMetadataBuilder` | 构造一个 `flash` 描述整个 batch，接入 task-provider 协议；提前返回以跳过旧的 prefill/decode metadata 拆分。 |
| `attention_v1.py` 中的 FlashMLA helpers | 为 Q、schedule、长度、block table、slots 和 positions 分配并复用设备 buffer。普通 GQA backend 没有因此切换到 Flash GQA。 |
| `_build_flash_attention_metadata` | 每步按设备侧 query offsets 和 KV 长度更新 buffer，调用真实 metadata 算子，把其结果复制到稳定 schedule 地址。 |
| `NPUModelRunner` | FlashMLA 开关下跳过旧的 optimistic CPU seq-len correction；向 builder 传递实际选择的 kernel block size。 |
| `AscendMLAImpl.__init__` | 限定 latent 512、PE 64、一个 KV head、BF16 attention，关闭这一路的 MLAPO、NZ 和 head padding。 |
| `AscendMLAImpl.update_graph_params` | FlashMLA 路径提前返回；每步动态数据通过稳定设备 buffer 更新，不再更新 FIA task handle。 |
| `AscendMLAImpl.forward / _forward_flash` | 执行 Q/KV 投影、归一化、按层 RoPE、paged scatter、主算子、V-up、gate 和输出投影。 |
| `attention/flash_mla.py` | 延迟导入 `cann_ops_transformer` 并调用其两个 dispatcher；缺包时报错，无 `_C_ascend` 回退。 |
| `test_flash_mla_metadata.py` | CPU mock 覆盖长度更新、padding 屏蔽、buffer 复用和无 executor 时的直接执行；未执行真实算子。 |

`worker/device_metadata.py` 的 stream、Event、ExternalEvent 和 reuse fence 在基线中已经存在。
本 Draft 让 MLA builder 提交任务，并让 attention 使用同一个 schedule group ID 等待；没有新写一套 executor。

## 一次 attention 的具体顺序

```text
初始化：#16456 _reshape_kv_cache_tensors
  -> fused_cache[P,S,1,576]，保留首轴 stride 和 storage offset
  -> kv_caches[layer_name] -> bind_kv_cache -> attention 的 kv_cache

每步：runner 准备当前 batch 的 common metadata
  -> MLA builder 返回 flash 描述，并登记 metadata task
  -> 既有 DeviceMetadataExecutor 在独立 NPU stream 执行 task
     -> 更新 cu / used_q / cache_lens / block_table / slots / positions
     -> flash_mla_with_kvcache_metadata -> schedule.copy_(结果)
  -> AscendMLAImpl.forward -> _forward_flash
     -> 等待 metadata event
     -> 投影/归一化/RoPE -> Q[T,H,576]、cKV[T,1,512]、kPE[T,1,64]
     -> scatter 将当前 token 写入 fused cache 的两个共享切片
     -> flash_mla_with_kvcache(Q, fused_cache, ..., schedule)
     -> latent[H,T,512] -> V-up -> 可选 gate -> o_proj -> 屏蔽 padding 输出
```

开关分支位于总 `forward` 的入口，所以当前实现同时覆盖 **prefill、decode 和混合 batch**。
metadata 中旧的 `num_decodes/num_prefills=0` 是被跳过路径的占位值，不表示本次没有请求。

`block_table` 保存 kernel block ID；`slot_mapping` 决定当前 token 的写入位置。
`cu_seqlens_q` 是累计 Q 边界，`seqused_q` 和 `cache_seqlens` 是每请求的 token 数。
它们的 buffer 地址复用，内容每步更新；额外一行长度为零的 request 承载尾部 padding。
`slots=-1` 阻止这些 token 写真实缓存，`token_live` 同时屏蔽它们的输出。

`max_seqlen_q=T`、`max_seqlen_kv=table_width*kernel_block_size` 是该 buffer 的容量上界。
每请求的实际长度仍通过上述设备 Tensor 传递；KV 长度不需要除以 block size。
两次算子调用的长度、head 数、576/512、mask mode 和 Q layout 必须一致。

`mask_mode=3` 用于 causal，`mask_mode=0` 不传 mask。主算子 `layout_q="TND"`，
`layout_out="NTD"`，返回 `[H,T,512]`，恰好匹配已有 `_v_up_proj` 的 head-major 输入。
改为 TND 输出时必须同时适配投影；当前不读取 softmax LSE。

## 相对 #16468 的两处接口适配

1. **缓存输入**：#16468 使用 `[P,S,576].unsqueeze(1)`，按 `PA_BNBD` 传入。
   本 Draft 直接传 #16456 已有的 `[P,S,1,576]`，按 `PA_BBND` 解释。
   scatter 的两个切片已经是四维，所以省去参考代码的 `unsqueeze(2)`。
   缓存没有做 `cat`、整份重排或压紧；Q 和当前 token 的小张量处理仍存在。
2. **算子来源**：将参考中的 `torch.ops._C_ascend` 调用换为外部
   `torch.ops.cann_ops_transformer`。AICPU schedule 的计算由 metadata 算子执行，
   Python 不手工生成 section/核切分字段，也不移植这两个算子的 C++ 实现。

schedule buffer 仍沿用参考代码的 A5 `36+72` 核数容量公式。当前检查返回值的 dtype/numel，
这只是容量防错；包的实际执行、stride 消费和图回放仍需后续验证。
跳过 CPU seq-len correction 也不等于整个 runner 已经不使用 CPU metadata。

## 对照发现的缺口与未移植范围

| 范围 | 当前状态和影响 |
| --- | --- |
| 按 `num_heads_q` 分组 builder | #16468 有，当前 Draft 未移植。相同 cache spec、不同 Q head 数的 MLA 层可能错误共用 query/schedule buffer，需要补齐。 |
| 混合 MLA 与普通 attention 的 CPU seq-len correction | 当前按全局 FlashMLA 开关跳过；普通 attention 路径仍可能依赖 CPU mirror，混合 runner 需要按消费者收紧条件。 |
| MLA DSpark 的 context KV writer | #16468 在 `exec_kv_prefill` 另加 fused 写入分支；当前只在 `_forward_flash` 中实现写入，context-only 调用入口尚未适配 BBND。 |
| DSpark 的 query 重建、draft graph padding/等待 | proposer 改动未移植，不能宣称完整 DSpark/投机解码接入。设备侧长度更新只是其中一环。 |
| GQA Flash 算子与 GQA cache split | 未移植；不属于这两个 MLA 算子的调用。 |
| KDA/Mamba/卷积与相关保护条件 | 未移植 #16468 的这些改动，沿用 #16456 基线；混合 K3 模型依赖需单独确认。 |
| allocator/spec、预算、whole-page COW、清零 | 延续 #16456，未搬 #16468 的另一套分配和生命周期适配；本轮只在 fused 交接处标明来源。 |
| ACLGraph 日志等辅助改动 | 未移植；当前已接入 metadata event 依赖，但图能力尚未运行验收。 |

因此该 Draft 是 **#16456 缓存上的 FlashMLA Python 调用链初版**，不是 #16468 的完整移植。
上表中的 head 分组、DSpark 上下文和混合 backend 条件涉及正确性，不能笼统归为“无关改动”。

## 当前验证记录

- 首次功能提交：Python 语法和 `git diff --check` 通过。
- 本地没有 torch/pytest，新 CPU mock 单测尚未执行。
- 本轮注释更新：对照更新前后的 Python AST，确认不改变执行语义。
- 外部算子数值、ACL Graph、DSpark、COW/清零及整模型验收尚未执行。
