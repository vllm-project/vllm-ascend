# DeepSeek-V4-Flash Mooncake offload 开发与整体验收

本实现对应方案 v1.9。只将 Attention compressor KV（随上下文增长的压缩历史缓存）放入跨层轮转槽。
SWA（近期滑窗缓存）、两个 compressor state（主注意力和索引压缩器各自的续算状态）和 Indexer K/scale（索引历史及缩放信息）保留本层内容。

## 实现与运行配置

模型缓存布局入口为 `vllm_ascend/models/deepseek_v4/cache_config.py`。
实际压缩历史注册名为 `.self_attn.attn`（兼容以 `.self_attn` 结尾的注册名）；只按组件选择共享，不能把整层辅助缓存一起折叠。
普通布局继续使用既有跨组底层存储；开启共享时，将共享槽和驻留组件拆成独立分配，避免驻留数据与共享覆盖范围重叠。
容量计算、MRV2（第二版模型执行器）的实际分配和搬运地址均消费同一份描述。
每份分配额外预留 2 MiB（满足现有地址对齐分配器需要的空间），不会按逻辑别名重复收费。

配对 vLLM 提交：`24420b7a341af37a9bc3186e6fc6062512e0aa58`。
在该版本上，调度端通过 `kv_connector_block_state.get_block_ids()`（读取本轮真实有效分组块表的接口）替换请求跟踪表。
即使本轮没有分配新块也同步，保留滑窗回收后的空块位置。接口缺失或表缺失会报错。

首轮采用 eager（直接执行算子）、同步调度、关闭推测解码；PP（流水线并行）、PCP（预填充上下文并行）、DCP（解码上下文并行）均为1。
DP（独立数据并行副本）和 TP（张量并行）沿用现有服务配置，目标为 DP4/TP1＋EP（专家并行）。
必须启用 MRV2，并配置 `--enforce-eager --max-num-batched-tokens 4096`。
沿用原有模型、量化和 Mooncake 连接配置，Store（缓存池连接器）配置示例：

```json
{
  "kv_connector": "AscendStoreConnector",
  "kv_role": "kv_producer",
  "kv_connector_extra_config": {
    "backend": "mooncake",
    "use_layerwise": true,
    "layerwise_num_shared_buffers": 2,
    "layerwise_independent_layers": [0, 1],
    "layerwise_prefetch_layers": 2
  }
}
```

在原有 `--additional-config` 中合并 `"kv_pool_range_debug": true`，打开诊断。
分别以 B（每个兼容域的共享槽数）=1/2/3/4 启动并保留完整服务日志。
全驻留对照可使用相同配置并将 `layerwise_independent_layers` 设置为 `"all"`。
这里的对照仍保留外部池能力，可另用未开启共享的原服务核对普通布局。

## 六项整体验收

| 项目 | 操作和证据 | 判定边界 |
| --- | --- | --- |
| 共享和驻留布局 | B=1/2/3/4；核对 `config`、`allocation`、`component` 的全部地址、组号、页大小、步长、前驱 | 仅压缩历史同域轮转；辅助组件和 L0/L1 不误共享 |
| 物理预算一致 | 汇总唯一 `allocation.address` 的 `bytes`，与 `config.planned_bytes` 比较；另核对对齐预留 | 相同底层分配只计一次，驻留组件未按 B 折叠 |
| 跨 chunk（一次调度处理的一段 token）续算 | 4096 分段，4096/4097/8193/12289 等输入；运行下面的确定性输出对照 | 读取准确 E（上一段计算结束的原始 token 位置）；同请求辅助数据不轮转换入，输出一致 |
| 外部前缀和并发 | 冷请求、重复、不同后缀和混合长度并发；核对 `restore` 来源及已有 commit（对象发布可读）日志 | 不能仅凭主压缩历史可读就认为完整命中；缺少必需组必须缩短命中或失败 |
| 槽位覆盖保护 | 新池或新输入下开启一次 PUT（从设备保存到池）延迟；核对 `source_done → slot_release → range_write_begin/compute_release` | 必须观察到真实 `slot_wait.blocked=true`；同进程使用单调时间比较 |
| 生命周期和错误 | 正常完成、客户端取消、GET（从池恢复）失败、PUT 失败、commit 失败及小容量池 | 必需数据失败后不能继续使用不完整缓存；核对撤销、读会话释放和 `terminal_cleanup` |

同请求续算时，混合组中的索引数据依旧可发布给外部读者，但恢复搬运只选择共享压缩历史。
外部前缀恢复保留全部必需组件任务，并复用基线协调器的端点与可达范围检查。
公共对象身份包含实际组件规格、格式、模型配置和并行分片，不添加 DP 专用隔离。

### 确定性输出与并发对照

在服务器仓库根目录运行；两个服务使用相同权重、分词器和采样配置。
脚本直接使用整数 token 输入，确保输入长度准确。并发批次包含重复输入及混合长度。
对照结果只证明已运行用例的文本一致，不代替异常路径验证。

```bash
python tests/e2e/cases/features/kv_pool/tools/compare_dsv4_offload.py \
  --baseline http://BASELINE_HOST:PORT \
  --candidate http://OFFLOAD_HOST:PORT \
  --model dsv4 --concurrency 4 > comparison.log
```

### 延迟与失败注入

以下参数放在 `kv_connector_extra_config` 中，仅用于服务器验收。
要求同时打开 `kv_pool_range_debug`；默认关闭。
每个后端实例只在第一次非空操作时注入一次，事件会明确标记为故障注入。

| 参数 | 值 | 用途 |
| --- | --- | --- |
| `offload_debug_put_delay_ms` | 0～5000 的整数，建议500 | 第一次实际 PUT 延迟，验证后继层等待 |
| `offload_debug_fail_once` | `"get"`、`"put"` 或 `"commit"` | 对应操作返回失败，验证错误处理；默认空字符串 |

各故障场景单独重启服务。GET 故障需要先保存再触发后续恢复。
延迟场景应使用未发布的新输入或新池，否则已存在对象可能跳过 PUT，不能得到有效延迟证据。
池容量不足通过服务器 Mooncake 容量配置复现，不用注入失败冒充容量不足。
取消可在长请求运行时中断客户端，再发送正常请求检查资源回收与后续可用性。

### 回传不超过300行

```bash
python tests/e2e/cases/features/kv_pool/tools/summarize_dsv4_offload.py \
  b1.log b2.log b3.log b4.log comparison.log delay.log get-failure.log put-failure.log \
  > dsv4-summary.txt
wc -l dsv4-summary.txt
```

汇总脚本逐行读取，输出硬上限280行，优先保留首批错误、配置、传输计数和六项验收摘要。
服务器完整日志保留；只需回传汇总文件。输出中的 `INSUFFICIENT` 表示证据不足，不等于通过。
故障注入运行中出现的预期错误需要结合场景核对，不能按正常成功场景解释。
不同文件按独立时钟范围处理，建议同一进程的完整日志不要拆成多个文件后再做顺序判定。

## 本机验证范围

```bash
python -m pytest --noconftest -q tests/host/test_dsv4_mooncake_offload.py
bash format.sh ci
```

主机测试直接执行生产代码中的规划、地址构建和范围选择函数，隔离框架导入，包含真实主机字节复制检查。
这不证明 NPU 算子、跨流同步、服务启动或多卡输出正确；这些结论由服务器六项整体验收提供。
现有平台布局补丁只增加对模型布局入口的调用。后续上游支持相同模型布局入口时可移除该版本桥接，模型侧规格仍保留。

## P/D 部署修正（2026-11-10）

独立 Store 的配置只用于单节点 offload。`load_balance_proxy_layerwise_server_example.py`
先请求 D，由 D 回调 `/v1/metaserver` 后才调度 P；独立 Store 不实现此回调。
此前使用独立 Store 的 D 返回成功而 P 没有请求，不能作为 P/D 验收证据。

本次交接使用 D2D（P 的设备缓存直接复制到 D 的设备缓存）：

- P：`MultiConnector`，组合 `MooncakePushConnector` 与 `AscendStoreConnector`。
- D：独立 `MooncakePushConnector`，全驻留缓存，不配置共享槽，不从 Store 恢复交接数据。
- 两端使用同一版 Push 代码。P 在发送前核对模型、缓存组件和缓存格式。
- P 仅在最终 chunk 逐层发送该请求的完整有效历史、滑窗、状态及索引；中间 chunk 只 offload 到 Store。
- D 保留最后一个 prompt token 本地计算。P 在原始 token 端点 N−1 完成预填充并产生一个供交接流程使用的 token，代理最终响应来自 D。
- 每个组件按自身组号、当前有效块表和两端真实页步长传输；必要空块、格式不一致均拒绝。
- Store PUT 与 D2D 读取都释放后才允许共享槽覆盖；step 结束前排空 D2D，禁止回收仍被读取的请求块。

本路径限制 TP=PP=PCP=DCP=1，允许 DP4＋EP；首次基线验证使用 eager 和同步调度，关闭推测解码。
两端使用相同模型、量化、缓存格式及块大小。联合验证使用
`--async-scheduling --enable-prefix-caching`。P 的共享压缩槽只从 Mooncake
恢复经对象校验的前缀，D 的常驻缓存可复用本地前缀。D 有部分本地命中时，
Push 从命中边界补齐历史页，并按最终端点恢复不参与前缀缓存的有效滑窗和状态。
对同一长前缀分别跑冷请求、重复请求和共享前缀但后缀不同的并发请求，对比输出。
从汇总中检查 D 本地前缀和 Push 补齐 token 数。

仓库根目录中，P 的启动命令使用：

```bash
--kv-transfer-config "$(cat examples/disaggregated_prefill_v1/dsv4_offload_p.json)" \
--enforce-eager --no-async-scheduling --max-num-batched-tokens 4096
```

D 的启动命令使用：

```bash
--kv-transfer-config "$(cat examples/disaggregated_prefill_v1/dsv4_offload_d.json)" \
--enforce-eager --no-async-scheduling
```

两端保留原有 MRV2、模型、DP/EP、Mooncake 环境配置，并开启 `kv_pool_range_debug=true`。
P2P 侧信道示例使用 P 14500～14503、D 14600～14603，需可互访且不被其他服务占用。
代理仍使用用户已有命令：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_layerwise_server_example.py \
  --host 141.61.133.128 --port 1999 \
  --prefiller-hosts 141.61.133.128 --prefiller-ports 8099 \
  --decoder-hosts 141.61.133.128 --decoder-ports 8066
```

压测和对照脚本的 candidate 必须指向 `http://141.61.133.128:1999`。
观察 P 的逐层 Push 写入与共享槽等待、D 的接收完成，以及任何 Push 失败或超时。
先跑新输入的短请求确认 P/D 都参与，再跑 4097/8193/12289 等跨 chunk 用例。
汇总仍不超过280行；P/D 接通不自动代表六项验收全部通过。

### 本轮方案核对

| 方案项 | 源码核对及修正 | 待服务器验证 |
| --- | --- | --- |
| 第2～3章布局与预算 | 现有组件计划、独立分配与统一预算保留；P/D 地址差异由真实页步长处理 | B=1/3/4，普通布局回归 |
| 第4章续算和块表 | Store 已同步有效块表；新增 P2P 最终端点同步，避免携带已回收 SWA 块 | 长输入、多并发及非对齐尾页 |
| 第5章外部完整恢复和身份 | 复用现有协调器；P2P 新增模型、组件及格式身份校验，不以页长相等代替兼容性 | 缺辅助组件、不同后缀、并发发布 |
| 第7章诊断 | 补无 GET 时的槽位等待日志、step 端点与交接计数；D 普通布局不再误计为共享布局缺失 | 延迟 PUT、失败、取消、容量不足 |
| P/D 部署（原方案未明确） | 补最终 chunk 多缓存组 D2D、回调连接器组合及两端示例 | 首次完整 P/D 服务运行和确定性输出 |

本轮 CPU 用例和静态检查只能证明其覆盖的生产函数行为；全部 NPU 场景仍以服务器日志判定。
