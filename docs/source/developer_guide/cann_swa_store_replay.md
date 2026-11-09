# CANN 分支 MultiConnector 与 AscendStore SWA replay

此扩展位于 `backport/cann-swa-bounded-replay`，包含之前的
`222d4cb7a` 启动兼容修复。vLLM 核心仍固定为
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`（v0.30.0）。

实现提交：`0b3737e3b` 为 MultiConnector 来源转发，`c48fba428` 为 Store 恢复与失败处理，
`eaedc0408` 防止待接收缓存被并发保存钩子提前发布。
原来的直接 MooncakeHybridConnector 路径可以继续独立验证。

## 1 行为与范围

| 本次请求选择的来源 | 恢复的内容 | SWA 处理 |
| --- | --- | --- |
| MooncakeHybridConnector | 请求自有全局 KV、SWA 尾部及原有续算状态 | 已接收窗口时跳过重建 |
| AscendStoreConnector | 参与前缀持久化的缓存组 | 等待加载完成，再补算缺失的 SWA 窗口 |
| 本地前缀缓存 | 本地缓存命中 | 沿用原来的窗口重建 |

MultiConnector 每次 lookup 都重新选择来源，报告的是选中 child 的组，
不会把 Hybrid 与 Store 的能力取并集。抢占和重新准入不会沿用旧来源。

Store 的查询、读入和写出都排除非持久化组，同时保留原有 group ID。
因此开启 bounded replay 时，缺少 SWA 快照不会把本来有效的 Store 全局前缀命中压到零。
当 Store 命中太短、不足以抵消窗口重建成本时，沿用调度器的冷算回退。

本次支持请求级、非 layerwise Store，要求 PCP=1、DCP=1。
`hybrid_cache_c128_chunk=true` 这一可选分块传输模式尚未接入，启动会明确拒绝。
这里的限制针对可选传输模式，不是禁止模型正常的 C128 压缩 KV 缓存组；
正常压缩组的哈希粒度适配已包含 C4/C128 CPU 回归。

MRV1 与 MRV2 都有协议回归，MRV2 不支持旧 `enable_dsa_cp=true`。
模型原有的 SP、图模式、PP、DCP 等 decoder 裁剪门禁仍然生效。
不能仅根据 flag 被接受或服务启动成功，认定 decoder 裁剪已经执行。

## 2 P、D 配置

两端都更新本分支的完整提交链，重启引擎和全部 worker。
已有自定义算子安装的情况下，这次扩展只修改 Python，无须单独为它重新编译算子。
使用 wheel 安装的环境需要更新 Python 包，源码更新不会自动替换旧 wheel。

先保持现有 P DP4/TP4、D DP8/TP2 和 PCP=1，不同时改变拓扑。
两端使用相同权重、tokenizer、量化与兼容的 KV 布局、block size、模型目录名。
Store 沿用已有的缓存键和布局，不新增跨 TP 布局转换。
CPU 验证不等于已验证 A3 TP4→TP2 的真实数据布局与网络传输。

下面是 D 的组合配置；root 和 child 的 engine ID 不要重复声明：

```shell
KV_TRANSFER_CONFIG='{
    "kv_connector": "MultiConnector",
    "kv_role": "kv_consumer",
    "kv_load_failure_policy": "recompute",
    "engine_id": "1",
    "kv_connector_extra_config": {
        "connectors": [
            {
                "kv_connector": "MooncakeHybridConnector",
                "kv_role": "kv_consumer",
                "kv_port": 30100,
                "kv_connector_extra_config": {
                    "prefill": {"dp_size": 4, "tp_size": 4},
                    "decode": {"dp_size": 8, "tp_size": 2}
                }
            },
            {
                "kv_connector": "AscendStoreConnector",
                "kv_role": "kv_consumer",
                "kv_connector_extra_config": {
                    "lookup_rpc_port": 10001,
                    "backend": "mooncake",
                    "consumer_is_to_load": true,
                    "consumer_is_to_put": false,
                    "load_async": true,
                    "use_layerwise": false
                }
            }
        ]
    }
}'
```

P 保留自身 engine ID、端口，把 root、Hybrid child 和 Store child 的角色设为
`kv_producer`；Store producer 可以读取前缀并写入缓存。
两端的 `prefill`/`decode` DP、TP 描述保持一致。
Mooncake Store 沿用现有的 `MOONCAKE_CONFIG_PATH`、master 地址及内存池配置。

D 不需要读取 Store 时，可以继续省略 `consumer_is_to_load` 或设为 false；
此时 Store 不参加 D 的命中选择，但外层 MultiConnector 仍可把 Hybrid 恢复组正确转发。
要验证 Store 命中，必须有实际启用的读取端和可访问的有效缓存对象。

replay ON 时 Store 会自动使用异步加载，即使原来写了 `load_async=false`。
这样调度器可以在首次模型计算前处理接收失败。
`load_async=true` 本身不会启用 `consumer_is_to_load`。

## 3 加载失败与缓存保护

Store 后端返回失败码、空结果、长度不符或抛异常，都会结束本次接收并按请求 ID
报告失败，不会把混合缓存中的数字 block ID 当作唯一失败身份。
MultiConnector 将失败结果转发给核心调度器：

- `kv_load_failure_policy=recompute`：回退重算；同一请求不反复读取已失败的 Store 对象。
- `kv_load_failure_policy=fail`：结束请求，不进入窗口重建。
- 接收期间取消请求：沿用核心的延迟释放，在接收完成之前保留目标缓存块。

本地已命中的共享全局块不重复加载；本地边界不能满足 Store 整页读取要求时，
放弃本次 Store 扩展命中，继续使用本地缓存与重建路径。
重建通过已有 replay slot 屏蔽保护全局 KV，CPU 回归同时检查字节内容、block ID 和引用计数。
真实 NPU 内存安全仍需上机验证。

## 4 上机验收

先两端 `--no-swa-bounded-replay` 跑组合配置基线，再两端
`--swa-bounded-replay` 跑相同请求。固定 runner 和其他配置。
OFF/ON 保持相同的 Store 冷/热条件；不能用一组清空缓存、另一组已有热缓存的结果比较收益。

检查三类日志：

```text
SWA KV load source: ...AscendStoreConnector; groups=(...).
SWA bounded replay is active: a prefix hit is rebuilding its window.
SWA P/D handoff: received SWA window; skipping prefix replay.
```

前两条用于确认 Store 来源和缺失窗口重建，第三条用于确认 Hybrid 完整窗口交接。
Store 只是配置存在、启动时输出 `AscendStore SWA replay`，都不证明实际命中。
来源选择和重建标记是进程级首次激活日志，应使用本次服务启动的日志；
它们不能证明两条日志属于同一个请求。

可沿用 curl 重复前缀测试，通过代理验证 P 命中 Store 后向 D 交接：

```shell
BASE_URL=http://<PROXY_IP>:<PROXY_PORT> MODEL=dsv41 \
    bash examples/swa_replay/03_prefix_reuse.sh
```

如果本地 prefix cache 把所有命中都接住，就不会触发 Store 读取。
单独的 Store 功能验收可关闭本地 prefix caching，让重复前缀进入外部查找；
性能对照的 OFF/ON 必须采用相同策略。

probe 也可保存 Store 的日志证据。先记录 OFF 基线，重启 ON 后执行：

```shell
python tests/e2e/models/swa_bounded_replay_probe.py \
    --base-url http://<PROXY_IP>:<PROXY_PORT> --model dsv41 \
    --baseline /tmp/store-off.json --output /tmp/store-on.json \
    --store-log /tmp/prefill-on.log
```

`--store-log` 选实际命中 Store 的 P 或 D 的日志，不固定为某一端。
完整 P/D 交接验收仍可同时提供 `--prefill-log`、`--decode-log`。
这个 probe 的算术文本检查仅是 smoke test；bounded replay 是近似计算，
还需要业务质量集和失败率检查。最后通过代理测 TTFT、TPOT/ITL 与吞吐。

## 5 验证记录

2026-11-07 CPU 主回归：**1658 passed、23 skipped、513 subtests passed**。
覆盖真实 MultiConnector 构造、元数据与完成通知，真实 Store worker/scheduler、
发送和接收线程；后端以 CPU 内存模拟，模型计算、NPU 和真实网络未执行。
另用真实 v0.30 C4/C128 MLA 规格覆盖不同物理行数、本地缓存开关和 P/D 角色，
检查逻辑页地址、传输字节数及共享页保护，保留现有缓存键映射。
旧 Store 扩展测试仍有与修复前一致的失败，完整 CI 不能视作通过。
详细范围、基线对照与复现命令见[验证记录](cann_swa_store_replay_validation.json)。

## 6 2026-11-09 小环形状态组的哈希断言修复

收到 13 个缓存组、`hash_block_size=128` 的发送线程整除报错后，
CPU 回归复现了相同路径：非持久化环形状态组的 `block_size` 是物理容量，
可以为 4、8 等，不能作为 128 token 前缀哈希的分组粒度。
旧实现先生成哈希、后应用过滤掩码，导致本应排除的组仍触发断言。
仅凭日志中的组数量无法确定实际出错的 group ID。

修复在 token/chunk 生成前按 `prefix_cacheable` 语义排除私有组；
发送线程也在确认存在待发送 chunk 后才生成事件哈希。
同步加载和异步接收共用这一过滤，SWA 关闭时私有环形组同样不参与 Store。
可持久化组仍必须满足整除约束，不采用“所有不整除的组都静默跳过”的规避。
既有合法对象的键映射和布局保持不变。

新增 30 个 CPU 回归覆盖原生 C4/C128、SWA、私有环形状态混合的 13 组布局，
检查真实字节写入/读回、私有组未传输、KV 事件开关和非法可缓存组仍报错。
相关主回归结果为 **1688 passed、23 skipped、513 subtests passed**。
这仍不包含 A3、Mooncake 网络或模型精度/性能验证。
新增记录见[小环形组修复验证](cann_swa_store_ring_hash_validation.json)。

更新后两端重启全部 worker，沿用原来的 OFF/ON 对照配置。
不要只检查报错消失，还要通过选中 Store 的来源日志和实际缓存读入确认外部命中生效。
