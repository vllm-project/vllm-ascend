# CANN 分支升级到 vLLM Ascend 0.32

## 固定基线与合并方法

本次从已经验证过 SWA/P-D/Store 的 `backport/cann-swa-bounded-replay`
分支创建 `upgrade/cann-v0.32`。结果采用合并提交，保留两边历史。

| 项目 | 固定版本 |
| --- | --- |
| CANN / SWA / P-D / Store 功能基线 | `128a25b0af9a048a4346f6a66237b565db015f9b` |
| 原 CANN 基线 | `7e37e29a7803d6897a7113088e2b771f83205bc0` |
| 合入的 `upstream/releases/v0.32.0rc` | `21609b11ee574c477e5f36c925921e2355df160c` |
| 配套 vLLM 核心 | `v0.32.0`，`5df63a7203cdb9328d824a27b682e77c7b9928be` |
| CANN 上次升级使用的实际 0.30 内容基线 | `d94a832bb` |

CANN 的历史升级提交 `0040652a66d31a2b277588ac90d7d37e92757bc5`
明确记录了以 `d94a832bb` 的完整快照升级。由于此前采用 squash，Git
计算出的共同祖先仍在 2026 年 6 月。直接使用那个祖先会把已升级过的内容
再次作为冲突处理。因此，本次用已记录的 0.30 内容基线执行三方内容合并，
最终提交的两个父提交仍然是上表中的功能基线和 0.32 发布分支。
没有改写、覆盖或强推原有分支。

这不是“只替换版本号”或将所有冲突选择同一侧。初始 68 个冲突文件的记录见
[cann_v032_upgrade_conflicts.json](cann_v032_upgrade_conflicts.json)。
验证结果及基线对照见
[cann_v032_upgrade_validation.json](cann_v032_upgrade_validation.json)。

本次没有顺带合入功能基线之后的 CANN 提交 `4ec6dc772`（#20020）。
需要考察该提交时，应单独比较它与 0.32 原生 MegaMoE 的实现。

## 特性处理

| 特性 | 0.32 迁移后的实现 |
| --- | --- |
| SWA / encoder、decoder bounded replay | 使用 0.32 原生实现；删除 0.30 调度器源码改写补丁 |
| MRV1 的 replay 起点与共享 KV 保护 | 使用 0.32 V1 payload 补丁、slot masking 和窗口元数据 |
| Mooncake P/D 恢复窗口 | 使用 0.32 的 transfer-group 声明；传输请求自己的 SWA，不按 prefix-cacheable 过滤 P/D 窗口 |
| MultiConnector 选择恢复来源 | 使用 vLLM 0.32 的逐请求转发；新一次 lookup 清理旧选择，不合并多个子 connector 的缓存组 |
| 普通 Store + SWA | 使用 0.32 Store；强制异步接收完成后重建窗口，拒绝覆盖未对齐的共享前缀；接收失败走请求级回退 |
| Store 抢占与取消 | 同一调度输出中重新接纳的请求保留新登记信息；取消后接收注册保留到完成 |
| 私有 SWA / compressor ring | 普通 Store 使用 0.32 的 prefix-cacheable 组过滤；保留 CANN 私有实现的整除回归用例 |
| CANN range / C128 协议 | 隔离到 `ascend_store/cann/`；已有专有开关选择旧协议，普通配置继续使用 0.32 Store |
| CANN workspace MultiConnector / layerwise transport | 隔离到 `cann_multi_connector.py`、`cann_mooncake_layerwise_connector.py`，与 range Store 成对选择 |
| W4A8 MXFP / EPLB / NZ buffer | 使用 0.32 已演进的实现，避免重复保留旧专家权重列表 |
| A5 ds41 适配 | 保留 ds41 ABI，另外保留 0.32 原有 DSL ABI；无 ds41 包时使用原有路径 |
| A5 fused compressor / indexer prologue | 算子可用时保留 CANN 融合；原有 pooling、write/fold 路径继续存在 |
| W8A8 3D linear、MTP e/h projection prefix | 保留 CANN 改动 |
| force_topk sampler | 保留；NPU 编译延迟到实际使用，导入 MRV1 不初始化可选编译器；CPU 保留参考算法 |
| LoPT / timeline trace / 离线量化脚本 | 保留；trace 中的 V1 路径更新为 `worker.v1.model_runner` |
| encoder graph、310P、PCP 元数据 | 使用 0.32 对应实现，移除过时的重复适配 |

普通 `use_layerwise=true` 配置不会自动切换到 CANN 专有 range 协议。
只有现有 `use_layerwise_range`、`hybrid_cache_c128_chunk` 或
`layerwise_host_kv_offload` 开关选择对应兼容实现。
这些代码隔离是为了保留原协议及其测试，不代表它们完成了新一轮硬件验收。

### 已知的原有问题

原功能基线本身已有未通过的 CANN 专有测试。本次同时执行了旧基线对照，
没有把这些测试宣称为迁移成功：

- 旧 `test_coordinator.py`、`test_kv_transfer.py`、`test_pool_worker.py`
  等存在缓存可达性、接口/fixture 不一致问题。
- CANN workspace planner 的测试使用已经不存在的合并式 SFA
  `sparse_head_dim` 等字段；原分支的 spec 已经拆开 MLA 与 indexer，旧基线
  也有 11 个失败。该专有 offload 路径仍需单独适配及硬件验证。
- CANN Mooncake layerwise transport 的旧测试有 18 个失败，旧基线同样失败。
- 原 0.32 分支也存在配置模块导入隔离、未安装 Mooncake 时的 layerwise
  backend 测试失败。对照详情保存在验证 JSON 中。

因此，本次面向 DSv4.1 的普通 MooncakeHybridConnector / MultiConnector +
普通 Store 路径验证，不能作为 CANN workspace/range/C128 全功能验收。
没有在本机验证 NPU 算子构建、精度、网络传输或吞吐/延迟收益。

## 构建与启动调整

### 软件版本一起升级

vLLM 和 vLLM Ascend 仍是两个包。不能把本分支插件装在原 0.30 核心上继续跑。
构建文件沿用 0.32 的依赖、CANN 和算子安装配置。

A3 可以直接从本分支构建：

```bash
docker build -f Dockerfile.a3 \
  --build-arg VLLM_COMMIT=5df63a7203cdb9328d824a27b682e77c7b9928be \
  -t vllm-ascend:cann-v032 .
```

本地构建的插件版本字符串由 setuptools-scm 根据可见 Git 标签推断，
没有 `.git` 信息的构建也可能显示 `0.0.0`；因此请同时保存源码 SHA，
不要仅凭插件 wheel 的版本字符串判断升级是否成功。

Dockerfile 的上游默认基座为 CANN 9.2.1，并安装相应的 A3 ops-transformer
包；具体固定的下载地址和版本以本分支 `Dockerfile.a3` 为准。
本次包含原生算子变更，应重新构建插件及其算子。

先在新镜像中记录 `vllm.__version__`、两个包的加载路径、Git SHA、CANN、
torch/torch_npu 和实际加载的 Mooncake 版本。P、D、缓存池相关组件一起切换，
使用独立的测试缓存池，避免将旧布局的缓存对象当作新布局读取。

### 你的 1P1D 拓扑

保持 P 为 DP4 × TP4、D 为 DP8 × TP2，PCP=1、DCP=1。
MooncakeHybridConnector 的 `prefill` / `decode` extra config 也保持这组数值。

建议显式设置：

```bash
export VLLM_USE_V2_MODEL_RUNNER=1
```

从旧配置删除 `additional_config.enable_dsa_cp`。0.32 的 PCP 与旧 DSA-CP
接口不同；此次先按 PCP=1 验证，不需要添加 PCP 并行。
MRV1 仍有对应实现，源码移动到 `vllm_ascend/worker/v1/model_runner.py`。

0.32 核心中 `swa_bounded_replay` 默认值为 true，但还受模型及 Ascend 模式
检查影响。对比测试时两边都显式指定，避免继承默认值：

```bash
# ON
--swa-bounded-replay

# OFF
--no-swa-bounded-replay
```

普通 Store 遇到需要重建的 SWA 窗口时，会将接收切换到异步完成后再重建；
不要把命中 token 数量理解为所有 SWA 缓存组已经恢复。
此次保留 request-level Store、PCP=1、DCP=1 的验证范围；
`use_layerwise=true` 与 SWA 重建、C128 分块与 SWA 重建仍有明确限制。

先只用 MooncakeHybridConnector 跑通 P/D，再加普通 Store。使用已有
`examples/swa_replay/` 请求和 bench 脚本比较 ON/OFF，记录 TTFT、ITL/TPOT、
吞吐、精度和峰值内存。脚本的 `--mode` 只是结果标签，不会替你重启服务。

## 本地验证的含义

本次共运行 71 个独立文件：3130 项通过、48 项失败、18 项跳过，
另有 1 个收集错误。8 个文件存在失败，均匹配固定旧 CANN 或原 0.32
基线已有的失败节点；选定范围内没有未匹配的新增失败节点。
这不是“全仓库测试全部通过”的结论。

其中，普通 Store 的 SWA 集成测试 78 项通过，保留的 CANN Store
兼容路径 192 项通过；MRV1、MRV2、PCP、DSpark、W4A8、MoE 与配置检查
也执行了对应 UT，逐文件结果见验证 JSON。

Ruff、格式检查、拼写、导入限制、logger、包初始化、symbolic meta、
Python 编译及相对 0.32 的 Git whitespace 增量检查通过。完整 `bash format.sh ci` 在
依赖安装阶段被本机 HTTPS 证书链问题阻塞；改用本机 Node 再尝试后，
npm registry 返回截断 JSON，仍无法完成全部 hooks。

CPU 验证使用固定的 vLLM `v0.32.0` 源码、empty 构建和 torch CPU，
在设备边界使用仓库 UT mocks。不同文件分进程运行，避免 Store 测试的
模块替换影响其他 suite。具体 Python、torch 版本、环境、用例和结果均记录在
验证 JSON 中。

重点新增/保留的验证包括真实调度器配合 CPU 字节后端的 Store 加载、
接收完成前不能 replay、失败回退、共享 KV 不被覆盖、抢占后再次接纳、
私有 ring 不进入哈希、构造器选择，以及两种算子 ABI 的契约测试。
CPU 模拟不能代替 A3 上的正确性、P/D 网络传输、精度或性能验收。

新增的状态均有明确归属：失败请求集合属于 connector scheduler，接收失败
集合受 transfer thread 的锁保护；编译函数 cache 属于单个 worker 进程，
最多缓存一个 force_topk 编译入口，没有跨请求共享 KV 内容的新全局状态。

原 `cann_swa_*` 文档和 JSON 是 0.30 反合阶段的历史记录，不能作为本次
0.32 的测试结果。升级分支的验证结论以本页及新验证 JSON 为准。
