# MLA leader fanout 镜像构建与真机验证

本说明对应当前源码中的原生实现，连接器名称是 `MooncakeHybridConnector`。
它要求 D 侧单节点 TP > 1、复制型 hybrid MLA KV、P/D PP=1、D PCP/DCP=1。
首次验证建议沿用已有的 DeepSeek-V4-Flash 2P1D / TP8 部署。

当前修改仍在工作区，尚未创建 Git commit，也尚未构建镜像或进行 Ascend 真机验证。

## 1. 固定源码并传到 Linux 构建机

在 Mac 上执行，下面只暂存本次修改涉及的文件：

```bash
cd /Users/junlin/Projects/vllm-ascend
git switch -c codex/mla-leader-fanout
git add \
  vllm_ascend/distributed/kv_transfer/kv_p2p/mla_fanout.py \
  vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py \
  tests/ut/kv_offload/test_mla_fanout.py \
  tests/ut/kv_offload/test_mooncake_hybrid_connector.py \
  examples/disaggregated_prefill_v1/mla_leader_fanout.md \
  examples/disaggregated_prefill_v1/mla_leader_fanout_validation_zh.md
git commit -s -m 'perf(kv-transfer): deduplicate replicated MLA reads with leader fanout'
git bundle create /tmp/vllm-ascend-fragpack.bundle --all
```

将 bundle 上传到构建机。示例中的登录地址和路径需要替换成实际值：

```bash
scp /tmp/vllm-ascend-fragpack.bundle USER@BUILD_HOST:/data/
```

构建机使用与测试节点相同的 CPU 架构，并能访问 CANN 镜像仓库和依赖下载源：

```bash
git clone -b codex/mla-leader-fanout \
  /data/vllm-ascend-fragpack.bundle /data/vllm-ascend-fragpack
cd /data/vllm-ascend-fragpack
git submodule update --init --recursive
```

Mac 用于修改和传源码。优先在 Linux 构建机或测试节点构建，避免引入跨架构模拟编译。

## 2. 构建匹配版本的完整镜像

不要将当前 main 的两个 Python 文件直接覆盖到压缩包里的旧定制镜像。
那样会混用 vLLM API、Ascend 源码和编译扩展，无法判断失败来自版本差异还是本次优化。

当前 checkout 的 vLLM 配套提交记录在 `.github/vllm-main-verified.commit`。
在构建机执行：

```bash
cd /data/vllm-ascend-fragpack
ascend_revision=$(git rev-parse --short=12 HEAD)
vllm_revision=$(tr -d '[:space:]' < .github/vllm-main-verified.commit)
fragpack_image="vllm-ascend:mla-fanout-${ascend_revision}"

docker build \
  -f Dockerfile \
  --build-arg VLLM_COMMIT="$vllm_revision" \
  --build-arg SOC_VERSION=ascend910b1 \
  --label "org.opencontainers.image.revision=$(git rev-parse HEAD)" \
  --label "org.vllm.upstream.revision=$vllm_revision" \
  -t "$fragpack_image" .
```

上面对应 Ascend 910B / A2 Ubuntu。A3 使用 `Dockerfile.a3` 和
`SOC_VERSION=ascend910_9391`；A5 使用其对应 Dockerfile 和实际 SoC。
当前 A2 Dockerfile 默认 CANN 9.1.0，Mooncake wheel 默认 0.3.11.post1。
驱动/CANN、通信后端以及模型启动参数应在基线启动阶段先确认兼容。
P 和 D 建议使用同一个新镜像，保持 vLLM、Ascend 和 Mooncake 版本一致。

旧部署若必须保留定制 CANN/HIXL/Mooncake 组合，应先把优化移植到旧镜像的
**确切源码基线**，再基于旧镜像构建派生镜像；不能用当前整份 connector 文件替代移植。

## 3. 分发镜像并确认加载了修改

可推送到团队私有 registry，或离线保存和加载：

```bash
# 构建机
archive_path="/data/mla-fanout-${ascend_revision}.tar"
docker save -o "$archive_path" "$fragpack_image"
sha256sum "$archive_path"
```

将归档传到各 P/D 测试节点，核对校验和后执行：

```bash
docker load -i /data/mla-fanout-COMMIT.tar
```

复用测试环境已有的设备、驱动、模型和网络挂载。先在模型尚未启动的测试容器中验证：

```bash
npu-smi info
python3 - <<'PY'
import inspect
import torch
from vllm_ascend.distributed.kv_transfer.kv_p2p.mla_fanout import MLAFanout
from vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector import KVCacheRecvingThread
print('NPU:', torch.npu.is_available())
print('fanout module:', inspect.getfile(MLAFanout))
print('connector:', inspect.getfile(KVCacheRecvingThread))
print('dispatch:', inspect.getsource(KVCacheRecvingThread._transfer_kv_cache_all_groups))
PY
```

在具备 UT 依赖的容器中运行：

```bash
cd /vllm-workspace/vllm-ascend
python3 -m pytest -sv \
  tests/ut/kv_offload/test_mla_fanout.py \
  tests/ut/kv_offload/test_mooncake_hybrid_connector.py
```

需要依赖时，在独立验证容器里安装 `requirements-dev.txt`，随后再使用原始镜像启动模型。

## 4. 复用现有 PD 启动脚本

先复制为测试专用启动脚本，使用独立测试容器名和 proxy 配置。
保持模型、量化、MTP、图模式、TP/DP、通信环境、batch 和 block 参数一致。
基线跑通后再切换优化开关。两种模式都使用同一个新镜像。

如果从 bundle 中的 `run_node_c2.sh` 改起：

1. 将 `image` 改为新镜像。
2. 删除覆盖 `mooncake_hybrid_connector.py` 的 shim bind mount。
3. 删除 `FRAGPACK_C2_MODE`、`FRAGPACK_LEADER_RANK`、`FRAGPACK_ORIG_PATH` 等原型环境变量。
4. 保留设备、驱动、模型、日志、NIC 等实际环境配置。
5. 若 D ranks 位于多个容器，给它们 bind mount **同一个宿主机本地目录**到同一路径。

例如，为 fanout 增加 Docker 挂载：

```bash
# D 宿主机创建目录
mkdir -p /data01/mla-fanout

# 加入 D 容器的 docker run 参数
# -v /data01/mla-fanout:/run/vllm-mla-fanout
```

同一容器内的 TP workers 可以直接使用上述目录。不同容器不能使用各自独立的 `/dev/shm`
充当共享目录，也不能放在 NFS 上。
所有 workers 应使用相同的宿主机 UID（包含容器用户映射）。ready 文件权限为 0644，
但锁、ACK 创建和目录清理还需要一致的目录所有权和写权限。

P/D 都选择 `MooncakeHybridConnector`。仅在 D 的
`kv_connector_extra_config` 中添加下面对象；现有 prefill/decode 拓扑配置保持不变：

```json
{
  "leader_fanout": {
    "shared_dir": "/run/vllm-mla-fanout",
    "session_id": "REPLACE-WITH-A-FRESH-UUID",
    "leader_rank": 0,
    "ready_timeout": 30,
    "ack_timeout": 60
  }
}
```

每次 D 引擎启动生成一次 UUID，写入该引擎的完整启动配置；该 TP 组所有 rank 使用同一值。
不同 DP 引擎使用不同值。Linux 可用 `cat /proc/sys/kernel/random/uuid` 生成。
基线模式完全删除 `leader_fanout` 对象。旧 `FRAGPACK_*` 环境变量不会启用原生实现。

所有 D TP ranks 必须在同一宿主机，并使用兼容的 monotonic 时钟命名空间。
PCP=1 与单节点是两个独立条件；共享挂载不能替代单节点要求。

每轮传输的 `transfer_attempt` 由 D scheduler 自动生成，并随 metadata 广播给所有 TP
workers，无需用户配置。重试必须创建新传输 metadata，不能重放旧 attempt。
ready/ACK 文件按 attempt 隔离；缺少 attempt 的旧 metadata 会回退到 P。

排查时关注 `MLA fanout pin blocked`：锁等待每 5 秒告警，KV 仍保持 pin，需检查
TE 或恢复故障 worker，不能靠超时强行释放正在被读取的 KV。
reader 等待锁和补发 ACK 受 ready_timeout 的剩余时间限制，超时回退；leader 清理 pin
不设置强制超时，以免释放活跃 reader 正在访问的 KV。
`MLAFanout.stats()` 提供每 worker 的 leader_pull、local_read、fallback、分原因 fallback、
ack_timeout 和 pin_wait_warnings 累计计数；fallback 日志带原因和计数快照。
这些计数未接入 Prometheus。成功事件可通过 DEBUG 日志观察。
轮询从 1ms 退避至 10ms，仍需在真机上确认尾延迟和 CPU 开销。

## 5. 正确性、性能和故障验证

先跑短请求，确认请求确实经 proxy 完成 P→D 交接。再用相同固定 prompts、seed、
`temperature=0` 和输出长度比较开关两种模式的输出，覆盖长上下文 needle、并发和 prefix hits。
性能比较初期关闭 prefix caching，避免重复输入命中缓存而跳过 KV 传输。
单独完成 prefix-hit 正确性用例后再打开缓存测试实际业务负载。

使用镜像内的 `vllm bench serve --help` 确认参数支持后，在能访问模型 tokenizer 和测试 proxy
的容器中运行下面的 128K / 并发 4 示例。请求必须发到 proxy，而非直接发到 D 的 API。

```bash
vllm bench serve \
  --backend openai \
  --base-url http://TEST_PROXY_IP:8000 \
  --endpoint /v1/completions \
  --model dsv4 \
  --tokenizer /model \
  --tokenizer-mode deepseek_v4 \
  --dataset-name random \
  --random-input-len 128000 \
  --random-output-len 256 \
  --num-prompts 16 \
  --max-concurrency 4 \
  --request-rate inf \
  --ignore-eos \
  --seed 42 \
  --save-result \
  --result-dir /dsv4/logs/fanout-bench \
  --result-filename off-128k-c4-r1.json
```

模式切换后用相同参数执行，结果文件名改为 `on-128k-c4-r1.json`。
正式矩阵：输入 32K / 128K / 200K，并发 1 / 2 / 4 / 8，输出固定 256 tokens。
每种模式先预热，再做至少三轮，按 off→on→off→on 交错运行。
确认 tokenizer/服务实际接受的 prompt token 数，不要只依赖重复文本估算输入长度。

记录 TTFT、TPOT、P95/P99、成功率、输出 token 数、fallback/ACK deadline 日志和跨节点
网卡字节数。需要通过 TransferEngine profiling 区分跨机 P→D 流量与机内 leader→reader 流量；
不能仅依据 TTFT 或混合网卡计数断言跨机流量减少到 1/8。理想的 TP8 正常路径应只有 leader
发起 P 读取，其他 rank 的 TE session 应指向 D leader。

故障测试在隔离测试实例上执行：

- 配置不可共享的 rendezvous 目录：reader 应超时并回退 P。
- Reader 机内 TE 读取失败：应 ACK 后回退 P。
- Reader 读取超过 ACK deadline：leader 不应在 active reader 返回前完成并释放 KV。
- 取消请求、并发 block reuse、worker crash/restart：验证无挂起、错误输出或 stale session。

不要把 kill TP worker 当作“服务必须继续运行”的测试；worker failure 可导致整个引擎退出。
重点检查是否正确失败、恢复，以及重启后是否读取旧状态。
运行中的 session 目录不能删除。永久挂起的 TE 调用需要 worker recovery。

验收应同时满足：正常路径输出一致、无异常 fallback、跨机重复读取确实下降、长输入并发
TTFT 改善且 TPOT 无明显回归。压缩包里的性能数字是旧原型结果，不是这个新镜像的保证。
