# CI 镜像构建耗时 2h37m 根因分析与复现指南

> 对象：`Nightly Image Build (with base)` run 39560893083 中 `a2-ubuntu arm64`
> 基础镜像构建（`Build base image (a2-ubuntu) / build (linux-aarch64-cpu-4, arm64)`）。
>
> 结论：2h37m 不是机器正常所需的构建时间，而是「缓存未命中导致全量重编译」+
> 「6 构建抢 4 pod 的 CPU 争抢」+「runner 与 buildkitd 断连后卡死约 1.5h 才报错」
> 三段叠加的结果。

## 一句话结论

三段里，最后一档（runner 断连后卡死 ~1.5h）是把它从「本可 ~1h 完成」拖到 2h37m
的直接元凶；前两档是让它变慢的根因。

## 根因拆解（对到时间）

| 时间段(UTC) | 发生了什么 | 耗时 | 证据 |
|---|---|---|---|
| 08:06 | 缓存查询 404 → 缓存未命中 | — | E1 |
| 08:11:12 | `Build and push` 开始，进入全量 C++ 重编译 | — | E2 |
| 08:11→09:12 | 编译被拖慢：6 个 arm64 构建抢 4 个 buildkitd pod（每 pod 32 核，`ninja -j32` 互相踩） | >1h（正常仅 ~28min） | E3 |
| 09:12:29 | runner 断连，构建被 kill：`context canceled` | — | E4 |
| 09:41→10:48 | runner（buildx 客户端）断连后卡死，buildkitd 零活动 | ~1.5h | E5 |
| 10:48:31 | GitHub 才记录 failure | — | E2 |

## 证据明细

### E1 — 缓存未命中（cache 404）

cn12001 集群 pod `buildkitd-arm64-deployment-0` 日志：

```text
time="2026-10-21T08:06:05Z" level=info msg="fetch failed after status: 404 NOT FOUND" host=quay.io method=HEAD url="https://quay.io/v2/wjunlu/vllm-ascend-workflow/manifests/buildcache-ci-unify-image-deps-pinning-arm64"
```

### E2 — 失败 job 的时间戳与结果

GitHub job `119111996225`（`a2-ubuntu arm64`，run_attempt=2）：

```text
started_at   : 2026-10-21T08:09:42Z
conclusion   : cancelled          (completed_at: 10:52:44Z)

step "Set up job"    : 08:09:46Z → 08:10:01Z  (success)
step "Build and push": 08:11:12Z → 10:48:31Z  (failure)
```

### E3 — 08:10 前后同时有 6 个 arm64 构建在跑，抢 4 个 pod

arm64 构建均跑在 4 个 `buildkitd-arm64-deployment-0..3` pod 上：

| job | 开始 | 结束 | 结果 |
|---|---|---|---|
| a3-ubuntu arm64 | 08:09:52 | 09:18:14 | success |
| a3-openeuler arm64 | 08:09:57 | 09:20:37 | success |
| 310p-openeuler arm64 | 08:09:58 | 09:07:04 | success |
| a5-ubuntu arm64 | 08:09:59 | 09:32:43 | success |
| a5-openeuler arm64 | 08:10:01 | 09:37:43 | success |
| **a2-ubuntu arm64（失败）** | 08:09:42 | 10:52:44 | cancelled |

6 个构建 vs 4 个 pod，其中 2 个 pod 要同时接 2 个编译任务。

buildkitd 配置 `max-parallelism=2`（每 pod 全局最多 2 个并行 step），编译步骤内部
`ninja -j$(nproc)=32` 线程，叠起来就是 64 线程抢 32 核。

### E4 — runner 断连（context canceled）

cn12001 集群 pod `buildkitd-arm64-deployment-0` 日志，09:12:29：

```text
time="2026-10-21T09:12:29Z" level=warning msg="container id w6hgxxbhc4d12n0apxhyl014k is taking a long time to exit after kill" span="[ 8/11] RUN ... pip install -e vllm-ascend ..."
time="2026-10-21T09:12:29Z" level=error msg="/moby.buildkit.v1.Control/Solve returned error: rpc error: code = Canceled desc = context canceled"
```

gy006 测试集群复现（强杀 buildctl 客户端模拟断连）：

```text
... failed to close tar writer: context canceled ... fresh.bin: context canceled
level=error msg="/moby.buildkit.v1.Control/Solve returned error: rpc error: code = Canceled desc = ... context canceled"
```

两者日志一致，证明「断连 → context canceled → 容器被杀」这一机制成立。

### E5 — buildkitd 静默 1.5h（runner 卡死）

pod `buildkitd-arm64-deployment-0` 在 09:41 到 10:53 之间的日志量（按 10 分钟桶统计）：

```text
09:12  9 条   ← kill + context canceled（E4）
09:41  2 条   ← fetch 404
（09:41 之后到 10:53 之间：0 条）
10:53  3 条   ← 下一轮重试的 context canceled
```

即 09:12 构建被取消后，buildkitd 端「零活动」约 1.5h，直到 10:48:31 GitHub 才记
failure（E2），说明卡死发生在 **runner（buildx 客户端）侧**，不是 buildkitd。

旁证：同批另 5 个 arm64 构建都在 09:07~09:37 正常结束（E3），只有 a2-ubuntu 卡到
10:48 才失败——是这条 runner/构建特有的问题，不是集群整体故障。

## 「GC 卡顿」假设已被实测证伪

早期怀疑 buildkitd 的 `maxUsedSpace` GC 会在构建中途同步卡住 ~29min，在 gy006 实测
**不成立**：写 512MB / 3GB 跨过 GC 阈值，构建均无任何卡顿（19s / 1m50s 完成）；GC 是
**后台异步**触发的，构建结束后才把缓存从 253GB 收回到 249GB。故「GC 卡顿」不列入根因。

## gy006 实测验证（证据）

gy006 是生产的同构测试集群（同为 buildkit v0.33.0-ci、`max-parallelism=2`、GPSSD 盘、
`log.level=info`，只是副本数更少、GC 阈值 250GB）。

| 验证项 | 结果 | 说明 |
|---|---|---|
| GC 是否阻塞构建 | 否（异步） | 跨阈值构建无卡顿，GC 后台回收 |
| GPSSD 磁盘慢 | 是 | 3GB 层导出 84.2s |
| 客户端断连 → context canceled | 复现 | 与生产 09:12:29 日志一致（见 E4） |

## 解决方案（防止再次发生）

1. **给 image build job 加 `timeout-minutes`**：直接解决 runner 卡死拖 1.5h 的问题（最直接）。
2. **让 csrc 编译命中缓存**：`9a7874bb9`（"parallelize and cache csrc compilation"）已加入，消掉 ~28min 编译。
3. **限制并发 / 扩容 buildkitd pod（或用 dispatcher 动态扩缩）**：避免 6 抢 4 的 CPU 争抢。
4. **GPSSD 换 ESSD/SSD**：降低层导出/编译 IO 耗时（3GB 层导出 84s 属偏慢）。
5. **排查自托管 runner 与 buildkitd 的连接稳定性**：定位为何断连、断连后为何不重试/退出。

## 复现方法（gy006 已实测可行）

测试集群外网受限（docker.io/quay.io 走 squid 均超时），所以统一用 `FROM scratch + COPY`
本地文件，无需拉取基础镜像。

### 复现 1：验证 GC 是否阻塞构建

```bash
KC=/home/wuhejun/.kube/config/gy006.kubeconfig
POD=buildkitd-amd64-deployment-0

kubectl --kubeconfig $KC -n buildkitd exec $POD -- sh -c '
  mkdir -p /tmp/repro && cd /tmp/repro
  dd if=/dev/urandom of=big.bin bs=1M count=512 status=none
  printf "FROM scratch\nCOPY big.bin /big.bin\n" > Dockerfile'

kubectl --kubeconfig $KC -n buildkitd exec $POD -- sh -c '
  df -h /home/user/.local/share/buildkit | tail -1
  time buildctl --addr unix:///run/buildkit/buildkitd.sock build \
    --frontend dockerfile.v0 --local context=/tmp/repro --local dockerfile=/tmp/repro \
    --output type=image,name=repro-gc:latest'
```

结论：构建不卡顿（实测 19s），数分钟后 GC 在后台把缓存收回到阈值之下。

### 复现 2：验证客户端断连 → context canceled（对应生产 09:12:29）

```bash
kubectl --kubeconfig=/home/wuhejun/.kube/config/gy006.kubeconfig -n buildkitd \
  exec buildkitd-amd64-deployment-0 -- sh -c '
  cd /tmp/repro
  dd if=/dev/urandom of=fresh.bin bs=1M count=2048 status=none
  printf "FROM scratch\nCOPY fresh.bin /fresh.bin\n" > Dockerfile
  buildctl --addr unix:///run/buildkit/buildkitd.sock build \
    --frontend dockerfile.v0 --local context=/tmp/repro --local dockerfile=/tmp/repro \
    --output type=image,name=repro-kill:latest > /tmp/kill.log 2>&1 &
  BPID=$!; sleep 18; kill -9 $BPID'

kubectl --kubeconfig=/home/wuhejun/.kube/config/gy006.kubeconfig -n buildkitd \
  logs buildkitd-amd64-deployment-0 --since=3m | grep -i 'context canceled'
```

结论：出现 `Solve returned error: ... context canceled`，与生产日志一致（E4）。

### 复现 3：验证 GPSSD 磁盘慢

```bash
kubectl --kubeconfig=/home/wuhejun/.kube/config/gy006.kubeconfig -n buildkitd \
  exec buildkitd-amd64-deployment-0 -- sh -c \
  'time du -ah /home/user/.local/share/buildkit/runc-overlayfs >/dev/null'
```