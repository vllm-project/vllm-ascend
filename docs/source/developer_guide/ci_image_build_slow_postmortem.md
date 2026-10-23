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
| 08:06 | 缓存查询 404 → 缓存未命中 | — | buildkitd 日志 `fetch failed 404 ...buildcache-...-arm64` |
| 08:11:12 | `Build and push` 开始，进入全量 C++ 重编译 | — | GitHub job 时间 |
| 08:11→09:12 | 编译被拖慢：6 个 arm64 构建抢 4 个 buildkitd pod（每 pod 32 核，`ninja -j32` 互相踩） | >1h（正常仅 ~28min） | 并发 job 时间表 + 4 pod |
| 09:12:29 | runner 断连，构建被 kill：`context canceled` | — | buildkitd 日志（已在 gy006 复现） |
| 09:41→10:48 | runner（buildx 客户端）断连后卡死，buildkitd 零活动 | ~1.5h | 各 pod 日志全静默 |
| 10:48:31 | GitHub 才记录 failure | — | GitHub job 时间 |

旁证：同一批同时跑的另外 5 个 arm64 构建（a3/310p/a5）都在 09:07~09:37 正常结束，
只有 a2-ubuntu 卡到 10:48 才失败——说明卡死是 a2-ubuntu 这条 runner/构建特有的，
不是集群整体故障。

## 三个根因

1. **runner 断连 + 卡死（主因，~1.5h）**

   09:12:29 buildkitd 日志出现 `Solve returned error: ... context canceled`，构建被取消；
   之后直到 10:48 才有 GitHub failure，中间 1.5h buildkitd 零活动。这是**自托管 runner
   （buildx 客户端）与 buildkitd 断连后没有及时退出/重试**导致的卡死，不是 buildkitd
   本身的问题。

2. **缓存未命中 → 全量 csrc 重编译（~28min）**

   cache 404 → `COMPILE_CUSTOM_KERNELS=1` 全量重编译 csrc。

3. **CPU 争抢**

   08:11 同时 6 个 arm64 构建 vs 4 个 pod，再加 `max-parallelism=2`，编译
   （`ninja -j32`）互相踩导致编译翻倍变慢。

## 「GC 卡顿」假设已被实测证伪

早期怀疑 buildkitd 的 `maxUsedSpace` GC 会在构建中途同步卡住 ~29min，在 gy006 实测
**不成立**：

- 写 512MB / 3GB 跨过 GC 阈值，构建均无任何卡顿（19s / 1m50s 完成）；
- GC 是**后台异步**触发的，构建结束后才把缓存从 253GB 收回到 249GB。

故「GC 卡顿」不列入根因。

## gy006 实测验证（证据）

gy006 是生产的同构测试集群（同为 buildkit v0.33.0-ci、`max-parallelism=2`、GPSSD 盘、
`log.level=info`，只是副本数更少、GC 阈值 250GB）。

| 验证项 | 结果 | 说明 |
|---|---|---|
| GC 是否阻塞构建 | 否（异步） | 跨阈值构建无卡顿，GC 后台回收 |
| GPSSD 磁盘慢 | 是 | 3GB 层导出 84.2s |
| 客户端断连 → context canceled | 复现 | 与生产 09:12:29 日志一致 |

断连复现日志（gy006，强杀 buildctl 客户端后）：

```text
... failed to close tar writer: context canceled ... fresh.bin: context canceled
Solve returned error: rpc error: code = Canceled desc = ... context canceled
```

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

# 生成 512MB 不可压缩文件 + scratch Dockerfile
kubectl --kubeconfig $KC -n buildkitd exec $POD -- sh -c '
  mkdir -p /tmp/repro && cd /tmp/repro
  dd if=/dev/urandom of=big.bin bs=1M count=512 status=none
  printf "FROM scratch\nCOPY big.bin /big.bin\n" > Dockerfile'

# 计时构建（观察跨过 250GB GC 阈值后是否卡顿）
kubectl --kubeconfig $KC -n buildkitd exec $POD -- sh -c '
  df -h /home/user/.local/share/buildkit | tail -1
  time buildctl --addr unix:///run/buildkit/buildkitd.sock build \
    --frontend dockerfile.v0 --local context=/tmp/repro --local dockerfile=/tmp/repro \
    --output type=image,name=repro-gc:latest'
```

结论：构建不卡顿（实测 19s），数分钟后 `df` 会看到 GC 在后台把缓存收回到阈值之下。

### 复现 2：验证客户端断连 → context canceled（对应生产 09:12:29）

```bash
KC=/home/wuhejun/.kube/config/gy006.kubeconfig
POD=buildkitd-amd64-deployment-0

# 用一个较大的文件让层导出够慢，以便中途 kill
kubectl --kubeconfig $KC -n buildkitd exec $POD -- sh -c '
  cd /tmp/repro
  dd if=/dev/urandom of=fresh.bin bs=1M count=2048 status=none
  printf "FROM scratch\nCOPY fresh.bin /fresh.bin\n" > Dockerfile
  buildctl --addr unix:///run/buildkit/buildkitd.sock build \
    --frontend dockerfile.v0 --local context=/tmp/repro --local dockerfile=/tmp/repro \
    --output type=image,name=repro-kill:latest > /tmp/kill.log 2>&1 &
  BPID=$!; sleep 18; kill -9 $BPID'

# 观察 buildkitd 日志
kubectl --kubeconfig $KC -n buildkitd logs $POD --since=3m | grep -i 'context canceled'
```

结论：出现 `Solve returned error: ... context canceled`，与生产日志一致。

### 复现 3：验证 GPSSD 磁盘慢

```bash
kubectl --kubeconfig=/home/wuhejun/.kube/config/gy006.kubeconfig -n buildkitd \
  exec buildkitd-amd64-deployment-0 -- sh -c \
  'time du -ah /home/user/.local/share/buildkit/runc-overlayfs >/dev/null'
```