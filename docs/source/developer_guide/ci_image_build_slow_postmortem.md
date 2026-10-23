# CI 镜像构建耗时 2h37m 根因分析与复现指南

> 对象：`Nightly Image Build (with base)` run 39560893083 中 `a2-ubuntu arm64`
> 基础镜像构建（`Build base image (a2-ubuntu) / build (linux-aarch64-cpu-4, arm64)`）。
>
> 结论：2h37m 不是机器正常所需的构建时间，而是「缓存未命中导致全量重编译」+
> 「buildkitd 磁盘缓存顶到 GC 阈值触发卡顿」+「6 构建抢 4 pod 的 CPU 争抢」+
> 「构建被取消后 runner 卡死约 1.5h 才报错」四段叠加的结果。

## 一句话结论

四段叠加里，最后一档（runner 卡死 ~1.5h）是把它从「本可 ~1h 完成」拖到 2h37m
的直接元凶；前三档是让它变慢的根因。

## 根因拆解（对到时间）

| 时间段(UTC) | 发生了什么 | 耗时 | 证据 |
|---|---|---|---|
| 08:06 | 缓存查询 404 → 缓存未命中 | — | buildkitd 日志 `fetch failed 404 ...buildcache-...-arm64` |
| 08:11:12 | `Build and push` 开始，进入全量 C++ 重编译 | — | GitHub job 时间 |
| 08:11→09:12 | 编译被拖慢：6 个 arm64 构建抢 4 个 buildkitd pod（每 pod 32 核，`ninja -j32` 互相踩） | >1h（正常仅 ~28min） | 并发 job 时间表 + 4 pod |
| 09:12:29 | 构建被 kill：`container ... taking a long time to exit after kill` + `context canceled` | — | buildkitd 日志 |
| 09:41→10:48 | runner（buildx 客户端）断连后卡死，buildkitd 零活动 | ~1.5h | 各 pod 日志全静默 |
| 10:48:31 | GitHub 才记录 failure | — | GitHub job 时间 |

旁证：同一批同时跑的另外 5 个 arm64 构建（a3/310p/a5）都在 09:07~09:37 正常结束，
只有 a2-ubuntu 卡到 10:48 才失败——说明卡死是 a2-ubuntu 这条 runner/构建特有的，
不是集群整体故障。

## 三个已确认的集群级根因

1. **磁盘缓存顶阈值触发 GC**

   ```text
   buildkitd.toml:   maxUsedSpace = "600GB"      # GC 回收阈值
   实测:             598.1 GB used (59%)          # 紧贴阈值
   存储:             csi-disk-topology, GPSSD 通用型云盘
   ```

   编译写海量缓存把磁盘顶过 600GB → buildkitd 中途触发 GC。实测 `du` 遍历 600GB
   缓存目录超时 >120s，证明这块盘上 GC/元数据操作极慢（成功重跑里就有一个 29min
   的 GC 空档）。

2. **CPU 争抢**：08:11 同时 6 个 arm64 构建 vs 4 个 pod。

3. **缓存未命中**：cache 404 → `COMPILE_CUSTOM_KERNELS=1` 全量重编译 csrc。

## 解决方案（防止再次发生）

1. 上调 GC 阈值 / 扩容磁盘：`maxUsedSpace` 600GB→900GB，或 PVC 扩到 2Ti，让构建期间不再触发 GC。（最直接）
2. GPSSD 换 ESSD/SSD：同时消除「GC 慢」和「编译 IO 慢」。
3. 让 csrc 编译命中缓存：`9a7874bb9`（"parallelize and cache csrc compilation"）已加入并行化+缓存。
4. 限制并发 / 增加 buildkitd pod：避免 6 抢 4。
5. 给 job 加超时：`Build and push` 加 `timeout-minutes`，避免 runner 卡死再拖 1.5h 才失败。

## 复现方法（闲时执行）

### 目标 A：验证「磁盘顶阈值 → GC 卡顿」（推荐，非破坏性）

```bash
# 1. 先确认缓存确实贴阈值
kubectl --kubeconfig=<KUBECONFIG> -n buildkitd \
  exec buildkitd-arm64-deployment-0 -- df -h /home/user/.local/share/buildkit

# 2.（可选，建议）临时开 debug 日志，才能看到 GC 事件（闲时做）
kubectl --kubeconfig=<KUBECONFIG> -n buildkitd \
  edit cm buildkitd-arm64-config          # log.level = "info" 改成 "debug"
kubectl --kubeconfig=<KUBECONFIG> -n buildkitd \
  rollout restart statefulset/buildkitd-arm64-deployment

# 3. 在 pod 内构造一个写 ~3GB 的构建上下文
kubectl --kubeconfig=<KUBECONFIG> -n buildkitd \
  exec buildkitd-arm64-deployment-0 -- sh -c '
    mkdir -p /tmp/repro && cd /tmp/repro
    printf "FROM busybox\nRUN dd if=/dev/zero of=/big bs=1M count=3072\n" > Dockerfile'

# 4. 通过 unix socket 构建并计时（观察是否在跨过 600GB 后卡住几分钟）
kubectl --kubeconfig=<KUBECONFIG> -n buildkitd \
  exec buildkitd-arm64-deployment-0 -- sh -c '
    time buildctl --addr unix:///run/buildkit/buildkitd.sock build \
      --frontend dockerfile.v0 --local context=/tmp/repro --local dockerfile=/tmp/repro \
      --output type=image,name=repro-gc-test:latest'

# 5. 另开一个窗口，观察 GC 日志（若已开 debug）
kubectl --kubeconfig=<KUBECONFIG> -n buildkitd \
  logs -f buildkitd-arm64-deployment-0 | grep -iE 'gc|releas|reclaim'
```

判定：若构建在写完层、磁盘跨过 600GB 后出现明显卡顿（分钟级），且 debug 日志出现
`gc ... released`，即坐实「GC 卡顿」根因。

### 目标 B：验证 GPSSD 磁盘慢（秒级，零副作用）

```bash
kubectl --kubeconfig=<KUBECONFIG> -n buildkitd \
  exec buildkitd-arm64-deployment-0 -- sh -c \
  'time du -ah /home/user/.local/share/buildkit/runc-overlayfs >/dev/null'
# 预期：600GB 目录遍历明显偏慢（此前实测 >120s 超时）
```

### 目标 C：完整复现 2h37m 那次（最重，非必要）

直接在工作流里 `workflow_dispatch` 触发同样的 `a2 + ubuntu + arm64` 构建，并叠加 6
并发。耗时 ~1h+，只在需要给外部团队演示完整链路时做。
