# test-case-006: squid 代理下载测试（GitHub / Gitee）

## 背景

- Issue: https://github.com/opensourceways/ascend-ci-deployment/issues/945（新增 buildkit runner 测试 squid 代理）
- Runner 部署 PR: https://github.com/opensourceways/ascend-ci-deployment/pull/946
- 集群: openmerlin-guiyang-006 (gy006)
- Runner: `linux-aarch64-cpu-2-buildkit-gy006`

在给这个 runner 的 job pod template 配置 buildctl 自动安装脚本时，发现走 squid 代理
（`http://squid-cache.squid.svc.cluster.local:3128`）下载 GitHub Releases 资源
（`github.com/moby/buildkit/releases/download/...`）不稳定，多次测试出现三种不同的失败：

| 测试 | 走代理 | 信任 squid CA | 结果 |
|---|---|---|---|
| 真实 postStart 脚本 | 是 | 是 | `curl: (47) Maximum (50) redirects followed` |
| 手工复测 1 | 是 | 否 | `curl: (60) SSL certificate problem: self-signed certificate` |
| 手工复测 2 | 是 | 是 | `curl: (28) Operation timed out after 30001 milliseconds` |
| 手工复测 3 | 否（直连） | — | 成功 |

**已验证的结论**：不经过 squid 代理时下载稳定成功；经过 squid 代理时会以不同方式失败
（不是单一、可稳定复现的故障模式）。**未验证/不确定的部分**：squid 侧具体是什么原因导致
（需要 squid 自己的 access log / cache log 才能定位，本测试无法从客户端单独确认）。

本目录提供可重复执行的测试用例，用于：
1. 验证/复现上述现象
2. 同时覆盖 GitHub 和 Gitee 两个下载源，确认问题范围（是否只影响 GitHub，还是所有 HTTPS 出站都受影响）
3. 便于后续排查 squid 问题时快速复测，或提供给 squid 维护方作为复现步骤

## 测试矩阵

`test-download.sh` 会对每个下载源分别测试「走代理」和「不走代理」两种情况：

| 下载源 | URL | 说明 |
|---|---|---|---|
| GitHub Releases | `https://github.com/moby/buildkit/releases/download/v0.29.0/buildkit-v0.29.0.linux-arm64.tar.gz` | 直连速度极慢（~20KB/s），走 squid 代理行为已修复（之前有 URL 重写 bug） |
| gh-proxy 镜像 | `https://gh-proxy.test.osinfra.cn/https://github.com/moby/buildkit/releases/download/v0.29.0/buildkit-v0.29.0.linux-arm64.tar.gz` | smart-git-proxy 部署在 gy006 集群，HTTP 层反向代理 GitHub；85MB 下载 1-3s（27-81MB/s） |
| Gitee Releases | `https://gitee.com/mirrors/buildkit/releases/download/v0.29.0/buildkit-v0.29.0.linux-arm64.tar.gz`（如无该 release，退化为 `https://gitee.com` 首页连通性测试） | 用于对比，确认问题是否只存在于 GitHub 域名 |

## 已验证结果（2026-07-09 gy006 集群）

| 测试 | 结果 | 详情 |
|---|---|---|
| github-no-proxy | FAIL (28, 超时) | 60s 只下载 1MB/85MB，~18KB/s |
| github-via-proxy | FAIL (28, 超时) | squid URL 重写 bug **已修复**（Location 不再畸形），同样 60s 只下载 ~500KB |
| **gh-proxy-no-proxy** | **OK** | **85MB in 3s, 27MB/s** |
| **gh-proxy-via-proxy** | **OK** | **85MB in 1s, 81MB/s**（squid 缓存命中加速） |
| gitee-homepage-no-proxy | OK | 642KB in 2s |
| gitee-homepage-via-proxy | OK | 642KB in 2s

## 如何运行

### 方式一：在 gy006 集群里起一个临时 debug pod（推荐，贴近真实 runner 环境）

```bash
export KUBECONFIG=~/.kube/gy-006.yaml
kubectl apply -f test-case-006/debug-pod.yaml
kubectl wait --for=condition=Ready pod/test-case-006-debug -n ascend-gha-runners --timeout=60s
kubectl cp test-case-006/test-download.sh ascend-gha-runners/test-case-006-debug:/tmp/test-download.sh
kubectl exec -n ascend-gha-runners test-case-006-debug -- sh /tmp/test-download.sh
kubectl delete -f test-case-006/debug-pod.yaml
```

### 方式二：通过 GitHub Actions workflow 在真实 runner 上跑

见 `.github/workflows/test-case-006-download.yaml`（使用 `linux-aarch64-cpu-2-buildkit-gy006` runner，
带 `container:` 块以确保走完整的 job pod template，包括 squid CA 注入）。

```bash
gh workflow run test-case-006-download.yaml --repo ascend-gha-runners/vllm-ascend --ref main
```

## 文件说明

- `test-download.sh` — 核心测试脚本，对 GitHub / Gitee 分别测试有/无代理两种场景，输出每次测试的
  curl exit code 和关键 verbose 日志（TLS 握手阶段、HTTP 状态码、Location 重定向链）
- `debug-pod.yaml` — 独立调试用 Pod 定义，挂载 squid-ca-cert ConfigMap，运行在 gy006 arm64 节点
- `.github/workflows/test-case-006-download.yaml` — 在真实 buildkit runner 上跑同样的测试

## 已知限制

- 代理侧的失败模式不稳定，同一测试可能这次超时、下次重定向循环、下次证书错误，属于预期内的
  "flaky" 现象，**不代表脚本有 bug**，而是反映 squid 代理本身的行为不稳定
- 本测试无法访问 squid pod 自身的日志，因此无法给出 squid 端的根因，只能证明"经过代理 vs
  不经过代理"这个对照结果
