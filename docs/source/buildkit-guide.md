# Buildkit CI User Guide

## Overview

The buildkit infrastructure provides **native** Docker image building for both `arm64` and `amd64` architectures. Each architecture runs on its own buildkitd daemon, so there is **no QEMU emulation** — builds are fast and native.

## Runners

Two self-hosted runners are available:

| Runner | Target Architecture | Description |
|---|---|---|
| `linux-amd64-cpu-4` | `amd64` (x86_64) | Native amd64 build |
| `linux-aarch64-cpu-4` | `arm64` (aarch64) | Native arm64 build |

### Build for both architectures

```yaml
jobs:
  build:
    strategy:
      matrix:
        runner_info:
          - {runner: linux-amd64-cpu-4, arch: amd64}
          - {runner: linux-aarch64-cpu-4, arch: arm64}
    runs-on: ${{ matrix.runner_info.runner }}
```

### Build for a single architecture

```yaml
jobs:
  build:
    runs-on: linux-amd64-cpu-4  # amd64 only
```

> **Important**: Do `NOT` use `--platform linux/amd64,linux/arm64` in `docker/build-push-action`. The runner architecture determines the target natively. Using `--platform` with a single runner unnecessarily triggers QEMU emulation.

## Dockerfile

The Dockerfiles are fully parameterized via `ARG` with sensible defaults. **You do not need to modify the Dockerfile.** The defaults work out of the box.

### Available ARGs

| ARG | Default | Description |
|---|---|---|
| `CANN_QUAY_URL` | `quay.io/ascend/cann` | CANN base image registry |
| `CANN_VERSION` | `9.1.0` | CANN version |
| `BASE_OS` | `ubuntu22.04` | Base OS |
| `PIP_INDEX_URL` | `https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple` | PyPI mirror |
| `MOONCAKE_INDEX_URL` | `https://mirrors.aliyun.com/pypi/web/simple` | Mooncake PyPI mirror |
| `PYTORCH_INDEX_URL` | `https://download.pytorch.org/whl/cpu/` | PyTorch CPU wheel index |
| `ASCEND_INDEX_URL` | `https://mirrors.huaweicloud.com/ascend/repos/pypi` | Ascend package index |
| `APTMIRROR` | `""` (empty) | Apt cache/proxy mirror |
| `GIT_PROXY` | `""` (empty) | Git proxy prefix |
| `PIP_TRUSTED_HOST` | `""` (empty) | Trusted pip hosts |
| `SOC_VERSION` | (per Dockerfile) | Ascend SOC version |
| `COMPILE_CUSTOM_KERNELS` | `1` | Enable custom kernel compilation |

### Override ARGs

Override via `build-args` in the workflow:

```yaml
- name: Build and push
  uses: docker/build-push-action@v7
  with:
    build-args: |
      CANN_QUAY_URL=swr.cn-north-12.myhuaweicloud.com/base_image/ascend-ci/cann
      PYTORCH_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu
      PIP_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple
```

> The `PYTORCH_INDEX_URL` ARG is used as `--extra-index-url`. The internal cache service (`cache-service.../whl/cpu`) whitelists PyTorch packages (`torch`, `torchvision`, `torchaudio`) and returns an instant empty index page for all other packages, so pip never hits the slow international `download.pytorch.org` for non-PyTorch dependencies. With the default value (`https://download.pytorch.org/whl/cpu/`), it behaves exactly as before.

## Authentication

Use `docker/login-action` to authenticate with the registry:

```yaml
- name: Login to SWR
  uses: docker/login-action@v3
  with:
    registry: swr.cn-north-12.myhuaweicloud.com
    username: ${{ secrets.SWR_USERNAME }}
    password: ${{ secrets.SWR_PASSWORD }}
```

> Configure `SWR_USERNAME` and `SWR_PASSWORD` as repository secrets in **Settings → Secrets and variables → Actions**.

## Complete Workflow Example

### Build + push for both architectures

```yaml
name: docker-build

on:
  pull_request:
    paths:
      - 'Dockerfile'
  workflow_dispatch:

jobs:
  build:
    name: "build (${{ matrix.runner_info.arch }})"
    runs-on: ${{ matrix.runner_info.runner }}
    strategy:
      fail-fast: false
      matrix:
        runner_info:
          - {runner: linux-amd64-cpu-4, arch: amd64}
          - {runner: linux-aarch64-cpu-4, arch: arm64}

    steps:
      - uses: actions/checkout@v7

      - name: Login to SWR
        uses: docker/login-action@v3
        with:
          registry: swr.cn-north-12.myhuaweicloud.com
          username: ${{ secrets.SWR_USERNAME }}
          password: ${{ secrets.SWR_PASSWORD }}

      - name: Build and push
        uses: docker/build-push-action@v7
        with:
          context: .
          file: Dockerfile
          push: true
          tags: swr.cn-north-12.myhuaweicloud.com/modelfoundry/my-image:${{ matrix.runner_info.arch }}-${{ github.sha }}
          build-args: |
            CANN_QUAY_URL=swr.cn-north-12.myhuaweicloud.com/base_image/ascend-ci/cann
            APTMIRROR=http://cache-service.nginx-pypi-cache.svc.cluster.local:8081
            PIP_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple
            PIP_TRUSTED_HOST=cache-service.nginx-pypi-cache.svc.cluster.local
            PYTORCH_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/whl/cpu
            ASCEND_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/ascend/repos/pypi
            GIT_PROXY=https://gh-proxy.test.osinfra.cn/
          provenance: false
```

### Build only for amd64

```yaml
jobs:
  build:
    runs-on: linux-amd64-cpu-4
    # ... same steps as above
```

### Build only for arm64

```yaml
jobs:
  build:
    runs-on: linux-aarch64-cpu-4
    # ... same steps as above
```

## Key Rules

1. **Do not use `--platform`** — the runner architecture is the target architecture. Use the matrix to select both architectures.
2. **Do not modify the Dockerfile** — the defaults work for internal builds. Only override `build-args` if needed.
3. **Always set `provenance: false`** — to avoid metadata push issues.
4. **Always set `fail-fast: false`** — when building a matrix, so one architecture failure doesn't cancel the other.