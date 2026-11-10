# Dependency Version Management

This page lists the pinned dependency stack of vLLM Ascend that changes most
often: which files to update when a version changes, and where each component
is installed from. The versions shown are example values of one known-good
stack.

## Pinned dependency stack

| Component | Example version | Files to update | Installed from |
|---|---|---|---|
| torch | 2.10.0 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | PyTorch CPU index <https://download.pytorch.org/whl/cpu/> (images), PyPI <https://pypi.org/simple/> (CI) |
| torch-npu | 2.10.0.post7 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| triton-ascend | 3.6.0.post2 | `requirements.txt`<br>`pyproject.toml` `[build-system]`<br>`.github/workflows/_selected_tests.yaml`<br>`.github/workflows/schedule_main2main.yaml` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| memfabric-hybrid / memcache-hybrid | 1.3.0 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| flash-linear-attention-npu (FLA) | 26.9.1; dev wheel 26.9.1+deva4a7958 | Image build —<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>PR —<br>`.github/workflows/_selected_tests.yaml`<br>main2main —<br>`.github/workflows/schedule_main2main.yaml` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi>; the dev wheel is not on the mirror — main2main fetches it from the URL pinned in the workflow |
| flash_attn_npu | 0.5.0 | Image build —<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler` | PyPI <https://pypi.org/simple/> (sdist only) |
| mooncake-transfer-engine-npu | 0.3.14rc3 | Image build —<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>PR / main2main —<br>`.github/workflows/_selected_tests.yaml`<br>`.github/workflows/schedule_main2main.yaml` | Ascend mirror (nightly) <https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly> only; `manylinux_2_35` wheels, never in `requirements.txt` |
| cannbotdsl / cannbot-arena-net-ops | 0.6.0+g91e7c8f / 0.1.0 | Image build —<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi>, aarch64 only |
| CANN | 9.2.1 | Image build — `FROM` tag of:<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.310p`<br>`Dockerfile.310p.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>PR —<br>`mkdocs.yml` | `quay.io/ascend/cann` |

By default, update all files listed for a component.

## Example: changing a version

Every version pin each file contains, so a version change is one pass per
file. The `-` lines use the example versions from the table; replace
`<new-version>` with the target version.

### `requirements.txt`

```diff
-torch==2.10.0
-torch-npu==2.10.0.post7
-triton-ascend===3.6.0.post2
-memfabric-hybrid===1.3.0
-memcache-hybrid===1.3.0
+torch==<new-version>
+torch-npu==<new-version>
+triton-ascend===<new-version>
+memfabric-hybrid===<new-version>
+memcache-hybrid===<new-version>
```

### `pyproject.toml` `[build-system]`

```diff
-    "torch-npu==2.10.0.post7",
-    "torch==2.10.0",
-    "memfabric-hybrid===1.3.0",
-    "memcache-hybrid===1.3.0",
-    "triton-ascend===3.6.0.post2"
+    "torch-npu==<new-version>",
+    "torch==<new-version>",
+    "memfabric-hybrid===<new-version>",
+    "memcache-hybrid===<new-version>",
+    "triton-ascend===<new-version>"
```

### `Dockerfile`, `Dockerfile.openEuler`, `Dockerfile.a3`, `Dockerfile.a3.openEuler`

```diff
-FROM quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12
-FROM quay.io/ascend/cann:9.2.1-910b-openeuler24.03-py3.12
-FROM quay.io/ascend/cann:9.2.1-a3-ubuntu22.04-py3.12
-FROM quay.io/ascend/cann:9.2.1-a3-openeuler24.03-py3.12
+FROM quay.io/ascend/cann:<new-version>-910b-ubuntu22.04-py3.12
+FROM quay.io/ascend/cann:<new-version>-910b-openeuler24.03-py3.12
+FROM quay.io/ascend/cann:<new-version>-a3-ubuntu22.04-py3.12
+FROM quay.io/ascend/cann:<new-version>-a3-openeuler24.03-py3.12
...
-ARG FLA_VERSION="26.9.1"
-ARG FLASH_ATTN_NPU_VERSION="0.5.0"
-ARG MOONCAKE_VERSION="0.3.14rc3"
+ARG FLA_VERSION="<new-version>"
+ARG FLASH_ATTN_NPU_VERSION="<new-version>"
+ARG MOONCAKE_VERSION="<new-version>"
```

### `Dockerfile.a5`, `Dockerfile.a5.openEuler`

```diff
-FROM quay.io/ascend/cann:9.2.1-950-ubuntu22.04-py3.12
-FROM quay.io/ascend/cann:9.2.1-950-openeuler24.03-py3.12
+FROM quay.io/ascend/cann:<new-version>-950-ubuntu22.04-py3.12
+FROM quay.io/ascend/cann:<new-version>-950-openeuler24.03-py3.12
...
-ARG FLA_VERSION="26.9.1"
+ARG FLA_VERSION="<new-version>"
...
-python3 -m pip install "mooncake-transfer-engine-npu===0.3.14rc3" && \
+python3 -m pip install "mooncake-transfer-engine-npu===<new-version>" && \
...
-python3 -m pip install "cannbotdsl===0.6.0+g91e7c8f" "cannbot-arena-net-ops===0.1.0" && \
+python3 -m pip install "cannbotdsl===<new-version>" "cannbot-arena-net-ops===<new-version>" && \
```

### `Dockerfile.310p`, `Dockerfile.310p.openEuler`

```diff
-FROM quay.io/ascend/cann:9.2.1-310p-ubuntu22.04-py3.12
-FROM quay.io/ascend/cann:9.2.1-310p-openeuler24.03-py3.12
+FROM quay.io/ascend/cann:<new-version>-310p-ubuntu22.04-py3.12
+FROM quay.io/ascend/cann:<new-version>-310p-openeuler24.03-py3.12
```

### `.github/workflows/_selected_tests.yaml`

```diff
-uv pip install 'mooncake-transfer-engine-npu===0.3.14rc3'
-uv pip install --force-reinstall --no-deps 'triton-ascend===3.6.0.post2'
-uv pip install --no-deps "flash-linear-attention-npu-${NPU_TYPE}===26.9.1"
-uv pip install --no-deps flash-linear-attention-npu-a3===26.9.1
+uv pip install 'mooncake-transfer-engine-npu===<new-version>'
+uv pip install --force-reinstall --no-deps 'triton-ascend===<new-version>'
+uv pip install --no-deps "flash-linear-attention-npu-${NPU_TYPE}===<new-version>"
+uv pip install --no-deps flash-linear-attention-npu-a3===<new-version>
```

### `.github/workflows/schedule_main2main.yaml`

```diff
-uv pip install 'mooncake-transfer-engine-npu===0.3.14rc3'
-uv pip install --force-reinstall --no-deps 'triton-ascend===3.6.0.post2'
-FLA_VERSION=26.9.1+deva4a7958
-FLA_SHA256=6a6f298a23fbc4902e8379a469696bca1cce108db8159d9c5bf6dfe819a22d58
-FLA_DOWNLOAD_URL="https://github.com/flashserve/flash-linear-attention-npu/releases/download/v26.9.1-beta2/${FLA_WHL}"
+uv pip install 'mooncake-transfer-engine-npu===<new-version>'
+uv pip install --force-reinstall --no-deps 'triton-ascend===<new-version>'
+FLA_VERSION=<new-version>
+FLA_SHA256=<new-sha256>
+FLA_DOWNLOAD_URL="https://github.com/flashserve/flash-linear-attention-npu/releases/download/<new-release-tag>/${FLA_WHL}"
```

`FLA_WHL` derives from `FLA_VERSION` and needs no change.

### `mkdocs.yml`

```diff
-  release_cann_version: "9.2.1"
-  cann_image_tag: 9.2.1-910b-ubuntu22.04-py3.12
-  main_cann_version: "9.2.1"
+  release_cann_version: "<new-version>"
+  cann_image_tag: <new-version>-910b-ubuntu22.04-py3.12
+  main_cann_version: "<new-version>"
```
