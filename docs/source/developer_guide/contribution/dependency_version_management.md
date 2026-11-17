# Dependency Version Management

This page lists the pinned dependency stack of vLLM Ascend that changes most
often: which files to update when a version changes, and where each component
is installed from. The versions shown are example values of one known-good
stack.

## Pinned dependency stack

| Component | Example version | Files to update | Installed from |
|---|---|---|---|
| torch | 2.10.0 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | PyTorch CPU index <https://download.pytorch.org/whl/cpu/> (images), PyPI <https://pypi.org/simple/> (CI) |
| torchvision / torchaudio | 0.25.0 / 2.10.0 | `requirements.txt` | PyTorch CPU index (images), PyPI (CI) |
| torch-npu | 2.10.0.post7 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| vLLM | v0.33.0 | `VLLM_TAG` in all eight `Dockerfile*` | GitHub <https://github.com/vllm-project/vllm> (tag clone; `VLLM_COMMIT` overrides) |
| triton-ascend | 3.6.0.post2 | `requirements.txt`<br>`pyproject.toml` `[build-system]`<br>`.github/workflows/_selected_tests.yaml`<br>`.github/workflows/schedule_main2main.yaml` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| memfabric-hybrid / memcache-hybrid | 1.3.0 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| flash-linear-attention-npu (FLA) | 26.9.1 | Image build —<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>PR —<br>`.github/workflows/_selected_tests.yaml`<br>main2main —<br>`.github/workflows/schedule_main2main.yaml` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi>; the wheel variant follows the SoC: `flash-linear-attention-npu-a2` / `-a3` / `-a5` |
| flash_attn_npu | 0.5.0 | Image build —<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler` | PyPI <https://pypi.org/simple/> (sdist only) |
| mooncake-transfer-engine-npu | 0.3.14rc3 | Image build —<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>PR / main2main —<br>`.github/workflows/_selected_tests.yaml`<br>`.github/workflows/schedule_main2main.yaml` | Ascend mirror (nightly) <https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly> only; `manylinux_2_35` wheels, never in `requirements.txt` |
| cannbotdsl / cannbot-arena-net-ops | 0.6.0+g91e7c8f / 0.1.0 | Image build —<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi>, aarch64 only |
| CANN | 9.3.0_20261113161300 | Image build — `FROM` tag of:<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.310p`<br>`Dockerfile.310p.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>Docs —<br>`mkdocs.yml` | SWR `swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann` |

By default, update all files listed for a component.

## Example: changing a version

Every version pin each file contains, so a version change is one pass per
file. The `-` lines use the example versions from the table; replace
`<new-version>` with the target version.

### `requirements.txt`

```diff
-torch==2.10.0
-torch-npu==2.10.0.post7
-torchvision==0.25.0
-torchaudio==2.10.0
-triton-ascend===3.6.0.post2
-memfabric-hybrid===1.3.0
-memcache-hybrid===1.3.0
+torch==<new-version>
+torch-npu==<new-version>
+torchvision==<new-version>
+torchaudio==<new-version>
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
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-910b-ubuntu22.04-py3.12
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-910b-openeuler24.03-py3.12
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-a3-ubuntu22.04-py3.12
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-a3-openeuler24.03-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-910b-ubuntu22.04-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-910b-openeuler24.03-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-a3-ubuntu22.04-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-a3-openeuler24.03-py3.12
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
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-950-ubuntu22.04-py3.12
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-950-openeuler24.03-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-950-ubuntu22.04-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-950-openeuler24.03-py3.12
...
-ARG FLA_VERSION="26.9.1"
-ARG FLA_PACKAGE="flash-linear-attention-npu-a5"
+ARG FLA_VERSION="<new-version>"
+ARG FLA_PACKAGE="<new-package>"
...
-python3 -m pip install "mooncake-transfer-engine-npu===0.3.14rc3" && \
+python3 -m pip install "mooncake-transfer-engine-npu===<new-version>" && \
...
-python3 -m pip install "cannbotdsl===0.6.0+g91e7c8f" "cannbot-arena-net-ops===0.1.0" && \
+python3 -m pip install "cannbotdsl===<new-version>" "cannbot-arena-net-ops===<new-version>" && \
```

### `Dockerfile.310p`, `Dockerfile.310p.openEuler`

```diff
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-310p-ubuntu22.04-py3.12
-FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:9.3.0_20261113161300-310p-openeuler24.03-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-310p-ubuntu22.04-py3.12
+FROM swr.cn-north-12.myhuaweicloud.com/atlas_inference/cann:<new-version>-310p-openeuler24.03-py3.12
```

### `VLLM_TAG` in every `Dockerfile*`

```diff
-ARG VLLM_TAG=v0.33.0
+ARG VLLM_TAG=<new-version>
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
-uv pip install --no-deps flash-linear-attention-npu-a3===26.9.1
+uv pip install 'mooncake-transfer-engine-npu===<new-version>'
+uv pip install --force-reinstall --no-deps 'triton-ascend===<new-version>'
+uv pip install --no-deps flash-linear-attention-npu-a3===<new-version>
```

### `mkdocs.yml`

```diff
-  release_cann_version: "9.2.1"
-  cann_image_tag: 9.2.1-910b-ubuntu22.04-py3.12
-  main_cann_version: "9.2.1"
+  release_cann_version: "<new-version>"
+  cann_image_tag: <new-version>-910b-ubuntu22.04-py3.12
+  main_cann_version: "<new-version>"
```
