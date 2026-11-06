# Dependency Version Management

This page lists the pinned dependency stack of vLLM Ascend that changes most
often: which files to update when a version changes, and where each component
is installed from.

## Pinned dependency stack

| Component | Version | Files to update | Installed from |
|---|---|---|---|
| torch | 2.10.0 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | PyTorch CPU index <https://download.pytorch.org/whl/cpu/> (images), PyPI <https://pypi.org/simple/> (CI) |
| torch-npu | 2.10.0.post7 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| triton-ascend | 3.6.0.post2 | `requirements.txt`<br>`pyproject.toml` `[build-system]`<br>`.github/workflows/_selected_tests.yaml`<br>`.github/workflows/schedule_main2main.yaml` (force-reinstall last, `--no-deps`) | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| memfabric-hybrid / memcache-hybrid | 1.3.0 | `requirements.txt`<br>`pyproject.toml` `[build-system]` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi> |
| flash-linear-attention-npu (FLA) | 26.9.1 (images / PR E2E); 26.9.1+deva4a7958 (main2main dev wheel) | Image build — (`FLA_VERSION`; the a5 files also carry `FLA_PACKAGE`)<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>PR —<br>`.github/workflows/_selected_tests.yaml` ("Install FLA NPU" step)<br>main2main —<br>`.github/workflows/schedule_main2main.yaml` (`FLA_VERSION`, `FLA_WHL`, `FLA_SHA256`, `FLA_DOWNLOAD_URL`) | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi>, direct `pip install flash-linear-attention-npu-aX===26.9.1`; the dev wheel is not on the mirror — main2main fetches it from the URL pinned in the workflow |
| flash_attn_npu | 0.5.0 | Image build — (`FLASH_ATTN_NPU_VERSION`; built in-image from the sdist)<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler` | PyPI <https://pypi.org/simple/> (sdist only) |
| mooncake-transfer-engine-npu | 0.3.14rc3 | Image build — (`MOONCAKE_VERSION`)<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler` (hardcoded specifier)<br>PR / main2main —<br>`.github/workflows/_selected_tests.yaml`<br>`.github/workflows/schedule_main2main.yaml` (install step) | Ascend mirror (nightly) <https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly> only; `manylinux_2_35` wheels, never in `requirements.txt` |
| cannbotdsl / cannbot-arena-net-ops | 0.6.0+g91e7c8f / 0.1.0 | Image build —<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler` | Ascend mirror (stable) <https://mirrors.huaweicloud.com/ascend/repos/pypi>, aarch64 only |
| CANN | 9.2.1 | Image build — `FROM` tag of:<br>`Dockerfile`<br>`Dockerfile.openEuler`<br>`Dockerfile.310p`<br>`Dockerfile.310p.openEuler`<br>`Dockerfile.a3`<br>`Dockerfile.a3.openEuler`<br>`Dockerfile.a5`<br>`Dockerfile.a5.openEuler`<br>PR —<br>`mkdocs.yml` (`cann_image_tag`, `main_cann_version`) | `quay.io/ascend/cann` |

By default, update all files listed for a component.
