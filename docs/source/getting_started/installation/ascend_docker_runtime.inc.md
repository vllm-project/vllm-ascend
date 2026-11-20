The commands below need [Ascend Docker Runtime](../../getting_started/installation.md#installation-ascend-docker-runtime) on the **host**: install it once per host, not once per container, and skip the step if it is already installed. It is the container engine's runtime, so it reads `ASCEND_VISIBLE_DEVICES` and binds the selected NPUs into the container; without it the container starts with no NPU inside and `npu-smi` reports no device. The installation and verification steps are documented in the [installation guide](../../getting_started/installation.md#installation-ascend-docker-runtime); the block below repeats them so this page stays self-contained.

```bash
chmod u+x Ascend-docker-runtime_<version>_linux-<arch>.run
./Ascend-docker-runtime_<version>_linux-<arch>.run --install
systemctl daemon-reload && systemctl restart docker
```
