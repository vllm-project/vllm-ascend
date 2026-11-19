The commands below need [Ascend Docker Runtime](../../getting_started/installation.md#installation-ascend-docker-runtime) on the **host**. It is the container engine's runtime: it reads `ASCEND_VISIBLE_DEVICES` and binds the selected NPUs into the container. Without it the container starts with no NPU inside, so `npu-smi` reports no device. Install it once on every node that runs containers, and skip the step if it is already installed.

```bash
chmod u+x Ascend-docker-runtime_<version>_linux-<arch>.run
./Ascend-docker-runtime_<version>_linux-<arch>.run --install
systemctl daemon-reload && systemctl restart docker
```
