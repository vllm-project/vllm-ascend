# Ascend NPU plugin for vLLM

## Use Docker

### 1. Download vllm and vllm_ascend

```bash
git clone https://github.com/cosdt/vllm-ascend
cd vllm-ascend
git clone https://github.com/cosdt/vllm -b apply_plugin
```

### 2. Build Docker Image

> [!NOTE]
> Modify the base image in Dockerfile according to your device. 
> More choices could be found at https://hub.docker.com/r/ascendai/cann

```bash
docker build -t vllm-npu .
```

### 3. Run docker container

> [!NOTE]
> Modify `--device /dev/davinci0` according to your device.

```bash
docker run -dit -v /usr/local/dcmi:/usr/local/dcmi -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi -v /usr/local/Ascend/driver:/usr/local/Ascend/driver -v /etc/ascend_install.info:/etc/ascend_install.info --device /dev/davinci0 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc --shm-size 16G --name vllm vllm-npu:latest bash
```
### 4. Enter the container

```bash
docker exec -it vllm bash
```

## Install from source

### 1. Prepare CANN env

Before install vllm_ascend, you need to install the Ascend CANN Toolkit and Kernels. Please follow the [installation tutorial](https://ascend.github.io/docs/sources/ascend/quick_install.html#id1) or use the following commands for quick installation:

```bash
# replace the url according to your CANN version and devices
# install CANN Toolkit
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-toolkit_8.0.RC3.alpha003_linux-"$(uname -i)".run
bash Ascend-cann-toolkit_8.0.RC1.alpha003_linux-"$(uname -i)".run --install

# install CANN Kernels
wget https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/Milan-ASL/Milan-ASL%20V100R001C17SPC701/Ascend-cann-kernels-910b_8.0.RC1.alpha003_linux.run
bash Ascend-cann-kernels-910b_8.0.RC1.alpha003_linux.run --install

# set env variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

### 2. Install vLLM cpu

```bash
git clone https://github.com/cosdt/vllm -b apply_plugin
cd vllm

sudo apt-get update  -y
sudo apt-get install -y gcc-12 g++-12 libnuma-dev
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 --slave /usr/bin/g++ g++ /usr/bin/g++-12

pip install cmake>=3.26 wheel packaging ninja "setuptools-scm>=8" numpy
pip install -r requirements-cpu.txt

VLLM_TARGET_DEVICE=cpu python setup.py install
```

> [!NOTE]
> Ubuntu 22.04 is highly recommended as the installation on Ubuntu 20.04 may come across some errors.

### 3. Install vllm_ascend

```bash
git clone https://github.com/cosdt/vllm-ascend
cd vllm-ascend
pip install -e .
```

| Requirement  | Minimum | Recommend   |
| ------------ | ------- | ----------- |
| CANN         | 8.0.RC2 | 8.0.RC3     |
| torch        | 2.4.0   | 2.5.1       |
| torch-npu    | 2.4.0   | 2.5.1rc3    |

> [!NOTE]
> Torch 2.5.1 is highly recommended because vLLM strongly depends on it.

## Support Devices

- Atlas 800I A2 Inference Server
- Atlas 800T A2 Training Server
- Atals 300T A2 Training Card

## Contributing

Linting and formatting:

```bash
pip install -r requirements-lint.txt

# 1. Do work and commit your work.
# 2. Format files that differ from origin/main.
bash format.sh
# 3. Commit changed files with message 'Run yapf and ruff'
git commit -m "Run yapf and ruff"
```
