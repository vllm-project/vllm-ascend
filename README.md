# Ascend NPU plugin for vLLM

## Install

### 1. Prepare CANN env

Before install vllm_ascend_plugin, you need to install the Ascend CANN Toolkit and Kernels. Please follow the [installation tutorial](https://ascend.github.io/docs/sources/ascend/quick_install.html#id1) or use the following commands for quick installation:

```
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

### 2. Install vllm cpu

```
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

### 3. Install vllm_ascend_plugin

```
git clone https://github.com/cosdt/vllm-ascend
cd vllm-ascend-plugin
pip install -e .
```

| Requirement  | Minimum | Recommend   |
| ------------ | ------- | ----------- |
| CANN         | 8.0.RC2 | 8.0.RC3     |
| torch        | 2.4.0   | 2.5.1       |
| torch-npu    | 2.4.0   | 2.5.1rc3    |

> [!NOTE]
> Torch 2.5.1 is highly recommended because vLLM strongly depends on it.

## Support Device

- Atlas 800I A2 Inference Server
- Atlas 800T A2 Training Server
- Atals 300T A2 Training Card
