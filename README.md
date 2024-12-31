# Ascend NPU plugin for vLLM

## Install

1. Install vllm cpu

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

2. Install cllm_ascend_plugin

```
git clone https://github.com/cosdt/vllm-ascend-plugin
cd vllm-ascend-plugin
pip install -e .
```
