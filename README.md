# Ascend NPU plugin for vLLM

## Install
```
git clone https://github.com/cosdt/vllm -b apply_plugin
cd vllm
VLLM_TARGET_DEVICE=CPU python setup.py install

git clone https://github.com/cosdt/vllm-ascend-plugin
cd vllm-ascend-plugin
pip install -e .
```
