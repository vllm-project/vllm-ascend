#!/bin/bash
mkdir pta
cd pta
wget https://pytorch-package.obs.cn-north-4.myhuaweicloud.com/pta/Daily/v2.5.1/20250218.4/pytorch_v2.5.1_py310.tar.gz
tar -zxvf pytorch_v2.5.1_py310.tar.gz

if [ "$(uname -i)" == "aarch64" ]
then
    pip install ./torch_npu-2.5.1.dev20250218-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
else
    pip install ./torch_npu-2.5.1.dev20250218-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
fi

cd ..
rm -rf pta
