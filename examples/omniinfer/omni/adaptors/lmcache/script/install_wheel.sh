#!/bin/bash
# Script to build the install wheel package
set -e

ROOT_PATH=$(cd "$(dirname "$0")" && pwd)

export NO_CUDA_EXT=1
echo "Installing the wheel package..."

yum install -y etcd

# Install the wheel package
pip install $ROOT_PATH/dist/mooncake**.whl
pip install $ROOT_PATH/dist/lmcache**.whl
pip install $ROOT_PATH/dist/ascend**.whl

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib:${ROOT_PATH}/lib

python3 -c "import mooncake.engine" && echo "Success: Import succeeded after installation" || { echo "ERROR: Import failed after installation!"; exit 1; }

which mooncake_master || { echo "ERROR: mooncake_master entry point not found!"; exit 1; }
if mooncake_master -version &> /dev/null; then
    echo "mooncake_master entry point is found and executable."
else
    echo "mooncake_master entry point is not found or not executable."
    exit 1
fi

echo "Verifying transfer_engine_bench entry point..."
which transfer_engine_bench || { echo "ERROR: transfer_engine_bench entry point not found!"; exit 1; }

echo "Verifying lmcache and ascend_lmcache..."
if python3 -c "import lmcache; import torch; import ascend_lmcache.c_ops" &> /dev/null; then
    echo "Success: Import succeeded after installation"
else
    echo "ERROR: Import failed after installation!"
    exit 1
fi

echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib64:/usr/local/lib' >> ~/.bashrc
echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${ROOT_PATH}/lib" >> ~/.bashrc
echo -e "Note: You may need to restart your terminal or run 'source ~/.bashrc' to use ascend_lmcache"
