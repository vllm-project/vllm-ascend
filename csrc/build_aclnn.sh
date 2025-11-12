#!/bin/bash

# build custom ops
cd custom_ops/
bash build.sh custom_ops -cascend910_93

# install custom ops
./build_out/custom_ops/run/CANN_ascend910_93_ubuntu_aarch64.run --install-path=/usr/local/Ascend/ascend-toolkit/latest/opp/
source /usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/bin/set_env.bash
