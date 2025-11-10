#!/bin/bash

# build custom ops
cd custom_ops/
bash build.sh custom_ops -cascend910_93

# install custom ops
# ./output/CANN-custom_ops--linux.x86_64.run
# export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
