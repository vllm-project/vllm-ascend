#!/bin/bash

# build custom ops
cd csrc/cann_ops_custom
bash build.sh -n grouped_matmul_swiglu_quant -c ascend910b --disable-check-compatible

# install custom ops
./output/CANN-custom_ops--linux.x86_64.run
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
