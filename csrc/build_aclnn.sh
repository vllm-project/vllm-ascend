#!/bin/bash

# build custom ops
cd csrc/cann_ops_custom
# TODO: support auto detect npu device (ascend910b/ascend910_93)
bash build.sh -n grouped_matmul_swiglu_quant -c ascend910b --disable-check-compatible

# install custom ops
./output/CANN-custom_ops*.run
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/opp/vendors/customize/op_api/lib/:${LD_LIBRARY_PATH}
