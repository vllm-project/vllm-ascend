#!/bin/bash\n"
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
set -e
if find $1 -type f -name "GroupedMatmul_*json" -print0 | grep -q .; then
    echo "Found GroupedMatmul_*json files, performing sed operation..."
    find $1 -type f -name "GroupedMatmul_*json" -print0 | xargs -0 sed -i 's/FormatAgnostic/FormatDefault/g'
fi

if find $1 -type f -name "MlaProlog_*json" -print0 | grep -q .; then
    echo "Found MlaProlog_*json files, performing sed operation..."
    find $1 -type f -name "MlaProlog_*json" -print0 | xargs -0 sed -i 's/FormatAgnostic/FormatDefault/g'
fi

if find $1 -type f -name "MlaPrologV2_*json" -print0 | grep -q .; then
    echo "Found MlaPrologV2_*json files, performing sed operation..."
    find $1 -type f -name "MlaPrologV2_*json" -print0 | xargs -0 sed -i 's/FormatAgnostic/FormatDefault/g'
fi

if find $1 -type f -name "MlaPrologV3_*json" -print0 | grep -q .; then
    echo "Found MlaPrologV3_*json files, performing sed operation..."
    find $1 -type f -name "MlaPrologV3_*json" -print0 | xargs -0 sed -i 's/FormatAgnostic/FormatDefault/g'
fi

if find $1 -type f -name "GroupedMatmulSwigluQuant_*json" -print0 | grep -q .; then
    echo "Found GroupedMatmulSwigluQuant_*json files, performing sed operation..."
    find $1 -type f -name "GroupedMatmulSwigluQuant_*json" -print0 | xargs -0 sed -i 's/FormatAgnostic/FormatDefault/g'
fi