# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

# This package must stay side-effect free: importing any V2 submodule runs
# this file first, and Model Runner V1 must not pick up the V2 kernel registry
# or any default Triton kernel definition.
