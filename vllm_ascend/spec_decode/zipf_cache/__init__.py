# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2025 The vLLM team.
#
# This file is a part of the vllm-ascend project.

from vllm_ascend.spec_decode.zipf_cache._zipf_cache_cpp import ZipfCache, zipf_hash

__all__ = ["ZipfCache", "zipf_hash"]
