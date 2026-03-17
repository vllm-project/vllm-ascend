#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#

from vllm_ascend.model_loader.netloader import register_netloader
from vllm_ascend.model_loader.rfork import register_rforkloader

__all__ = ["register_netloader", "register_rforkloader"]
