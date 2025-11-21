# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import os
from copy import deepcopy

from omni.accelerators.pd.ranktable.device import Device, Server, ServerGroup
from omni.accelerators.pd.ranktable.rank_table import RankTableConfig

LOCAL_RANK_TABLE_ENV = "RANK_TABLE_FILE_PATH"


class LocalInfo(ServerGroup):

    def __init__(self, config: RankTableConfig):
        self.config = config
        self._rank_table_info = self.get_ranktable_dict()
        super().__init__(self._rank_table_info)

    def get_ranktable_dict(self):
        env_path = self.config.local_rank_table_path

        with open(env_path, 'r', encoding='utf-8') as f:
            rank_table = json.load(f)

        return rank_table
