#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Vendored rare_chars (rc4) algorithm for Runtime Guard.

Algorithm and tables adapted from RinshannKaihou/self_evolving_inference_detector
(v35_detector 0.4.0 / research commit 5118334). Guard owns the Ascend collect
path and ConfigBackedDetector wiring; this package is CPU-only (numpy).

See NOTICE in this directory for provenance.
"""

from __future__ import annotations

import os

from .rare import RareDetector
from .result import Result

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
TABLES_DIR = os.path.join(PKG_DIR, "tables")
CONFIG_RC4 = os.path.join(PKG_DIR, "rc4.json")

from .rare_vocab import THINK_END  # noqa: E402

__all__ = [
    "CONFIG_RC4",
    "PKG_DIR",
    "RareDetector",
    "Result",
    "TABLES_DIR",
    "THINK_END",
]
