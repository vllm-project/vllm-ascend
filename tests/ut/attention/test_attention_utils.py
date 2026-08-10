#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from vllm_ascend.attention.utils import filter_chunked_req_indices


def test_filter_chunked_req_indices_empty_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [False, False, False],
    )

    torch.testing.assert_close(indices, torch.empty(0, dtype=torch.long))


def test_filter_chunked_req_indices_mixed_mask() -> None:
    indices = filter_chunked_req_indices(
        torch.tensor([2, 1, 3]),
        [True, False, True],
    )

    torch.testing.assert_close(indices, torch.tensor([0, 1, 3, 4, 5]))
