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
# This file is a part of the vllm-ascend project.
#

import torch
from vllm.v1.utils import CpuGpuBuffer


class AscendCpuGpuBuffer(CpuGpuBuffer):
    """Keep each asynchronous H2D source independent of the next CPU update."""

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        source = self.cpu if n is None else self.cpu[:n]
        target = self.gpu if n is None else self.gpu[:n]
        # Retaining the reusable CPU buffer does not prevent its next update.
        # Give this copy its own allocation; the pinned host allocator delays
        # recycling that allocation until the asynchronous copy completes.
        snapshot = torch.empty_like(source, pin_memory=source.is_pinned())
        snapshot.copy_(source)
        return target.copy_(snapshot, non_blocking=True)
