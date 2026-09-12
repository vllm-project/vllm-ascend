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

import pytest
import torch

from vllm_ascend.worker.cpu_gpu_buffer import AscendCpuGpuBuffer


class DeferredCopy:
    def __init__(self, pending, selection=None):
        self.pending = pending
        self.selection = selection

    def __getitem__(self, selection):
        return DeferredCopy(self.pending, selection)

    def copy_(self, source, non_blocking=False):
        assert non_blocking
        self.pending.append((self.selection, source))
        return self


@pytest.mark.parametrize("length", [None, 3, 0])
def test_copy_preserves_multiple_pending_generations(length):
    buffer = AscendCpuGpuBuffer(5, dtype=torch.int32, device=torch.device("cpu"), pin_memory=False)
    pending = []
    buffer.gpu = DeferredCopy(pending)
    expected = []
    for values in ([0, 6, 12, 18, 24], [0, 3, 9, 9, 9], [0, 2, 2, 2, 2]):
        buffer.np[:] = values
        expected.append(values if length is None else values[:length])
        buffer.copy_to_gpu(length)
    buffer.np.fill(-1)
    # Consume only after every source generation has been overwritten.
    assert [source.tolist() for _, source in pending] == expected
    assert all(source.data_ptr() != buffer.cpu.data_ptr() for _, source in pending if source.numel())
    assert all(selection == (None if length is None else slice(None, length)) for selection, _ in pending)


def test_copy_preserves_destination_identity_and_tail():
    buffer = AscendCpuGpuBuffer(5, dtype=torch.int64, device=torch.device("cpu"), pin_memory=False)
    buffer.gpu.fill_(-7)
    pointer = buffer.gpu.data_ptr()
    buffer.np[:] = [0, 6, 12, 18, 24]
    result = buffer.copy_to_gpu(3)
    assert result.data_ptr() == pointer
    assert buffer.gpu.tolist() == [0, 6, 12, -7, -7]
    buffer.np.fill(100)
    assert buffer.gpu.tolist() == [0, 6, 12, -7, -7]
