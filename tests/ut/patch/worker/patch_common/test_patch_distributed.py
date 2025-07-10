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

from tests.ut.base import TestBase


class TestPatchDistributed(TestBase):

    def test_GroupCoordinator_patched(self):
        from vllm.distributed.parallel_state import GroupCoordinator

        from vllm_ascend.patch.worker.patch_common.patch_distributed import \
            GroupCoordinatorPatch

        self.assertIs(GroupCoordinator, GroupCoordinatorPatch)
