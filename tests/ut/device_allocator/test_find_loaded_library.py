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
# See the License for the language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from tests.ut.base import PytestBase
from vllm_ascend.device_allocator.camem import find_loaded_library


class TestFindLoadedLibrary(PytestBase):
    def test_find_loaded_library_success_and_not_found(self):
        path = find_loaded_library("libc")
        assert path is not None, "Expected to find libc library"
        assert path.endswith(".so.6") or ".so" in path
        assert "libc" in path

        path = find_loaded_library("non_existent_library")
        assert path is None, "Expected to not find non-existent library"
