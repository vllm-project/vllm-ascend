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

import unittest

from vllm_ascend.utils import adapt_patch, register_ascend_customop

# fused moe ops test will hit the infer_schema error, we need add the patch
# here to make the test pass.
import vllm_ascend.patch.worker.patch_common.patch_utils  # type: ignore[import]  # isort: skip  # noqa


class TestBase(unittest.TestCase):

    def setUp(self):
        # adapt patch by default.
        adapt_patch(True)
        adapt_patch()
        register_ascend_customop()
        super().setUp()
