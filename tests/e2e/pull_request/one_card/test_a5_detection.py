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
"""A5 (Ascend 950) chip-detection contract on real hardware.

The SoC->device-type mapping boundaries (adjacent values rejected, install/
runtime mismatch) can only be exercised by faking ``soc_version``, which is a
CPU UT concern. What *cannot* be validated without real A5 is that an actual
A5 install reports a consistent device type at runtime. This e2e guard runs on
A5 hardware and asserts that; it skips on every other device (there is no A5
CI runner yet).
"""

import pytest

from vllm_ascend import utils
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, is_950


def test_a5_detection_is_consistent_on_hardware():
    """On A5, runtime detection must match the install and is_950() is True."""
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("A5 (Ascend 950) hardware only")

    assert is_950() is True
    # No mismatch between the installed device type and the runtime SoC on a
    # consistent A5 install; must not raise.
    utils.check_ascend_device_type()
