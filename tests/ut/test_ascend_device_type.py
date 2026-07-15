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
"""Regression guards for A5 (Ascend 950) chip detection.

A5 is the only device type mapped from a single discrete SoC version value
(``260``), unlike A2/A3/310P which use integer ranges. See
``vllm_ascend/utils.py::check_ascend_device_type``. These CPU-mock tests pin
that contract so that landing a new A5 SoC variant is an intentional,
reviewed change rather than a silent breakage.

Background: A5 (Ascend 950) has extensive runtime support but no dedicated
CI runner yet, so chip-detection regressions only surface on real hardware.
Guarding the detection contract on CPU closes that gap.
"""

from unittest import mock

import pytest

from vllm_ascend import utils
from vllm_ascend.utils import AscendDeviceType

# SoC version integer reported by ``torch_npu.npu.get_soc_version()`` that
# uniquely identifies A5 (Ascend 950). Keep in sync with
# ``vllm_ascend/utils.py::check_ascend_device_type``.
A5_SOC_VERSION = 260


def test_a5_soc_version_260_maps_uniquely():
    """A5 maps from the single SoC value 260, and adjacent values are rejected.

    A2/A3/310P are mapped from ranges (220-225 / 250-255 / 200-205), so a new
    SoC variant within the range is tolerated. A5 is a single discrete value,
    which is fragile: this test documents that 260 is the *only* accepted A5
    SoC version and that 259/261 must not silently resolve to A5.

    The installed device type is set via ``mock.patch`` (not direct mutation)
    so the module-level global is restored after the test and cannot pollute
    other tests in the suite.
    """
    with (
        mock.patch("vllm_ascend.utils._ascend_device_type", AscendDeviceType.A5),
        mock.patch(
            "vllm_ascend.utils.torch_npu.npu.get_soc_version",
            return_value=A5_SOC_VERSION,
        ),
    ):
        # Must not raise: installed device type matches the detected SoC.
        utils.check_ascend_device_type()

    # Adjacent values are unsupported and must raise, not fall back to A5.
    with mock.patch("vllm_ascend.utils._ascend_device_type", AscendDeviceType.A5):
        for adjacent in (A5_SOC_VERSION - 1, A5_SOC_VERSION + 1):
            with mock.patch(
                "vllm_ascend.utils.torch_npu.npu.get_soc_version",
                return_value=adjacent,
            ):
                with pytest.raises(RuntimeError, match="Can not support soc_version"):
                    utils.check_ascend_device_type()


@pytest.mark.parametrize(
    "device_type, expected",
    [
        (AscendDeviceType.A5, True),
        (AscendDeviceType.A2, False),
        (AscendDeviceType.A3, False),
        (AscendDeviceType._310P, False),
    ],
)
def test_is_950_only_true_for_a5(device_type, expected):
    """``is_950()`` must return True iff the device type is A5.

    Several A5-specific code paths branch on ``is_950()``; a broken helper
    would silently route A5 through non-A5 paths (or vice versa).
    """
    with mock.patch(
        "vllm_ascend.utils.get_ascend_device_type",
        return_value=device_type,
    ):
        assert utils.is_950() is expected
