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

The tests follow the AAA (Arrange-Act-Assert) pattern: each parametrized
case has exactly one arrangement, one act, and one assertion. The positive
case uses ``nullcontext`` (failure if the call raises) and the negative
cases use ``pytest.raises`` so every case carries an explicit assertion.
"""

from contextlib import nullcontext
from unittest import mock

import pytest

from vllm_ascend import utils
from vllm_ascend.utils import AscendDeviceType

# SoC version integer reported by ``torch_npu.npu.get_soc_version()`` that
# uniquely identifies A5 (Ascend 950). Keep in sync with
# ``vllm_ascend/utils.py::check_ascend_device_type``.
A5_SOC_VERSION = 260


@pytest.mark.parametrize(
    "soc_version, installed_type, expectation",
    [
        # SoC 260 uniquely resolves to A5: a consistent install must not raise.
        (A5_SOC_VERSION, AscendDeviceType.A5, nullcontext()),
        # SoC 260 must NOT resolve to any other device type -> AssertionError.
        (A5_SOC_VERSION, AscendDeviceType.A2, pytest.raises(AssertionError)),
        (A5_SOC_VERSION, AscendDeviceType.A3, pytest.raises(AssertionError)),
        (A5_SOC_VERSION, AscendDeviceType._310P, pytest.raises(AssertionError)),
        # Adjacent SoC values are unsupported, not a silent fallback to A5.
        (A5_SOC_VERSION - 1, AscendDeviceType.A5, pytest.raises(RuntimeError)),
        (A5_SOC_VERSION + 1, AscendDeviceType.A5, pytest.raises(RuntimeError)),
    ],
    ids=[
        "soc_260_a5_ok",
        "soc_260_vs_a2_mismatch",
        "soc_260_vs_a3_mismatch",
        "soc_260_vs_310p_mismatch",
        "soc_259_unsupported",
        "soc_261_unsupported",
    ],
)
def test_check_ascend_device_type_resolves_a5_soc_uniquely(
    soc_version, installed_type, expectation
):
    """SoC 260 must resolve to A5 and only A5; adjacent values are rejected.

    A2/A3/310P are mapped from ranges (220-225 / 250-255 / 200-205), so a new
    SoC variant within a range is tolerated. A5 is a single discrete value
    (260), which is fragile: this test documents that 260 is the *only*
    accepted A5 SoC version. The mismatch cases prove 260 does not silently
    resolve to A2/A3/310P; the adjacent cases prove 259/261 do not silently
    fall back to A5.
    """
    # Arrange
    with (
        mock.patch("vllm_ascend.utils._ascend_device_type", installed_type),
        mock.patch(
            "vllm_ascend.utils.torch_npu.npu.get_soc_version",
            return_value=soc_version,
        ),
    ):
        # Act + Assert
        with expectation:
            utils.check_ascend_device_type()


@pytest.mark.parametrize(
    "device_type, expected",
    [
        (AscendDeviceType.A5, True),
        (AscendDeviceType.A2, False),
        (AscendDeviceType.A3, False),
        (AscendDeviceType._310P, False),
    ],
    ids=["a5_true", "a2_false", "a3_false", "310p_false"],
)
def test_is_950_returns_true_only_for_a5(device_type, expected):
    """``is_950()`` must return True iff the device type is A5.

    Several A5-specific code paths branch on ``is_950()``; a broken helper
    would silently route A5 through non-A5 paths (or vice versa).
    """
    # Arrange
    with mock.patch(
        "vllm_ascend.utils.get_ascend_device_type",
        return_value=device_type,
    ):
        # Act
        result = utils.is_950()

    # Assert
    assert result is expected
