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
"""A5 MC2 token-dispatch MXFP quant path on real hardware.

``quant_mode=4`` / ``y_dtype`` selection is internal to
``TokenDispatcherWithMC2.get_dispatch_mc2_kwargs`` and not observable from
model outputs, so it cannot be a model-level e2e check. Instead this runs the
dispatcher method on real A5 (real device detection, not mocked) and asserts
the A5 MXFP branch selects quant_mode=4 and forwards y_dtype. It skips on
every non-A5 device (there is no A5 CI runner yet).
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2
from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


def _make_dispatcher() -> TokenDispatcherWithMC2:
    """Build a dispatcher without the heavy distributed ``__init__``.

    On A5 ``a5_need_extra_args`` is True (set in ``__init__`` from the real
    device type); we set it directly here since the test is already gated to
    A5 at the top.
    """
    d = object.__new__(TokenDispatcherWithMC2)
    d.a5_need_extra_args = True
    d.need_extra_args = True
    d.moe_expert_num = 8
    d.global_bs = 0
    d.ep_world_size = 1
    d.ep_rank_id = 0
    d.moe_all_to_all_group_name = "mc2-group"
    d.need_expert_scale = False
    d.need_comm_alg = False
    d.enable_dispatch_v2 = False
    return d


def _make_dispatch_input():
    quant = SimpleNamespace(
        comm_quant_mode=None,
        dispatch_with_quant=True,
        is_mxfp=True,
        is_fp8=False,
        mxfp=None,
        use_w4a8_per_channel_gmm_swiglu=False,
    )
    routing = SimpleNamespace(
        expert_map=list(range(8)),
        global_redundant_expert_num=0,
        mc2_mask=torch.tensor([1, 0, 1]),
    )
    return SimpleNamespace(
        hidden_states=torch.randn(3, 8),
        topk_weights=torch.randn(3, 2),
        topk_ids=torch.zeros(3, 2, dtype=torch.int64),
        routing=routing,
        quant=quant,
    )


def test_a5_mxfp_dispatch_uses_quant_mode_4_and_y_dtype():
    """On A5, MC2 MXFP dispatch must select quant_mode=4 and pass y_dtype."""
    if get_ascend_device_type() != AscendDeviceType.A5:
        pytest.skip("A5 (Ascend 950) hardware only")

    kwargs = _make_dispatcher().get_dispatch_mc2_kwargs(_make_dispatch_input())
    assert kwargs["quant_mode"] == 4
    assert kwargs["y_dtype"] == torch.float8_e4m3fn
