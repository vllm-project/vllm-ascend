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
"""Guards for A5 (Ascend 950) MC2 token-dispatch MXFP quant path.

On A5, MC2 token dispatch selects MXFP communication quantization
(``quant_mode=4``) and passes an explicit ``y_dtype`` through MC2, a path
that recently regressed (#11663 allgatherEP MXFPW4A8, #11287 MXFP4
allgatherEP). Because there is no A5 CI runner yet, such regressions only
surface on real hardware. These CPU-mock tests pin the A5 branch of
``TokenDispatcherWithMC2.get_dispatch_mc2_kwargs`` so it cannot silently
drift back to the non-A5 (``quant_mode=2``) path.
"""

from types import SimpleNamespace

import torch

from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2


def _make_dispatcher(*, a5: bool) -> TokenDispatcherWithMC2:
    """Build a dispatcher without running the heavy ``__init__``.

    ``a5`` controls both ``a5_need_extra_args`` (A5-only MXFP path) and
    ``need_extra_args`` (True for A3/A5). For the non-A5 case we simulate A2,
    where both flags are False, to isolate the A5 quant-mode distinction.
    """
    d = object.__new__(TokenDispatcherWithMC2)
    d.a5_need_extra_args = a5
    d.need_extra_args = a5
    d.moe_expert_num = 8
    d.global_bs = 0
    d.ep_world_size = 1
    d.ep_rank_id = 0
    d.moe_all_to_all_group_name = "mc2-group"
    d.need_expert_scale = False
    d.need_comm_alg = False
    d.enable_dispatch_v2 = False
    return d


def _make_dispatch_input(*, is_mxfp: bool, dispatch_with_quant: bool):
    quant = SimpleNamespace(
        comm_quant_mode=None,
        dispatch_with_quant=dispatch_with_quant,
        is_mxfp=is_mxfp,
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


def _kwargs(dispatcher, dispatch_input):
    return dispatcher.get_dispatch_mc2_kwargs(dispatch_input)


def test_a5_mxfp_dispatch_uses_quant_mode_4_and_y_dtype():
    """A5 + MXFP dispatch must select quant_mode=4 and pass y_dtype."""
    dispatcher = _make_dispatcher(a5=True)
    dispatch_input = _make_dispatch_input(is_mxfp=True, dispatch_with_quant=True)
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["quant_mode"] == 4
    # A5 MXFP path forwards an explicit fp8 dtype through MC2.
    assert kwargs["y_dtype"] == torch.float8_e4m3fn


def test_non_a5_dispatch_uses_quant_mode_2_without_y_dtype():
    """Non-A5 dispatch must NOT take the A5 MXFP path.

    On A2 (simulated here) MXFP dispatch quantization stays at quant_mode=2
    and must not inject y_dtype. If the A5 branch leaked into the non-A5
    path, communication would be quantized with the wrong mode.
    """
    dispatcher = _make_dispatcher(a5=False)
    dispatch_input = _make_dispatch_input(is_mxfp=True, dispatch_with_quant=True)
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["quant_mode"] == 2
    assert "y_dtype" not in kwargs
