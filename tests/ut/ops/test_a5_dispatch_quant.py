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
surface on real hardware.

These CPU-mock tests pin every branch of
``TokenDispatcherWithMC2.get_dispatch_mc2_kwargs``:

* A5 + MXFP      -> quant_mode=4, y_dtype=float8_e4m3fn
* non-A5 (A2)    -> quant_mode=2, no y_dtype
* comm_quant_mode override -> wins over the A5/MXFP selection
* no dispatch_with_quant  -> quant_mode=0
* A5 + FP8       -> y_dtype injected (quant_mode stays 2)
* A5 + mxfp.act_quant_type -> y_dtype taken from the quant config
* A3 (extra args, non-A5) -> tp args added, no y_dtype leak
* global_bs side effect   -> x_active_mask gated on global_bs==0

The dispatcher is built via ``object.__new__`` to bypass the heavy
``__init__`` (HCCL/distributed setup). ``_get_expert_token_nums_type`` is
mocked explicitly (it is a module-level helper) so the test contract is
unambiguous and its result is asserted rather than silently computed.
"""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithMC2

# Sentinel pinned by the _get_expert_token_nums_type mock so we can assert the
# value is forwarded into kwargs instead of being computed by real logic.
_EXPERT_TOKEN_NUMS_TYPE_SENTINEL = 999999


def _make_dispatcher(*, a5: bool, need_extra_args: bool | None = None, global_bs: int = 0) -> TokenDispatcherWithMC2:
    """Build a dispatcher without running the heavy ``__init__``.

    ``a5`` controls ``a5_need_extra_args`` (A5-only MXFP path).
    ``need_extra_args`` defaults to ``a5`` (True for A3/A5, False for A2) but
    is overridable so the A3 shape (extra args but non-A5) can be exercised.
    """
    if need_extra_args is None:
        need_extra_args = a5
    d = object.__new__(TokenDispatcherWithMC2)
    d.a5_need_extra_args = a5
    d.need_extra_args = need_extra_args
    d.moe_expert_num = 8
    d.global_bs = global_bs
    d.ep_world_size = 1
    d.ep_rank_id = 0
    d.moe_all_to_all_group_name = "mc2-group"
    d.need_expert_scale = False
    d.need_comm_alg = False
    d.enable_dispatch_v2 = False
    return d


def _make_dispatch_input(
    *,
    is_mxfp: bool,
    dispatch_with_quant: bool,
    comm_quant_mode=None,
    is_fp8: bool = False,
    act_quant_type=None,
):
    mxfp = None
    if act_quant_type is not None:
        mxfp = SimpleNamespace(act_quant_type=act_quant_type)
    quant = SimpleNamespace(
        comm_quant_mode=comm_quant_mode,
        dispatch_with_quant=dispatch_with_quant,
        is_mxfp=is_mxfp,
        is_fp8=is_fp8,
        mxfp=mxfp,
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
    """Call ``get_dispatch_mc2_kwargs`` with the expert-token-nums helper mocked."""
    with mock.patch(
        "vllm_ascend.ops.fused_moe.token_dispatcher._get_expert_token_nums_type",
        return_value=_EXPERT_TOKEN_NUMS_TYPE_SENTINEL,
    ):
        return dispatcher.get_dispatch_mc2_kwargs(dispatch_input)


# ---------------------------------------------------------------------------
# quant_mode selection
# ---------------------------------------------------------------------------


def test_a5_mxfp_dispatch_uses_quant_mode_4_and_y_dtype():
    """A5 + MXFP dispatch must select quant_mode=4 and pass y_dtype."""
    dispatcher = _make_dispatcher(a5=True)
    dispatch_input = _make_dispatch_input(is_mxfp=True, dispatch_with_quant=True)
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["quant_mode"] == 4
    assert kwargs["y_dtype"] == torch.float8_e4m3fn


def test_non_a5_dispatch_uses_quant_mode_2_without_y_dtype():
    """Non-A5 (A2) dispatch must stay at quant_mode=2 and not inject y_dtype."""
    dispatcher = _make_dispatcher(a5=False)
    dispatch_input = _make_dispatch_input(is_mxfp=True, dispatch_with_quant=True)
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["quant_mode"] == 2
    assert "y_dtype" not in kwargs


def test_comm_quant_mode_overrides_quant_mode_selection():
    """An explicit comm_quant_mode wins over the A5/MXFP quant_mode branch."""
    dispatcher = _make_dispatcher(a5=True)
    dispatch_input = _make_dispatch_input(
        is_mxfp=True,
        dispatch_with_quant=True,
        comm_quant_mode=7,
    )
    kwargs = _kwargs(dispatcher, dispatch_input)

    # 7 is the explicit override; the A5 path (4) must not be selected.
    assert kwargs["quant_mode"] == 7


def test_no_dispatch_with_quant_uses_quant_mode_zero():
    """Without dispatch quantization, quant_mode falls back to 0."""
    dispatcher = _make_dispatcher(a5=True)
    dispatch_input = _make_dispatch_input(is_mxfp=True, dispatch_with_quant=False)
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["quant_mode"] == 0
    # No quant dispatch -> no y_dtype even on A5.
    assert "y_dtype" not in kwargs


# ---------------------------------------------------------------------------
# y_dtype injection
# ---------------------------------------------------------------------------


def test_a5_fp8_dispatch_injects_y_dtype():
    """A5 + FP8 must inject y_dtype (the is_fp8 branch of the condition).

    Note quant_mode stays 2 here: FP8 is not MXFP, so the quant_mode selector
    keeps 2 while y_dtype is still forwarded.
    """
    dispatcher = _make_dispatcher(a5=True)
    dispatch_input = _make_dispatch_input(is_mxfp=False, is_fp8=True, dispatch_with_quant=True)
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["quant_mode"] == 2
    assert kwargs["y_dtype"] == torch.float8_e4m3fn


def test_a5_mxfp_act_quant_type_overrides_y_dtype():
    """When mxfp.act_quant_type is set, it overrides the default y_dtype."""
    dispatcher = _make_dispatcher(a5=True)
    dispatch_input = _make_dispatch_input(
        is_mxfp=True,
        dispatch_with_quant=True,
        act_quant_type=torch.float8_e5m2,
    )
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["y_dtype"] == torch.float8_e5m2


# ---------------------------------------------------------------------------
# A3 path (extra args, non-A5) and side effects
# ---------------------------------------------------------------------------


def test_a3_dispatch_adds_tp_args_without_y_dtype():
    """A3 (need_extra_args=True, non-A5) adds tp args but must not leak y_dtype."""
    dispatcher = _make_dispatcher(a5=False, need_extra_args=True)
    dispatch_input = _make_dispatch_input(is_mxfp=True, dispatch_with_quant=True)
    kwargs = _kwargs(dispatcher, dispatch_input)

    # Non-A5 -> quant_mode=2, no y_dtype.
    assert kwargs["quant_mode"] == 2
    assert "y_dtype" not in kwargs
    # A3 carries tp communication args.
    assert "group_tp" in kwargs
    assert kwargs["tp_world_size"] == 1
    assert kwargs["tp_rank_id"] == 0


@pytest.mark.parametrize(
    "global_bs, has_x_active_mask",
    [
        (0, True),
        (1, False),
    ],
    ids=["global_bs_zero_sets_mask", "global_bs_nonzero_omits_mask"],
)
def test_x_active_mask_gated_on_global_bs(global_bs, has_x_active_mask):
    """x_active_mask is only forwarded when global_bs == 0."""
    dispatcher = _make_dispatcher(a5=False, global_bs=global_bs)
    dispatch_input = _make_dispatch_input(is_mxfp=False, dispatch_with_quant=False)
    kwargs = _kwargs(dispatcher, dispatch_input)

    if has_x_active_mask:
        assert "x_active_mask" in kwargs
        assert kwargs["x_active_mask"] is dispatch_input.routing.mc2_mask
    else:
        assert "x_active_mask" not in kwargs


def test_expert_token_nums_type_is_forwarded():
    """The _get_expert_token_nums_type result is forwarded into kwargs."""
    dispatcher = _make_dispatcher(a5=False)
    dispatch_input = _make_dispatch_input(is_mxfp=False, dispatch_with_quant=False)
    kwargs = _kwargs(dispatcher, dispatch_input)

    assert kwargs["expert_token_nums_type"] is _EXPERT_TOKEN_NUMS_TYPE_SENTINEL
