import gc

import pytest
import torch

from vllm_ascend.ops.triton.fla.utils import clear_ssm_states


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "state_shape",
    [
        (6, 3, 5, 7),
        (4, 5, 25, 41),
    ],
)
def test_clear_ssm_states_ref_parity(state_shape, dtype):
    torch.manual_seed(0)
    device = "npu"

    ssm_states = torch.randn(*state_shape, device=device, dtype=dtype)
    has_initial_state = torch.tensor(
        [True, False, True, False, False, True][: state_shape[0]],
        device=device,
        dtype=torch.bool,
    )

    ssm_states_ref = ssm_states.clone()
    ssm_states_ref[~has_initial_state, ...] = 0

    clear_ssm_states(ssm_states, has_initial_state)

    torch.testing.assert_close(ssm_states, ssm_states_ref)

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
