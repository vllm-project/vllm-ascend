# SPDX-License-Identifier: Apache-2.0
from tests.e2e.pull_request.rlhf.step5_weight_transfer_utils import run_oracle


def test_step5_hccl_text_update_matches_reference(request):
    run_oracle(request, "hccl")
