# SPDX-License-Identifier: Apache-2.0

import pytest
from vllm.triton_utils import triton


def test_patch_v2_num_nans_uses_cann_libdevice_only():
    from vllm.v1.worker.gpu.metrics import logits as metrics_logits
    from vllm.v1.worker.gpu.sample import sampler
    from vllm.v1.worker.gpu.spec_decode import rejection_sampler

    import vllm_ascend.patch.worker.patch_v2.patch_triton  # noqa: F401

    try:
        cann_libdevice = triton.language.extra.cann.libdevice
    except AttributeError:
        pytest.skip("Triton-Ascend CANN libdevice is not available.")

    assert metrics_logits.libdevice is cann_libdevice
    assert metrics_logits.get_num_nans.__module__ == ("vllm.v1.worker.gpu.metrics.logits")
    assert sampler.get_num_nans is metrics_logits.get_num_nans
    assert rejection_sampler.get_num_nans is metrics_logits.get_num_nans
