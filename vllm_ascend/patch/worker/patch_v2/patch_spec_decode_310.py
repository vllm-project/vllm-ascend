# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P MRv2 patches: CPU spec-decode helpers + MTP routing (no mainline edits)."""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ar_speculator

from vllm_ascend._310p.worker.v2.spec_utils import (
    prepare_decode_inputs_cpu,
    prepare_prefill_inputs_cpu,
    update_draft_inputs_cpu,
)
from vllm_ascend.worker.v2 import aclgraph_utils as aclgraph_utils_mod
from vllm_ascend.worker.v2 import spec_decode as ascend_spec_decode

ar_speculator.prepare_prefill_inputs = prepare_prefill_inputs_cpu
ar_speculator.prepare_decode_inputs = prepare_decode_inputs_cpu
ar_speculator.update_draft_inputs = update_draft_inputs_cpu


def _embed_input_ids(self: nn.Module, input_ids: torch.Tensor, **kwargs):
    return self.original_model.embed_input_ids(input_ids, **kwargs)  # type: ignore[operator]


# Draft ModelWithContext must forward embed_input_ids for Qwen MTP.
if not hasattr(aclgraph_utils_mod.ModelWithContext, "embed_input_ids"):
    aclgraph_utils_mod.ModelWithContext.embed_input_ids = _embed_input_ids  # type: ignore[attr-defined]

_orig_init_speculator = ascend_spec_decode.init_speculator


def _init_speculator_310p(vllm_config, device):
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    if (
        speculative_config.method == "mtp"
        and not speculative_config.use_gemma4_mtp()
        and not speculative_config.use_step3p5_mtp()
    ):
        from vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator import (
            AscendMTPSpeculator310,
        )

        return AscendMTPSpeculator310(vllm_config, device)
    return _orig_init_speculator(vllm_config, device)


ascend_spec_decode.init_speculator = _init_speculator_310p

# Rebind if model_runner already imported ``init_speculator`` by name.
try:
    from vllm_ascend.worker.v2 import model_runner as _mr

    _mr.init_speculator = _init_speculator_310p
except Exception:
    pass
