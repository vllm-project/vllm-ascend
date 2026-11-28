# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# mypy: ignore-errors

"""310P MRv2 patches: CPU spec-decode helpers + MTP routing (no mainline edits)."""

from __future__ import annotations

from vllm.v1.worker.gpu.spec_decode.mtp.config import MTPSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.target_dependent_ar import speculator as ar_speculator

from vllm_ascend._310p.worker.v2.spec_utils import (
    prepare_decode_inputs_cpu,
    prepare_prefill_inputs_cpu,
    update_draft_inputs_cpu,
)
from vllm_ascend.worker.v2 import spec_decode as ascend_spec_decode

ar_speculator.prepare_prefill_inputs = prepare_prefill_inputs_cpu
ar_speculator.prepare_decode_inputs = prepare_decode_inputs_cpu
ar_speculator.update_draft_inputs = update_draft_inputs_cpu
# Draft ACLGraph manager override is applied in patch/worker/__init__.py
# *after* patch_eagle_speculator (which would otherwise win).


# Resolve only plain MTP to its Triton-free implementation on this worker.
ascend_spec_decode._UPSTREAM_TO_ASCEND_SPECULATOR[MTPSpeculatorConfig.speculator_class] = (
    "vllm_ascend._310p.worker.v2.spec_decode.mtp_speculator.AscendMTPSpeculator310"
)
