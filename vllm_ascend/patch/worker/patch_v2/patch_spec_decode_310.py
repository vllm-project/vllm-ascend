# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Replace autoregressive spec-decode Triton helpers with 310P CPU fallbacks."""

from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as ar_speculator

from vllm_ascend._310p.worker.v2.spec_utils import (
    prepare_decode_inputs_cpu,
    prepare_prefill_inputs_cpu,
    update_draft_inputs_cpu,
)

ar_speculator.prepare_prefill_inputs = prepare_prefill_inputs_cpu
ar_speculator.prepare_decode_inputs = prepare_decode_inputs_cpu
ar_speculator.update_draft_inputs = update_draft_inputs_cpu
