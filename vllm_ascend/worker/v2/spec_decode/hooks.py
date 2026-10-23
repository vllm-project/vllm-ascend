# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vllm-ascend project

from functools import wraps


def bind_verify_input_hook(speculator, update_positions) -> None:
    """Refresh Ascend's rotary buffers after a verifier rewrites positions.

    Bind once at initialization. The upstream runner continues to own hook
    dispatch, including dummy forwards and graph replay input staging.
    """
    prepare = speculator.prepare_verify_inputs

    @wraps(prepare)
    def prepare_verify_inputs(input_batch, model_inputs, *, dummy_run=False):
        prepare(input_batch, model_inputs, dummy_run=dummy_run)
        update_positions(input_batch.positions)

    speculator.prepare_verify_inputs = prepare_verify_inputs
