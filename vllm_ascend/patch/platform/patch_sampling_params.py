# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Avoid repeatedly walking P/D transfer metadata during request admission.

[Role] Backport the transfer-only fast path from vllm-project/vllm#64490.
[Invariants] Other extra_args and all other sampling checks stay upstream.
[Timeouts & Locks] No waits or locks; inspect only the top-level keys.
[Side-Effect Policy] Replace SamplingParams._verify_extra_args at platform setup;
never mutate request metadata or block-id lists.
"""

from functools import wraps

from vllm.sampling_params import SamplingParams


def _patch_sampling_params() -> None:
    original_verify_extra_args = SamplingParams._verify_extra_args

    @wraps(original_verify_extra_args)
    def verify_extra_args(self: SamplingParams) -> None:
        # Rechecking engine-generated P/D block-id lists after MessagePack
        # decoding delays request admission and competes with the decode loop
        # for the GIL. Keep the same transfer-only scope as upstream #64490.
        if self.extra_args and self.extra_args.keys() <= {"kv_transfer_params"}:
            return
        original_verify_extra_args(self)

    SamplingParams._verify_extra_args = verify_extra_args


_patch_sampling_params()
