# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# The CANN dispatcher handle is process-wide and immutable after registration.
# Concurrent first calls may resolve it twice; both resolve to the same handle.
_cann_situ_mx_quant = None


def situ_mx_quant(x, *, beta, linear_beta, activate_left, dst_type):
    """Use the CANN implementation, loading its extension only when needed."""
    global _cann_situ_mx_quant
    if _cann_situ_mx_quant is None:
        from cann_ops_nn import situ_mx_quant as cann_situ_mx_quant

        _cann_situ_mx_quant = cann_situ_mx_quant

    return _cann_situ_mx_quant(x, beta=beta, linear_beta=linear_beta, activate_left=activate_left, dst_type=dst_type)
