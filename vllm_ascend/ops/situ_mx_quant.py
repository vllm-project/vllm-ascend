# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import lru_cache


@lru_cache(maxsize=1)
def _get_cann_op():
    from cann_ops_nn import situ_mx_quant

    return situ_mx_quant


def situ_mx_quant(x, *, beta, linear_beta, activate_left, dst_type):
    """Use the CANN implementation, loading its extension only when needed."""
    return _get_cann_op()(x, beta=beta, linear_beta=linear_beta, activate_left=activate_left, dst_type=dst_type)
