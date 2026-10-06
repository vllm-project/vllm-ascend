# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A5 kernels awaiting an updated custom operator package.

QLI/QSLI must use the device worker count, including metadata scheduling.
The Q/W prologue preserves interleaved RoPE and BF16 projection rounding.
MQSMLA and its metadata continue to come from cannbot-arena-net-ops.
"""
