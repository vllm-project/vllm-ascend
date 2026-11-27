# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Optional Ascend 950 TileLang kernels and worker-side integration."""

from importlib.util import find_spec

HAS_TILELANG = find_spec("tilelang") is not None
