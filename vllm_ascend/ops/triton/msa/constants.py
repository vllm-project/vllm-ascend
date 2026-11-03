# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# FP32 represents every integer through 2**24 exactly. Reserve that endpoint
# as the invalid sort key: valid logical block IDs must be in [0, 2**24).
# Unlike +inf this finite key is supported by the Ascend FP32 sort extension.
FP32_INDEX_SENTINEL = 1 << 24
