# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Ascend project
#
# Ascend version of the upstream ``gemma4`` spec_decode module.
# Hosts ``AscendGemma4Speculator``, the Ascend port of the upstream Q-only
# Gemma4 MTP draft speculator that reads the KV cache shared with the
# target model via cross-model KV sharing.
