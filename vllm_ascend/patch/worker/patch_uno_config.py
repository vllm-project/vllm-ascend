# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Reuse the platform patch. EngineCore subprocesses only load global/platform
# patches, while workers also import this compatibility module.
#
# The UNO config patch rewrites pydantic schemas and monkey-patches
# ``VllmConfig.num_lookahead_tokens`` /
# ``InputProcessor._validate_lora``. Pickled configs do not carry class-level
# patches, so every fresh worker interpreter must re-apply them before it
# builds a ``SpeculativeConfig`` or reads the lookahead margin.
import vllm_ascend.patch.platform.patch_uno_config  # noqa: F401
