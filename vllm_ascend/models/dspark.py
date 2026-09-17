# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from pathlib import Path

from vllm_ascend.utils import get_rotation_path


def get_target_rotation_path(vllm_config):
    config = vllm_config.speculative_config.draft_model_config.hf_config
    if hasattr(config, "_ascend_target_rotation_path"):
        path = config._ascend_target_rotation_path
        return Path(path) if path is not None else None
    return get_rotation_path(vllm_config)


@contextmanager
def draft_model_load_context(vllm_config):
    """Preserve target rotation metadata while constructing the draft."""
    config = vllm_config.speculative_config.draft_model_config.hf_config
    existed = hasattr(config, "_ascend_target_rotation_path")
    previous = getattr(config, "_ascend_target_rotation_path", None)
    path = get_rotation_path(vllm_config)
    # TODO: Pass target rotation metadata through an upstream draft-loading interface.
    config._ascend_target_rotation_path = str(path) if path is not None else None
    try:
        yield
    finally:
        if existed:
            config._ascend_target_rotation_path = previous
        else:
            del config._ascend_target_rotation_path


def post_process_dspark_model(model, target_model):
    """Configure the target auxiliary states consumed by the loaded draft."""
    configure_capture = getattr(model, "configure_target_aux_hidden_capture", None)
    if configure_capture is not None:
        configure_capture(target_model)
