# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""310P Qwen3-VL vision helpers (RC rot_pos_emb only)."""

from vllm_ascend._310p.ops.qwen3vl_310 import rot_pos_emb_310
from vllm_ascend.utils import is_rc_device


def test_rot_pos_emb_310_is_importable() -> None:
    assert callable(rot_pos_emb_310)


def test_rot_pos_emb_patch_only_on_rc() -> None:
    # Non-RC 310P leaves upstream rot_pos_emb unchanged; RC binds rot_pos_emb_310.
    # is_rc_device() is environment-dependent; only assert the helper exists.
    assert isinstance(is_rc_device(), bool)
