# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import subprocess
import sys
import textwrap


def test_table_import_does_not_initialize_npu():
    """Importing the override table must not initialize the NPU.

    The worker hook imports the table before the device is selected, so the
    table's value modules must stay free of import-time device side effects.
    Importing vllm.third_party.flash_linear_attention.ops.utils initializes
    the NPU at import time (#19210), so value modules keep such imports inside
    function bodies.

    Registering the table also imports the upstream modules named in its keys
    to locate the originals; the FLA ones initialize the NPU through their own
    module-level imports. That is upstream behavior the #19210 TODO tracks,
    so this test locks only the table import, the side vllm-ascend controls.

    This runs in a subprocess because the surrounding pytest process has
    already registered and may have touched the device.
    """
    script = textwrap.dedent(
        """
        import torch
        import torch_npu
        import vllm_ascend

        assert not torch.npu.is_initialized()
        from vllm_ascend.triton_kernel_overrides import TRITON_KERNEL_OVERRIDES

        assert not torch.npu.is_initialized(), (
            "importing the override table must not initialize the NPU"
        )
        assert len(TRITON_KERNEL_OVERRIDES) > 0
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr
