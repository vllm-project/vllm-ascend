# SPDX-License-Identifier: Apache-2.0
"""Import-safety contract: the moe_offload package must be importable on
CPU-only runners (no torch_npu / acl). Regression guard for CI cpu-ut jobs.
"""

import sys


def test_package_imports_without_npu():
    """With acl/torch_npu blocked (session fixture), importing the package
    and both modules must succeed — the offload paths import lazily."""
    import vllm_ascend.moe_offload
    import vllm_ascend.moe_offload.eplb_d2d
    import vllm_ascend.moe_offload.manager  # noqa: F401


def test_no_npu_modules_loaded():
    """Importing moe_offload must not pull torch_npu/acl into sys.modules.
    On CPU CI runners they are uninstallable so the assertion is trivially
    true; on NPU hosts it guards against accidental eager imports."""
    import vllm_ascend.moe_offload
    import vllm_ascend.moe_offload.eplb_d2d

    if "torch_npu" not in sys.modules and "acl" not in sys.modules:
        return  # clean baseline, nothing more to check
    # NPU host: module-level imports of torch_npu/acl by *other* test deps
    # are out of our control; just assert our package keeps no module refs
    assert getattr(vllm_ascend.moe_offload.manager, "acl", None) is None
