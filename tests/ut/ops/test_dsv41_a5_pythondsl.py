# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from pathlib import Path

from vllm_ascend.ops.dsv41_a5.package_loader import _load_payload_package


def test_a5_payload_uses_vllm_ascend_sources():
    names = (
        "ops",
        "ops.mixed_quant_sparse_flash_mla",
        "ops.mixed_quant_sparse_flash_mla_metadata",
        "ops.quant_lightning_indexer_dsl",
        "ops.quant_lightning_indexer_metadata_dsl",
        "ops.quant_sparse_lightning_indexer_dsl",
        "ops.quant_sparse_lightning_indexer_metadata_dsl",
    )
    previous = {name: sys.modules.get(name) for name in names}
    try:
        package = _load_payload_package()
        source_dir = Path(package.__path__[0]).resolve()
        assert source_dir.name == "pythondsl"
        assert source_dir.parent.name == "ops"
        for name in names[1:]:
            spec = importlib.util.find_spec(name)
            assert spec is not None
            assert Path(spec.origin).resolve().parent == source_dir
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
