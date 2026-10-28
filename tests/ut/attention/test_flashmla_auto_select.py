# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import namedtuple
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import vllm_ascend.ops  # noqa: F401  # Register ops before importing attention/device modules.
from vllm_ascend.attention import mla_v1
from vllm_ascend.attention import utils as flashmla
from vllm_ascend.attention.flashmla import prepare_flashmla_batch_req_state, split_flashmla_requests
from vllm_ascend.attention.mla_v1 import AscendMLAImpl
from vllm_ascend.device.hardware_profile import HardwareCapability


def test_flashmla_api_detection_distinguishes_absence_from_broken_import(monkeypatch):
    monkeypatch.setattr(flashmla, "_has_flashmla_module", lambda: True)

    def missing_package(_name):
        raise ModuleNotFoundError("No module named 'cann_ops_transformer'", name="cann_ops_transformer")

    monkeypatch.setattr(flashmla, "import_module", missing_package)
    assert flashmla.get_flashmla_ops() is None

    monkeypatch.setattr(flashmla, "import_module", lambda _name: SimpleNamespace(flash_mla=lambda: None))
    assert flashmla.get_flashmla_ops() is None

    def attention_op():
        return None

    def metadata_op():
        return None

    monkeypatch.setattr(
        flashmla,
        "import_module",
        lambda _name: SimpleNamespace(
            flash_mla=attention_op,
            flash_mla_metadata=metadata_op,
        ),
    )
    assert flashmla.get_flashmla_ops() == (attention_op, metadata_op)

    def broken_dependency(_name):
        raise ModuleNotFoundError("No module named 'installed_package_dependency'", name="installed_package_dependency")

    monkeypatch.setattr(flashmla, "import_module", broken_dependency)
    with pytest.raises(ModuleNotFoundError, match="installed_package_dependency"):
        flashmla.get_flashmla_ops()


def test_flashmla_module_precheck_skips_absent_cann_import(monkeypatch):
    flashmla._has_flashmla_module.cache_clear()
    monkeypatch.setattr(flashmla, "find_spec", lambda _name: None)
    monkeypatch.setattr(flashmla, "import_module", lambda _name: pytest.fail("absent package must not be imported"))
    try:
        assert flashmla.get_flashmla_ops() is None
    finally:
        flashmla._has_flashmla_module.cache_clear()


def test_flashmla_selection_requires_operator_and_supported_mla_config(monkeypatch):
    config = SimpleNamespace(
        use_v2_model_runner=True,
        speculative_config=None,
        parallel_config=SimpleNamespace(decode_context_parallel_size=1),
    )
    impl = SimpleNamespace(
        vllm_config=config,
        num_heads=8,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        fa_quant_layer=False,
        dtype=torch.bfloat16,
        enable_kv_nz=False,
        use_pcp=False,
    )
    monkeypatch.setattr(
        mla_v1,
        "get_current_hardware_profile",
        lambda: SimpleNamespace(supports=lambda capability: capability is HardwareCapability.MLA_FLASH),
    )
    monkeypatch.setattr(mla_v1, "supports_component_major_mla_pd", lambda _config: True)
    monkeypatch.setattr(mla_v1, "enable_sfa", lambda _config: False)
    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (lambda: None, lambda: None))

    assert AscendMLAImpl._can_use_flashmla(impl)

    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: None)
    assert not AscendMLAImpl._can_use_flashmla(impl)

    monkeypatch.setattr(mla_v1, "get_flashmla_ops", lambda: (lambda: None, lambda: None))
    config.use_v2_model_runner = False
    assert not AscendMLAImpl._can_use_flashmla(impl)

    config.use_v2_model_runner = True
    impl.num_heads = 48
    assert not AscendMLAImpl._can_use_flashmla(impl)

    impl.num_heads = 8
    config.parallel_config.decode_context_parallel_size = 2
    assert not AscendMLAImpl._can_use_flashmla(impl)

    config.parallel_config.decode_context_parallel_size = 8
    assert AscendMLAImpl._can_use_flashmla(impl)
    impl.num_heads = 12
    assert AscendMLAImpl._can_use_flashmla(impl)
    config.parallel_config.decode_context_parallel_size = 4
    assert not AscendMLAImpl._can_use_flashmla(impl)


@pytest.mark.parametrize("prefilling", [[True, False, True], [False, False, False]])
@pytest.mark.parametrize("graph_eligible", [None, True, False])
def test_request_order_preserves_fields_and_real_prefill_graph_gate(prefilling, graph_eligible):
    fields = [
        "req_ids",
        "num_scheduled_tokens",
        "idx_mapping_np",
        "prefill_len_np",
        "num_computed_prefill_tokens_np",
        "is_prefilling_np",
        "has_prefill",
        "num_tokens",
    ]
    values = [
        ["a", "b", "c"],
        np.array([1, 1, 1]),
        np.array([10, 20, 30]),
        np.array([100, 200, 300]),
        np.array([11, 22, 33]),
        np.array(prefilling),
        any(prefilling),
        3,
    ]
    if graph_eligible is not None:
        fields += ["num_draft_tokens_np", "prefill_runs_as_decode_np", "max_seq_len_np", "decode_graph_eligible"]
        values += [np.array([0, 1, 0]), np.array(prefilling), np.array([100, 200, 300]), graph_eligible]
    state = cast(Any, namedtuple("BatchReqState", fields))(*values)
    result, uniform = prepare_flashmla_batch_req_state(state, 1)
    order = [1, 0, 2] if state.has_prefill else [0, 1, 2]
    assert result.req_ids == [state.req_ids[index] for index in order]
    for field in state._fields:
        expected = getattr(state, field)
        actual = getattr(result, field)
        if isinstance(expected, np.ndarray):
            np.testing.assert_array_equal(actual, expected[order])
        elif field == "decode_graph_eligible":
            assert actual is (expected and not state.has_prefill)
        elif field != "req_ids":
            assert actual == expected
    assert uniform == (None if state.has_prefill or graph_eligible is False else 1)
    common = SimpleNamespace(
        is_prefilling=torch.from_numpy(result.is_prefilling_np),
        query_start_loc_cpu=torch.tensor([0, 1, 2, 3, 3], dtype=torch.int32),
        num_reqs=4,
        num_actual_tokens=3,
    )
    # A one-token prompt stays FIA prefill; trailing empty graph rows are ignored.
    assert split_flashmla_requests(common) == ((1, 2, 1, 2) if state.has_prefill else (3, 0, 3, 0))


def test_request_split_rejects_decode_after_real_prefill():
    common = SimpleNamespace(
        is_prefilling=torch.tensor([True, False]),
        query_start_loc_cpu=torch.tensor([0, 1, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=2,
    )
    with pytest.raises(RuntimeError, match="decode requests before prefill"):
        split_flashmla_requests(common)
