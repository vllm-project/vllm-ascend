#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

from dataclasses import replace
from pathlib import Path

import pytest
import torch

from vllm_ascend._310p.graph_input_contract import (
    GraphInputContractError,
    capture_graph_input_contracts,
    validate_graph_input_contracts,
)


def test_capture_graph_input_contracts_discovers_nested_args_and_kwargs() -> None:
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    strided_view = base[1:4:2, 1:6:2]
    keyword_tensor = torch.ones((2, 3), dtype=torch.int32)

    contracts = capture_graph_input_contracts(
        (strided_view, {"nested": [keyword_tensor]}),
        {"mask": keyword_tensor[:, 1:]},
    )

    assert [contract.path for contract in contracts] == [
        "args[0]",
        "args[1].nested[0]",
        "kwargs.mask",
    ]
    view_contract = contracts[0]
    assert view_contract.data_ptr == strided_view.data_ptr()
    assert view_contract.base_ptr == base.untyped_storage().data_ptr()
    assert view_contract.storage_offset == strided_view.storage_offset()
    assert view_contract.storage_nbytes == base.untyped_storage().nbytes()
    assert view_contract.view_start_byte == strided_view.storage_offset() * 4
    assert view_contract.view_end_byte <= view_contract.storage_nbytes
    assert view_contract.dtype == "torch.float32"
    assert view_contract.shape == tuple(strided_view.shape)
    assert view_contract.stride == tuple(strided_view.stride())
    assert view_contract.contiguous is False
    assert view_contract.device == "cpu"
    assert view_contract.required_alignment == strided_view.element_size()
    assert view_contract.alignment_ok is True


@pytest.mark.parametrize(
    "replacement",
    [
        lambda tensor: tensor.clone(),
        lambda tensor: tensor.as_strided((2, 2), (4, 1), storage_offset=1),
        lambda tensor: tensor.to(torch.float64),
        lambda tensor: tensor.reshape(4, 1),
    ],
)
def test_validate_graph_input_contracts_rejects_changed_contract(replacement) -> None:
    tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, :2]
    expected = capture_graph_input_contracts((tensor,), {})
    actual = capture_graph_input_contracts((replacement(tensor),), {})

    with pytest.raises(GraphInputContractError, match=r"args\[0\]"):
        validate_graph_input_contracts(expected, actual)


@pytest.mark.parametrize(
    ("field_name", "replacement_value"),
    [
        ("data_ptr", lambda contract: contract.data_ptr + 4),
        ("base_ptr", lambda contract: contract.base_ptr + 4),
        ("storage_offset", lambda contract: contract.storage_offset + 1),
        ("storage_nbytes", lambda contract: contract.storage_nbytes + 4),
        ("view_end_byte", lambda contract: contract.storage_nbytes + 1),
        ("stride", lambda contract: (4, 1)),
        ("alignment_ok", lambda contract: False),
    ],
)
def test_validate_graph_input_contracts_rejects_metadata_mutation(
    field_name: str,
    replacement_value,
) -> None:
    expected = capture_graph_input_contracts(
        (torch.arange(4, dtype=torch.float32).reshape(2, 2),),
        {},
    )
    actual_contract = replace(
        expected[0],
        **{field_name: replacement_value(expected[0])},
    )

    with pytest.raises(GraphInputContractError, match=field_name):
        validate_graph_input_contracts(expected, (actual_contract,))


def test_contract_collection_does_not_use_tensor_item() -> None:
    source_path = Path(__file__).resolve().parents[3] / "vllm_ascend" / "_310p" / "graph_input_contract.py"
    source = source_path.read_text(encoding="utf-8")

    assert ".item(" not in source
