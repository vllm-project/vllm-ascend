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

import pytest
import torch
import torch_npu  # noqa: F401 -- registers torch.npu
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_ascend.lora.quant_moe import _apply_moe_activation


def _situ_reference(gate_up: torch.Tensor, beta: float | None, linear_beta: float | None) -> torch.Tensor:
    gate, up = gate_up.double().chunk(2, dim=-1)
    beta = 1.0 if beta is None else beta
    gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    if linear_beta is not None:
        up = linear_beta * torch.tanh(up / linear_beta)
    return (gate * up).to(gate_up.dtype)


@pytest.mark.parametrize(
    "dtype,rtol,atol",
    [(torch.float16, 2e-3, 2e-3), (torch.bfloat16, 2e-2, 2e-2), (torch.float32, 2e-4, 2e-5)],
)
@pytest.mark.parametrize("beta,linear_beta", [(None, None), (4.0, None), (4.0, 25.0)])
def test_moe_lora_situ_npu_and_graph_replay(
    dtype: torch.dtype,
    rtol: float,
    atol: float,
    beta: float | None,
    linear_beta: float | None,
) -> None:
    gate_up_cpu = torch.tensor(
        [[-10.0, -3.0, 2.0, 8.0, 100.0, -5.0, 9.0, -100.0], [0.0, 1.0, -1.0, 4.0, -1.0, 2.0, 3.0, 5.0]],
        dtype=dtype,
    )
    gate_up = gate_up_cpu.npu()

    def activate() -> torch.Tensor:
        return _apply_moe_activation(
            gate_up,
            MoEActivation.SITU,
            7.0,
            1.7,
            1.0,
            activation_situ_beta=beta,
            activation_situ_linear_beta=linear_beta,
        )

    expected = _situ_reference(gate_up_cpu, beta, linear_beta)
    for _ in range(3):
        result = activate()
    torch.testing.assert_close(result.cpu(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(gate_up.cpu(), gate_up_cpu, rtol=0, atol=0)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        graph_output = activate()
    for changed_input in (gate_up_cpu, -gate_up_cpu, gate_up_cpu * 0.5):
        gate_up.copy_(changed_input)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(
            graph_output.cpu(), _situ_reference(changed_input, beta, linear_beta), rtol=rtol, atol=atol
        )
