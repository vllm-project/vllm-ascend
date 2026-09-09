# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.lora.uno import UnoPackedLoRA, prepare_uno_packed_lora, uno_lora_weight_key


@pytest.mark.parametrize("sizes", [(32, 16, 16), (48, 48), (40,)])
def test_packed_projection_preserves_gating_scaling_and_refresh(sizes):
    torch.manual_seed(61)
    aa = tuple(torch.randn(1, 1, 16, 24) for _ in sizes)
    bb = tuple(torch.randn(1, 1, size, 16) for size in sizes)
    packed = UnoPackedLoRA(aa, bb, sizes)
    pointers = (packed.a.data_ptr(), packed.b.data_ptr())
    mask = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    x, base = torch.randn(4, 24), torch.randn(4, sum(sizes))
    for scale in (1.0, 0.5):
        for a, b in zip(aa, bb):
            a.normal_()
            b.normal_()
        packed.refresh()
        assert (packed.a.data_ptr(), packed.b.data_ptr()) == pointers
        reference = base.clone()
        offset = 0
        for a, b, size in zip(aa, bb, sizes):
            shrunk = (x @ a[0, 0].t()) * scale * mask
            reference[:, offset : offset + size] += shrunk @ b[0, 0].t()
            offset += size
        actual = base.clone()
        packed.apply(actual, x, scale, mask)
        torch.testing.assert_close(actual, reference, atol=1e-4, rtol=1e-4)
        assert torch.equal(actual[[0, 3]], base[[0, 3]])


def test_prepare_reloads_contents_into_the_existing_graph_buffers():
    model = torch.nn.Sequential(torch.nn.Module())
    module = model[0]
    module.lora_a_stacked = (torch.ones(1, 1, 4, 8),)
    module.lora_b_stacked = (torch.ones(1, 1, 6, 4),)
    module.output_slices = (6,)
    module.punica_wrapper = SimpleNamespace(_uno_packed_lora={})
    assert prepare_uno_packed_lora(model) == 1
    key = uno_lora_weight_key(module.lora_a_stacked, module.lora_b_stacked)
    original = module.punica_wrapper._uno_packed_lora[key]
    module.lora_b_stacked[0].fill_(2)
    assert prepare_uno_packed_lora(model) == 1
    assert module.punica_wrapper._uno_packed_lora[key] is original
    assert torch.equal(original.b, torch.full_like(original.b, 2))


@pytest.mark.parametrize("slots,output_size", [(2, 6), (1, 7)])
def test_packed_projection_rejects_unsupported_slot_or_output_layout(slots, output_size):
    aa = (torch.ones(slots, 1, 4, 8),)
    bb = (torch.ones(slots, 1, 6, 4),)
    with pytest.raises(ValueError):
        UnoPackedLoRA(aa, bb, (output_size,))


def test_prepare_requires_a_real_linear_adapter():
    with pytest.raises(ValueError, match="no fixed LoRA"):
        prepare_uno_packed_lora(torch.nn.Identity())
