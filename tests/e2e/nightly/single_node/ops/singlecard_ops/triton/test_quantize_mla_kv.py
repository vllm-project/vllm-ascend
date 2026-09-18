# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.triton.quantize_mla_kv import quantize_mla_kv
from vllm_ascend.worker.flash_kv_cache import split_flash_mla_c8_cache


@pytest.mark.parametrize("tokens", [1, 4, 32, 64, 129, 768])
@pytest.mark.parametrize("inverse_scale", [1.0, 8.0, 64.0])
@torch.inference_mode()
def test_quantize_mla_kv_noncontiguous_pages_graph(tokens, inverse_scale):
    torch.manual_seed(717)
    pages = (tokens + 254) // 128 + 1
    source = torch.randn((tokens, 1, 576), device="npu", dtype=torch.bfloat16)
    # No staging/contiguous operation should be needed for these split views.
    latent, rope = source[..., :512], source[..., 512:]
    latent_storage = torch.full((pages, 3, 128, 1, 512), 7, device="npu", dtype=torch.float32).to(torch.float8_e4m3fn)
    rope_storage = torch.full((pages, 2, 128, 1, 64), 19, device="npu", dtype=torch.bfloat16)
    latent_cache, rope_cache = latent_storage[:, 1], rope_storage[:, 1]
    slots = torch.arange(tokens, device="npu", dtype=torch.int64) + 127
    scale = torch.tensor([inverse_scale], device="npu", dtype=torch.float32)

    def run():
        quantize_mla_kv(latent, rope, latent_cache, rope_cache, slots, scale)

    def expected():
        c, r = latent_storage.float().cpu(), rope_storage.cpu()
        values = (latent.float().cpu() * scale.cpu()).clamp(-448, 448).to(torch.float8_e4m3fn).float()
        pe = rope.cpu()
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                c[slot // 128, 1, slot % 128] = values[row]
                r[slot // 128, 1, slot % 128] = pe[row]
        return c, r

    def check(want):
        torch.npu.synchronize()
        torch.testing.assert_close(latent_storage.float().cpu(), want[0], rtol=0, atol=0)
        torch.testing.assert_close(rope_storage.cpu(), want[1], rtol=0, atol=0)

    latent[0, 0, :2] = torch.tensor([1024, -1024], device="npu", dtype=torch.bfloat16)
    want = expected()
    run()
    check(want)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for owner_mask in ("odd", "even", "none"):
        source.normal_()
        scale.mul_(0.5)
        slots.copy_(torch.arange(tokens, device="npu", dtype=torch.int64) + 127)
        if owner_mask == "none":
            slots.fill_(-1)
            source.fill_(float("nan"))
        else:
            slots[0 if owner_mask == "odd" else 1 :: 2] = -1
        want = expected()
        graph.replay()
        check(want)


@pytest.mark.parametrize("tokens", [32, 64, 129])
@pytest.mark.parametrize("with_head_axis", [False, True])
@torch.inference_mode()
def test_quantize_mla_kv_paged_source_slots_graph(tokens, with_head_axis):
    torch.manual_seed(719)
    source_pages = (tokens + 127) // 128 + 2
    source = torch.randn(source_pages, 3, 128, 576, dtype=torch.bfloat16, device="npu")
    latent, rope = source[:, 1, :, :512], source[:, 1, :, 512:]
    if with_head_axis:
        latent, rope = latent.unsqueeze(2), rope.unsqueeze(2)
    target_pages = (tokens + 254) // 128 + 1
    storage = torch.full((target_pages, 2, 128, 640), 42, dtype=torch.uint8, device="npu")
    kc, kr = split_flash_mla_c8_cache(storage[:, 1].view(torch.float8_e4m3fn))
    source_slots = ((torch.arange(tokens, device="npu") * 73 + 127) % (source_pages * 128)).to(torch.int64)
    slots = torch.arange(tokens, device="npu", dtype=torch.int64) + 127
    scale = torch.tensor([64.0], dtype=torch.float32, device="npu")
    source[0, 1, 127, :2] = torch.tensor([1024, -1024], device="npu", dtype=torch.bfloat16)

    def run():
        quantize_mla_kv(latent, rope, kc, kr, slots, scale, source_slots=source_slots)

    def expected():
        want = storage.cpu()
        want_kc, want_kr = split_flash_mla_c8_cache(want[:, 1].view(torch.float8_e4m3fn))
        source_cpu = source.cpu()
        for dest, origin in zip(slots.cpu().tolist(), source_slots.cpu().tolist()):
            if dest < 0 or origin < 0:
                continue
            value = source_cpu[origin // 128, 1, origin % 128]
            q = (value[:512].float() * scale.cpu()).clamp(-448, 448).to(torch.float8_e4m3fn)
            want_kc[dest // 128, dest % 128, 0].copy_(q)
            want_kr[dest // 128, dest % 128, 0].copy_(value[512:])
        return want

    def check(want, source_before):
        torch.npu.synchronize()
        torch.testing.assert_close(storage.cpu(), want, rtol=0, atol=0)
        torch.testing.assert_close(source.cpu(), source_before, rtol=0, atol=0, equal_nan=True)

    want, source_before = expected(), source.cpu()
    run()
    check(want, source_before)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        run()
    for selector in ("dcp_owned", "source_padding", "all_padding"):
        source.normal_()
        source_slots.copy_((torch.arange(tokens, device="npu") * 37 + 126) % (source_pages * 128))
        slots.copy_(torch.arange(tokens, device="npu", dtype=torch.int64) + 127)
        if selector == "dcp_owned":
            slots[1::2] = -1
        elif selector == "source_padding":
            source_slots[::2] = -1
        else:
            slots.fill_(-1)
            source_slots.fill_(-1)
            source.fill_(float("nan"))
        scale.mul_(0.5)
        want, source_before = expected(), source.cpu()
        graph.replay()
        check(want, source_before)
