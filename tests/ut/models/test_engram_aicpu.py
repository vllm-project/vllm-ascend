# SPDX-License-Identifier: Apache-2.0
"""Real 950DT regressions for raw URMA bytes, Cube projection and replay."""

import subprocess
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("aligned,parallel", [(False, False), (True, False), (True, True)])
def test_urma_cube_bytes_projection_and_graph(id_dtype, aligned, parallel):
    try:
        subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("Requires Ascend 950 and the built URMA extension")
    import torch_npu

    # Match worker startup: register Ascend ops before importing device adaptors.
    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41.engram import npu
    from vllm_ascend.quantization.methods.w8a8.w8a8_mxfp8 import AscendW8A8MXFP8DynamicLinearMethod

    if torch_npu.npu.get_soc_version() != 260:
        pytest.skip("Requires Ascend 950")
    torch.npu.set_device(0)
    source = npu.HostUvaBuffer((256, 256), torch.float8_e4m3fn, torch.device("npu:0"))
    lookup = npu.EngramUrmaCubeLookup(parallel_scales=parallel)
    graph = None
    try:
        raw = torch.arange(256 * 256, dtype=torch.int32).reshape(256, 256).remainder(127).byte()
        source.tensor.view(torch.uint8).copy_(raw)
        cpu_scales = torch.arange(256 * 8, dtype=torch.int32).reshape(256, 8).remainder(8).add(117).byte()
        scales = cpu_scales.to("npu")
        # A slice retains stride 6, and only columns 1..4 belong to this lookup.
        cpu_ids = torch.tensor([[99, 100, 355, 356, -1, 0]] * 320, dtype=id_dtype)
        ids = cpu_ids.to("npu")[:, :5]
        packed = npu.allocate_compressed_engram_buffer(352, 4, 256, "npu")
        if not aligned:
            packed = torch.zeros(packed.numel() + 1, dtype=torch.uint8, device="npu")[1:].view_as(packed)
        assert npu.can_urma_write_output(packed) is aligned
        assert not npu.can_urma_write_output(packed.view(-1)[1:])
        codes, row_scales = npu.compressed_engram_views(packed, 4, 256)
        assert codes.is_contiguous() and row_scales.is_contiguous()
        weight, weight_scale = torch_npu.npu_dynamic_mx_quant(
            torch.randn(128, 1024, dtype=torch.bfloat16, device="npu"), dst_type=torch.float8_e4m3fn
        )
        layer = SimpleNamespace(weight=weight.t().contiguous(), weight_scale=weight_scale.transpose(0, 1).contiguous())
        scheme = AscendW8A8MXFP8DynamicLinearMethod.__new__(AscendW8A8MXFP8DynamicLinearMethod)
        scheme.group_size = 32

        def produce():
            lookup.lookup_codes(
                source,
                scales,
                ids,
                head_start=1,
                local_heads=4,
                vocab_start=100,
                vocab_end=356,
                output_codes=codes[:320].view(1280, 256),
                output_scales=row_scales[:320],
            )

        def project():
            return scheme.apply(
                layer,
                (codes[:320].view(torch.float8_e4m3fn), row_scales[:320].view(320, -1, 2).view(torch.float8_e8m0fnu)),
            )

        def check(actual):
            selected = ids.cpu()[:, 1:5].long().flatten()
            owned = (selected >= 100) & (selected < 356)
            local = torch.where(owned, selected - 100, 0)
            expected_codes, expected_scales = raw[local].clone(), cpu_scales[local].clone()
            expected_codes[~owned] = 0
            expected_scales[~owned] = 0
            assert torch.equal(codes[:320].cpu(), expected_codes.view(320, -1))
            assert torch.equal(row_scales[:320].cpu(), expected_scales.view(320, -1))
            reference = torch_npu.npu_quant_matmul(
                expected_codes.to("npu").view(320, -1).view(torch.float8_e4m3fn),
                layer.weight,
                layer.weight_scale,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale=expected_scales.to("npu").view(320, -1, 2),
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                output_dtype=torch.bfloat16,
                group_sizes=[1, 1, 32],
            )
            assert torch.equal(actual.cpu(), reference.cpu())
            assert torch.isfinite(actual.cpu()).all()
            assert not codes[320:].cpu().any() and not row_scales[320:].cpu().any()

        produce()
        check(project())
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            produce()
            actual = project()
        for index in [100, 227, 355, -1, 356]:
            ids.fill_(index)
            graph.replay()
            check(actual)
        # Alternate valid/invalid windows, including a wholly invalid bank.
        for invalid in (slice(0, 256), slice(128, 288), slice(256, 320)):
            ids.fill_(227)
            ids[invalid].fill_(-1)
            graph.replay()
            check(actual)
        with pytest.raises(ValueError, match="head range"):
            lookup.lookup_codes(
                source,
                scales,
                ids,
                head_start=2,
                local_heads=4,
                output_codes=codes[:320].view(1280, 256),
                output_scales=row_scales[:320],
            )
    finally:
        torch.npu.synchronize()
        del graph
        lookup.close()
        lookup.close()
        assert lookup.state_slot.cpu().item() == 0
        source.close()
        source.close()


def test_fused_masks_preserve_replay_count_and_padding():
    try:
        subprocess.run(["npu-smi", "info"], capture_output=True, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("Requires Ascend 950")
    import torch_npu

    import vllm_ascend.ops  # noqa: F401
    from vllm_ascend.models.deepseek_v41.engram.prep_mask import prepare_engram_masks

    if torch_npu.npu.get_soc_version() != 260:
        pytest.skip("Requires Ascend 950")
    torch.npu.set_device(0)
    ids = torch.tensor([12, 0, 17, 0, 18, 0, 13, 0], dtype=torch.int64, device="npu")[::2]
    lookback = torch.tensor([[-1, 0, 17, 0, 18, 0]], dtype=torch.int64, device="npu")[:, ::2]
    count = torch.tensor([4], dtype=torch.int64, device="npu")
    owner = torch.ones(15, dtype=torch.bool, device="npu")
    keep = owner[4:11]

    def prepare():
        return prepare_engram_masks(
            ids,
            lookback,
            image_token_id=17,
            image_pad_token_id=18,
            valid_token_count=count,
            mask_output_buffer=keep,
            output_tokens=7,
        )

    prepare()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        dead, _, prior_dead = prepare()
    for valid in (4, 1, 0, 3):
        count.fill_(valid)
        owner.fill_(True)
        graph.replay()
        torch.npu.synchronize()
        reference = (ids.cpu() == 17) | (ids.cpu() == 18) | (torch.arange(4) >= valid)
        assert torch.equal(dead.cpu(), reference)
        assert torch.equal(keep[:4].cpu(), ~reference)
        assert torch.equal(prior_dead.cpu(), torch.tensor([[False, True, True]]))
        assert not keep[4:].cpu().any()
        assert owner[:4].cpu().all() and owner[11:].cpu().all()
    del graph
