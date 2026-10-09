# SPDX-License-Identifier: Apache-2.0
"""Real 950DT regressions for raw URMA bytes, Cube projection and replay."""

import subprocess
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
def test_urma_cube_bytes_projection_and_graph(id_dtype):
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
    lookup = npu.EngramUrmaCubeLookup()
    graph = None
    try:
        raw = torch.arange(256 * 256, dtype=torch.int32).reshape(256, 256).remainder(127).byte()
        source.tensor.view(torch.uint8).copy_(raw)
        cpu_scales = torch.arange(256 * 8, dtype=torch.int32).reshape(256, 8).remainder(8).add(117).byte()
        scales = cpu_scales.to("npu")
        # A slice retains stride 6, and only columns 1..4 belong to this lookup.
        cpu_ids = torch.tensor([[99, 100, 355, 356, -1, 0]] * 17, dtype=id_dtype)
        ids = cpu_ids.to("npu")[:, :5]
        packed = torch.zeros((32, 4 * 256 * 33 // 32), dtype=torch.uint8, device="npu")
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
                output_codes=codes[:17].view(68, 256),
                output_scales=row_scales[:17],
            )

        def project():
            return scheme.apply(
                layer,
                (codes[:17].view(torch.float8_e4m3fn), row_scales[:17].view(17, -1, 2).view(torch.float8_e8m0fnu)),
            )

        def check(actual):
            selected = ids.cpu()[:, 1:5].long().flatten()
            owned = (selected >= 100) & (selected < 356)
            local = torch.where(owned, selected - 100, 0)
            expected_codes, expected_scales = raw[local].clone(), cpu_scales[local].clone()
            expected_codes[~owned] = 0
            expected_scales[~owned] = 0
            assert torch.equal(codes[:17].cpu(), expected_codes.view(17, -1))
            assert torch.equal(row_scales[:17].cpu(), expected_scales.view(17, -1))
            reference = torch_npu.npu_quant_matmul(
                expected_codes.to("npu").view(17, -1).view(torch.float8_e4m3fn),
                layer.weight,
                layer.weight_scale,
                scale_dtype=torch_npu.float8_e8m0fnu,
                pertoken_scale=expected_scales.to("npu").view(17, -1, 2),
                pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
                output_dtype=torch.bfloat16,
                group_sizes=[1, 1, 32],
            )
            assert torch.equal(actual.cpu(), reference.cpu())
            assert torch.isfinite(actual.cpu()).all()
            assert not codes[17:].cpu().any() and not row_scales[17:].cpu().any()

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
        with pytest.raises(ValueError, match="head range"):
            lookup.lookup_codes(
                source,
                scales,
                ids,
                head_start=2,
                local_heads=4,
                output_codes=codes[:17].view(68, 256),
                output_scales=row_scales[:17],
            )
    finally:
        torch.npu.synchronize()
        del graph
        lookup.close()
        lookup.close()
        assert lookup.state_slot.cpu().item() == 0
        source.close()
        source.close()
