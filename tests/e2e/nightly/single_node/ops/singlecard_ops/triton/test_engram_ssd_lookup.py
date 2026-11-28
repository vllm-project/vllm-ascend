# SPDX-License-Identifier: Apache-2.0
"""Real SSD/libaio/UVA checks; requires an NPU and the native SSD library."""

from contextlib import ExitStack

import numpy as np
import pytest
import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.deepseek_v41.engram.npu import HostUvaBuffer, gather_dequantize_host_uva
from vllm_ascend.models.deepseek_v41.engram.ssd.engine import SSDFillEngine, SSDReadError, SSDInputError, plan_row_extents
from vllm_ascend.models.deepseek_v41.engram.ssd.planes import SSDPlanes


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_ssd_fill_uva_output(tmp_path, dtype):
    if dtype == torch.float8_e4m3fn and not get_current_hardware_profile().supports(HardwareCapability.ENGRAM_MXFP8):
        pytest.skip("Native FP8 requires ENGRAM_MXFP8")
    torch.npu.set_device(0)
    torch.zeros(1, device="npu")
    # Include all byte encodings, NaNs, signed zero and FP8 subnormals.
    codes = np.stack([np.roll(np.arange(256, dtype=np.uint8), row * 7) for row in range(17)])
    fp8 = dtype == torch.float8_e4m3fn
    row_bytes = 264 if fp8 else 288
    if fp8:
        scales = np.tile(np.arange(123, 131, dtype=np.uint8), (17, 1))
        scale_values = torch.pow(2.0, torch.from_numpy(scales).float() - 127)
    else:
        scales = np.tile(np.array([0.25, 0.5, 1, 2, 4, 8, 0.125, 1], np.float32), (17, 1))
        scales *= (1 + np.arange(17, dtype=np.float32)[:, None] / 16)
        scale_values = torch.from_numpy(scales)
    packed = np.zeros((4096, row_bytes), np.uint8)
    packed[:17, :256] = codes
    packed[:17, 256:] = scales.view(np.uint8)
    table = tmp_path / "table.bin"
    packed.tofile(table)
    with ExitStack() as cleanup:
        planes = SSDPlanes(32, "npu", codes_dtype=dtype)
        cleanup.callback(planes.close)
        codes_np, scales_np = planes.numpy_views()
        engine = SSDFillEngine([str(table)], layer_rows=[17], layer_bases=[0],
                              threads_per_disk=2, qd=8, row_bytes=row_bytes)
        cleanup.callback(engine.close)
        # FP8 row 15 and INT8 row 14 cross a 4 KiB boundary.
        ids = np.array([16, 0, 15, 14, 16], np.uint64)
        slots = np.array([4, 7, 1, 9, 2], np.int32)
        result = engine.fill(ids, slots, codes_np, scales_np)
        assert result["rows"] == len(ids)
        assert engine.stats()["rows"] == len(ids)
        np.testing.assert_array_equal(codes_np[slots], codes[ids])
        np.testing.assert_array_equal(scales_np[slots], scales[ids].view(np.uint8))
        # Extent I/O must use the same native row stride and scale bytes.
        codes_np.fill(0)
        scales_np.fill(0)
        extents, scatters, _ = plan_row_extents(ids, slots, row_bytes=row_bytes)
        result = engine.read_extents(extents, scatters, codes_np, scales_np)
        assert result["rows"] == len(ids)
        assert engine.stats()["ge_events"] == len(extents)
        np.testing.assert_array_equal(codes_np[slots], codes[ids])
        np.testing.assert_array_equal(scales_np[slots], scales[ids].view(np.uint8))
        # Use the unchanged DDR UVA API: typed planes determine the encoding.
        lookup = torch.tensor([[99, 4, 7, 99], [99, 1, 9, 99], [99, 2, -1, 99]], device="npu")
        out = torch.full((9, 256), 42, device="npu", dtype=torch.bfloat16)
        actual = gather_dequantize_host_uva(
            planes.codes, planes.scales, lookup, head_start=1, local_heads=2,
            pad_heads=3, output=out).cpu().reshape(3, 3, 256)
        expected = (torch.from_numpy(codes).view(dtype).float().reshape(17, 8, 32)
                    * scale_values.unsqueeze(-1)).reshape(17, 256).bfloat16()
        selected = expected[torch.tensor([16, 0, 15, 14, 16])].reshape(5, 256)
        torch.testing.assert_close(actual[:, :2].reshape(6, 256)[:5], selected, rtol=0, atol=0, equal_nan=True)
        assert torch.count_nonzero(actual[2, 1]) == 0
        assert torch.all(actual[:, 2] == 42)
        with pytest.raises(SSDReadError):
            engine.fill(np.array([17], np.uint64), np.array([0], np.int32), codes_np, scales_np)
        with pytest.raises(SSDInputError):
            engine.fill(ids, slots, codes_np, np.zeros((32, 16), np.uint8))
        with pytest.raises(SSDInputError):
            engine.read_extents(extents, scatters, codes_np, np.zeros((32, 16), np.uint8))


def test_ddr_mxfp8_e8m0_default_is_preserved():
    if not get_current_hardware_profile().supports(HardwareCapability.ENGRAM_MXFP8):
        pytest.skip("Native FP8 requires ENGRAM_MXFP8")
    torch.npu.set_device(0)
    torch.zeros(1, device="npu")
    module = torch.nn.Module()
    with ExitStack() as cleanup:
        c = HostUvaBuffer((2, 256), torch.float8_e4m3fn, "npu", "codes", module)
        cleanup.callback(c.close)
        s = HostUvaBuffer((2, 8), torch.uint8, "npu", "scales", module)
        cleanup.callback(s.close)
        c.tensor.view(torch.uint8).copy_(torch.arange(256, dtype=torch.uint8).expand(2, -1))
        s.tensor.copy_(torch.tensor([[124, 125, 126, 127, 128, 129, 130, 131]] * 2, dtype=torch.uint8))
        actual = gather_dequantize_host_uva(c, s, torch.tensor([1, 0], device="npu")).cpu()
        scale = torch.pow(2.0, s.tensor.float() - 127)
        expected = (c.tensor.float().reshape(2, 8, 32) * scale.unsqueeze(-1)).reshape(2, 256).bfloat16()
        torch.testing.assert_close(actual, expected.flip(0), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("row_bytes", [264, 288])
def test_dual_disk_native_layout(tmp_path, row_bytes):
    paths = [tmp_path / "l1.bin", tmp_path / "l14.bin"]
    expected = []
    for layer, path in enumerate(paths):
        rows = np.zeros((4096, row_bytes), np.uint8)
        rows[:17] = (np.arange(17 * row_bytes).reshape(17, row_bytes) + layer * 31).astype(np.uint8)
        rows.tofile(path)
        expected.append(rows)
    ids = np.array([15, 4096 + 15, 16, 4096, 0], np.uint64)
    slots = np.arange(5, dtype=np.int32)
    codes = np.zeros((5, 256), np.uint8)
    scales = np.zeros((5, row_bytes - 256), np.uint8)
    gold = np.stack([expected[int(i >= 4096)][int(i % 4096)] for i in ids])
    with SSDFillEngine([str(p) for p in paths], layer_rows=[17, 17], layer_bases=[0, 4096],
                       threads_per_disk=1, qd=4, row_bytes=row_bytes) as engine:
        assert engine.fill(ids, slots, codes, scales)["rows"] == len(ids)
        np.testing.assert_array_equal(np.concatenate((codes, scales), axis=1), gold)
        codes.fill(0)
        scales.fill(0)
        extents, scatters, _ = plan_row_extents(ids, slots, base1=4096, dual=True, row_bytes=row_bytes)
        assert engine.read_extents(extents, scatters, codes, scales)["rows"] == len(ids)
        np.testing.assert_array_equal(np.concatenate((codes, scales), axis=1), gold)
