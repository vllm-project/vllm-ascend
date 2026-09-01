import glob

import torch
import torch_npu


def make_perm(rows):
    order = torch.tensor(
        [5, 0, 12, 3, 8, 1, 15, 7, 9, 2, 14, 6, 13, 4, 11, 10],
        dtype=torch.int32,
    )
    inactive = {2, 7, 13}
    encoded = order.clone()
    for pos, row in enumerate(order.tolist()):
        if row in inactive:
            encoded[pos] = -(row + 1)
    perm = torch.zeros((rows, 8), dtype=torch.int32)
    perm[:, 0] = encoded
    return order, encoded, perm, inactive


def run(dtype, width, output_offset):
    rows = 16
    output_width = output_offset + width + 7
    order, encoded, perm_cpu, inactive = make_perm(rows)
    source_cpu = (
        torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) / 29
    ).to(dtype)
    grouped = torch.full((rows, width), -7, dtype=dtype, device="npu")
    torch.ops._C_ascend.moe_lora_prefill_gather_by_perm(
        source_cpu.npu(), perm_cpu.npu(), grouped
    )
    torch.npu.synchronize()
    ref_grouped = source_cpu[order.to(torch.int64)]
    torch.testing.assert_close(grouped.cpu(), ref_grouped, rtol=0, atol=0)

    base_cpu = torch.linspace(
        -3, 3, rows * output_width, dtype=torch.float32
    ).reshape(rows, output_width).to(dtype)
    delta_cpu = torch.empty((rows, width), dtype=dtype)
    delta_cpu[0::3] = torch.tensor(1 / 128, dtype=dtype)
    delta_cpu[1::3] = torch.tensor(1 / 1024, dtype=dtype)
    delta_cpu[2::3] = torch.tensor(1 / 4096, dtype=dtype)
    y = base_cpu.npu()
    torch.ops._C_ascend.moe_lora_prefill_scatter_add(
        delta_cpu.npu(), perm_cpu.npu(), y, output_offset
    )
    torch.npu.synchronize()
    expected = base_cpu.clone()
    for pos, value in enumerate(encoded.tolist()):
        if value >= 0:
            expected[value, output_offset : output_offset + width] = (
                base_cpu[value, output_offset : output_offset + width].float()
                + delta_cpu[pos].float()
            ).to(dtype)
    torch.testing.assert_close(y.cpu(), expected, rtol=0, atol=0)
    for row in inactive:
        torch.testing.assert_close(y[row].cpu(), base_cpu[row], rtol=0, atol=0)
    torch.testing.assert_close(
        y[:, :output_offset].cpu(), base_cpu[:, :output_offset], rtol=0, atol=0
    )
    torch.testing.assert_close(
        y[:, output_offset + width :].cpu(),
        base_cpu[:, output_offset + width :],
        rtol=0,
        atol=0,
    )
    print(
        "PHASE2 PASS",
        dtype,
        "width=",
        width,
        "offset=",
        output_offset,
        "inactive=",
        sorted(inactive),
    )


if __name__ == "__main__":
    torch.npu.set_device(0)
    so = glob.glob(
        "/home/l00832868/codexWork/vllm-ascend/build_kernel/vllm_ascend_C*.so"
    )[0]
    torch.ops.load_library(so)
    for data_dtype in (torch.float16, torch.bfloat16):
        run(data_dtype, 33, 5)
        run(data_dtype, 4097, 3)
