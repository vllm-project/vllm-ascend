import sys

import torch
import torch_npu


m = int(sys.argv[1]) if len(sys.argv) > 1 else 512
dtype = getattr(torch, sys.argv[2]) if len(sys.argv) > 2 else torch.bfloat16
groups = 8
hidden = 4096
intermediate = 2048
rank = 16
torch.manual_seed(23)
torch.npu.set_device(0)


def randn(shape):
    return torch.randn(shape, dtype=dtype, device="npu") * 0.02


x_w13 = randn((m, hidden))
x_w2 = randn((m, intermediate))
a_w13_0 = randn((groups, rank, hidden))
a_w13_1 = randn((groups, rank, hidden))
b_w13_0 = randn((groups, intermediate, rank))
b_w13_1 = randn((groups, intermediate, rank))
a_w2 = randn((groups, rank, intermediate))
b_w2 = randn((groups, hidden, rank))

hotspot = torch.tensor([m] + [0] * (groups - 1), dtype=torch.int64, device="npu")
quotient, remainder = divmod(m, groups)
uniform = torch.tensor(
    [quotient + (group < remainder) for group in range(groups)],
    dtype=torch.int64,
    device="npu",
)
counts = hotspot.clone()


def gmm(inputs, weight):
    return torch_npu.npu_grouped_matmul(
        [inputs],
        [weight.transpose(1, 2)],
        group_list=counts,
        split_item=2,
        group_type=0,
        group_list_type=1,
    )[0]


def full_chain():
    shrink_0 = gmm(x_w13, a_w13_0)
    delta_0 = gmm(shrink_0, b_w13_0)
    shrink_1 = gmm(x_w13, a_w13_1)
    delta_1 = gmm(shrink_1, b_w13_1)
    shrink_2 = gmm(x_w2, a_w2)
    delta_2 = gmm(shrink_2, b_w2)
    return delta_0, delta_1, delta_2


# Warm up the exact API, shape and TensorList pattern before capture.
full_chain()
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(
    graph, capture_error_mode="thread_local", auto_dispatch_capture=True
):
    captured_outputs = full_chain()
torch.npu.synchronize()

count_data_ptr = counts.data_ptr()
output_data_ptrs = tuple(output.data_ptr() for output in captured_outputs)
allocated_before = torch.npu.memory_allocated(0)
reserved_before = torch.npu.memory_reserved(0)

for replay in range(100):
    counts.copy_(uniform if replay % 2 == 0 else hotspot)
    graph.replay()

# Finish with uniform counts and compare the captured graph against eager using
# the same Tensor address and values.
counts.copy_(uniform)
graph.replay()
torch.npu.synchronize()
allocated_after = torch.npu.memory_allocated(0)
reserved_after = torch.npu.memory_reserved(0)
replay_output_ptrs = tuple(output.data_ptr() for output in captured_outputs)
eager_uniform = full_chain()
torch.npu.synchronize()
errors = [
    (captured.float() - eager.float()).abs().max().cpu().item()
    for captured, eager in zip(captured_outputs, eager_uniform)
]

print(
    {
        "m": m,
        "dtype": str(dtype),
        "replays": 101,
        "count_data_ptr_stable": counts.data_ptr() == count_data_ptr,
        "output_data_ptr_stable": replay_output_ptrs == output_data_ptrs,
        "max_abs_vs_eager_uniform": errors,
        "allocated_before": allocated_before,
        "allocated_after": allocated_after,
        "allocated_growth": allocated_after - allocated_before,
        "reserved_before": reserved_before,
        "reserved_after": reserved_after,
        "reserved_growth": reserved_after - reserved_before,
    }
)
