import torch
import torch_npu


torch.npu.set_device(0)
m, k, n, groups = 512, 2048, 16, 8
x = torch.randn((m, k), dtype=torch.float16, device="npu")
weight = torch.randn((groups, n, k), dtype=torch.float16, device="npu")
weight_kn = weight.transpose(1, 2)
counts = torch.zeros((groups,), dtype=torch.int64, device="npu")
counts[0] = m


def run():
    return torch_npu.npu_grouped_matmul(
        [x],
        [weight_kn],
        group_list=counts,
        split_item=3,
        group_type=0,
        group_list_type=1,
    )[0]


eager = run()
torch.npu.synchronize()
graph = torch.npu.NPUGraph()
with torch.npu.graph(
    graph, capture_error_mode="thread_local", auto_dispatch_capture=True
):
    captured = run()
for _ in range(10):
    graph.replay()
torch.npu.synchronize()
print({"max_abs": (captured.float() - eager.float()).abs().max().cpu().item()})
