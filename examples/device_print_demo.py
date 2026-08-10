import torch
import torch.nn as nn

from vllm_ascend.utils import device_print


def compute_and_print(x: torch.Tensor) -> torch.Tensor:
    y = torch.square(x) - torch.cos(x)
    # Text / scalar prints take a carrier tensor and return it (FX dataflow).
    y = device_print("device_print from current execution mode", y)
    y = device_print(7, y)
    y = device_print(True, y)
    y = device_print(y)
    y = device_print(f"Compatible with f-strings: {x.dtype = }, {isinstance(x, torch.Tensor) = }", y)
    return y


class LinearWithDevicePrint(nn.Module):
    """Linear-like module used to show device_print inside a compiled layer."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.linear(x)
        y = device_print("device_print inside Linear.forward", y)
        return device_print(y)


def _run_eager_compile_graph(name: str, fn, x: torch.Tensor, modified_x: torch.Tensor) -> None:
    print(f"=== {name}: eager ===", flush=True)
    eager_out = fn(x)
    torch.npu.synchronize()

    print(f"=== {name}: torch.compile(backend='aot_eager') ===", flush=True)
    compiled_fn = torch.compile(fn, backend="aot_eager")
    compiled_out = compiled_fn(x)
    torch.npu.synchronize()
    assert torch.allclose(eager_out, compiled_out), f"{name}: eager vs compiled outputs mismatch."

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    x_capture = x.clone()

    with torch.npu.stream(capture_stream), torch.npu.graph(graph, stream=capture_stream):
        captured_out = compiled_fn(x_capture)

    print(f"=== {name}: replay graph ===", flush=True)
    graph.replay()
    torch.npu.synchronize()
    assert torch.allclose(eager_out, captured_out), f"{name}: eager vs graph outputs mismatch."

    print(f"=== {name}: modify input and replay graph ===", flush=True)
    x_capture.copy_(modified_x)
    graph.replay()
    torch.npu.synchronize()
    assert not torch.allclose(eager_out, captured_out), (
        f"{name}: outputs should change after modifying graph inputs."
    )


def main() -> None:
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)

    x = torch.arange(1, 28, dtype=torch.float32).reshape(3, 3, 3).npu()
    x_modified = torch.arange(28, 1, -1, dtype=torch.float32).reshape(3, 3, 3).npu()
    _run_eager_compile_graph("compute_and_print", compute_and_print, x, x_modified)

    # Prove device_print works inside Linear under compile + NPUGraph capture.
    linear_in, linear_out = 8, 4
    model = LinearWithDevicePrint(linear_in, linear_out).npu()
    linear_x = torch.randn(2, linear_in, dtype=torch.float32).npu()
    linear_x_modified = torch.randn_like(linear_x)
    _run_eager_compile_graph("linear_with_device_print", model, linear_x, linear_x_modified)

    print("All device_print demos passed.", flush=True)


if __name__ == "__main__":
    main()
