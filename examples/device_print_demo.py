import torch

from vllm_ascend.utils import device_print


def compute_and_print(x: torch.Tensor) -> torch.Tensor:
    y = torch.square(x) - torch.cos(x)
    device_print("device_print from current execution mode")
    device_print(7)
    device_print(True)
    device_print(y)
    device_print(f"Compatible with f-strings: {x.dtype = }, {isinstance(x, torch.Tensor) = }")
    return y


def main() -> None:
    torch.npu.set_device(0)
    torch.npu.set_compile_mode(jit_compile=False)

    x = torch.arange(1, 28, dtype=torch.float32).reshape(3, 3, 3).npu()

    print("=== eager ===", flush=True)
    eager_out = compute_and_print(x)
    torch.npu.synchronize()

    print("=== torch.compile(backend='aot_eager') ===", flush=True)
    compiled_compute_and_print = torch.compile(compute_and_print, backend="aot_eager")
    compiled_out = compiled_compute_and_print(x)
    torch.npu.synchronize()

    assert torch.allclose(eager_out, compiled_out), "Outputs from eager and compiled modes do not match."

    graph = torch.npu.NPUGraph()
    capture_stream = torch.npu.Stream()
    x_capture = x.clone()

    with torch.npu.stream(capture_stream), torch.npu.graph(graph, stream=capture_stream):
        captured_out = compiled_compute_and_print(x_capture)

    print("=== replay graph ===", flush=True)
    graph.replay()
    torch.npu.synchronize()

    assert torch.allclose(eager_out, captured_out), "Outputs from eager and graph modes do not match."

    print("=== modify input and replay graph ===", flush=True)
    x_capture.copy_(torch.arange(28, 1, -1, dtype=torch.float32).reshape(3, 3, 3).npu())
    graph.replay()
    torch.npu.synchronize()

    assert not torch.allclose(eager_out, captured_out), "Outputs from eager and modified graph modes should not match."


if __name__ == "__main__":
    main()
