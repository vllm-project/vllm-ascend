import torch
import torch_npu
import torchair
from worm_server._ops_loader import ensure_ops_loaded

ensure_ops_loaded()


class IPCSafeAllocator:
    def __init__(self, dtype_str="float16", device_id=0):
        self.dtype_str = dtype_str
        self.device_id = device_id
        self.device = torch.device(f"npu:{self.device_id}")
        self._original_ones = torch.ones
        self._original_empty = torch.empty
        self._original_zeros = torch.zeros
        self._original_full = torch.full
        self.max_tensor_elements = float("inf")
        self._allocated_ptrs = set()
        self.set_current_device(self.device_id)

    @property
    def current_device(self):
        return torch_npu.npu.current_device()

    def set_current_device(self, device_id):
        if self.current_device != device_id:
            torch_npu.npu.set_device(f"npu:{device_id}")

    def create_tensor(self, shape, dtype_str, ptr):
        dtype = eval(dtype_str)
        tensors = torchair.llm_datadist.create_npu_tensors(shape, dtype, [ptr])
        return tensors[0]

    def allocate_tensor(self, shape, dtype_str, device_id=None):
        self.set_current_device(device_id)
        ret_code, ptr = torch.ops.tensor_ipc_utils.allocate_ipc_safe_tensor(shape, dtype_str)
        if ret_code != 0:
            raise RuntimeError(f"Failed to allocate tensor: {shape} {dtype_str}")
        self._allocated_ptrs.add(int(ptr))
        return self.create_tensor(shape, f"torch.{dtype_str}", ptr)

    def deallocate_all(self):
        for ptr in list(self._allocated_ptrs):
            torch.ops.tensor_ipc_utils.free_ipc_safe_tensor(ptr)
            self._allocated_ptrs.remove(ptr)

    def _flatten_shape(self, shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = list(shape[0])
        shape = list(shape)
        shape = [int(s) if isinstance(s, torch.Tensor) else s for s in shape]
        return shape

    def _normalize_dtype(self, dtype):
        if dtype is None:
            dtype_str = self.dtype_str  # default override
        elif isinstance(dtype, torch.dtype):
            dtype_str = str(dtype).replace("torch.", "")
        elif isinstance(dtype, str):
            dtype_str = dtype
        return dtype_str

    def _normalize_device(self, device):
        if device is None:
            # print(f"[Override] Redirecting device None → {self.device}")
            device = self.device
        elif isinstance(device, str) and device == "cpu" or isinstance(device, torch.device) and device.type == "cpu":
            # print(f"[Override] Redirecting device 'cpu' → {self.device}")
            device = self.device
        elif isinstance(device, str) and device != "cpu":
            device = torch.device(device)
        return device

    def _preprocess(self, shape, dtype=None, device=None, **kwargs):
        new_shape = self._flatten_shape(shape)
        device = self._normalize_device(device)
        dtype_str = self._normalize_dtype(dtype)
        return new_shape, device, dtype_str

    def _custom_allocate(self, shape, dtype=None, device=None, **kwargs):
        shape, device, dtype_str = self._preprocess(shape, dtype=dtype, device=device, **kwargs)
        tensor = self.allocate_tensor(shape, dtype_str, device_id=device.index)
        return tensor

    def _custom_ones(self, *shape, dtype=None, device=None, **kwargs):
        # print(f"[Override] torch.ones intercepted -> shape={shape}, dtype={dtype}, device={device}")
        tensor = self._custom_allocate(shape, dtype, device, **kwargs)
        return tensor.fill_(1)

    def _custom_empty(self, *shape, dtype=None, device=None, **kwargs):
        # print(f"[Override] torch.empty intercepted -> shape={shape}, dtype={dtype}, device={device}")
        tensor = self._custom_allocate(shape, dtype, device, **kwargs)
        return tensor

    def _custom_zeros(self, *shape, dtype=None, device=None, **kwargs):
        # print(f"[Override] torch.zeros intercepted -> shape={shape}, dtype={dtype}, device={device}")
        tensor = self._custom_allocate(shape, dtype, device, **kwargs)
        return tensor.fill_(0)

    def _log_only_full(self, *args, **kwargs):
        return self._original_full(*args, **kwargs)

    def __enter__(self):
        torch.ones = self._custom_ones
        torch.empty = self._custom_empty
        torch.zeros = self._custom_zeros
        torch.full = self._log_only_full

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.ones = self._original_ones
        torch.empty = self._original_empty
        torch.zeros = self._original_zeros
        torch.full = self._original_full


if __name__ == "__main__":
    shape = [32000, 4096, 50]
    dtype_str = "float16"

    IPCSafeAllocator = IPCSafeAllocator(dtype_str=dtype_str, device_id=0)
    with IPCSafeAllocator:
        tensor = torch.zeros(shape)
        print(tensor.shape, tensor.dtype, tensor.device)
    input("IPCSafeAllocator.deallocate_all()")
    IPCSafeAllocator.deallocate_all()
    input("Press any key to exit")
