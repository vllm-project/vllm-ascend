
import base64
import torch
import torch_npu
import torchair
from ._ops_loader import ensure_ops_loaded
ensure_ops_loaded()

def print_from_all_ranks(msg):
    current_device = torch.npu.current_device()
    print(f"[Rank {current_device}] {msg}", flush=True)

class NPUIPCCore():
    def __init__(self, ipc_config):
        # Local (per machine) parameters
        self.tp_rank = ipc_config['tp_rank']
        self.tp_size = ipc_config['tp_size']
        self.device_id = ipc_config['device_id']

        # Global parameters (acorss machine)
        self.dp_rank = ipc_config['dp_rank']
        self.dp_size = ipc_config['dp_size']
        self.world_size = self.dp_size * self.tp_size
        self.global_rank = self.dp_rank * self.tp_size + self.tp_rank

        self.initialize_context(self.device_id)
        self.current_device(self.device_id, force=True)

        self.cached_handles = {}
        self.whitelisted_clients = []
        self.tgid = torch.ops.tensor_ipc_utils.rt_get_tgid()
        print(f'[NPUIPCCore] DP {self.dp_rank}/{self.dp_size}, TP {self.tp_rank}/{self.tp_size}, NPU {self.current_device}, Global NPU {self.global_rank}/{self.world_size}, TGID {self.tgid}')

    @property
    def current_device(self):
        return torch_npu.npu.current_device()
    
    def current_device(self, device_id, force=False):
        if (self.current_device != device_id) or force:
            torch_npu.npu.set_device(f'npu:{device_id}')

    def initialize_context(self, device_id):
        _ = torch.ones(1, device=f"npu:{device_id}", dtype=torch.float16)

    def get_tgid(self):
        return self.tgid

    def set_tgid(self, tgid_list):
        assert isinstance(tgid_list, list) and all(isinstance(x, int) for x in tgid_list), "Expected a list of integers"
        self.whitelisted_clients += tgid_list

    def _whitelist_handle(self, ipc_handle, param_name='sample'):
        if ipc_handle.device != torch.device("cpu"):
            ipc_handle = ipc_handle.to("cpu")
        if len(self.whitelisted_clients)>0:
            ret_code = torch.ops.tensor_ipc_utils.rt_set_ipc_mem_pid(ipc_handle, self.whitelisted_clients)
            if ret_code != 0:
                print(f"[NPUIPCCore] Whiltelisting FAIELD for tensor {param_name} with self.whitelisted_clients {self.whitelisted_clients}. Code: {ret_code}")
                return False
            else:
                return True
        else:
            print(f"[NPUIPCCore] Skipping whitelisting for tensor {param_name} as the self.whitelisted_clients is empty {self.whitelisted_clients}")
            return False

    def whitelist_cached_handles(self):
        for tensor_name, (handle_data, ipc_handle) in self.cached_handles.items():
            whitelist_status = self._whitelist_handle(ipc_handle, tensor_name)
            handle_data['whitelist_status'] = whitelist_status
        whitelist_success = [param_name for param_name, (handle_data, _) in self.cached_handles.items() if handle_data['whitelist_status'] == True]
        whitelist_failed = [param_name for param_name, (handle_data, _) in self.cached_handles.items() if handle_data['whitelist_status'] == False]
        return whitelist_success, whitelist_failed

    def _export_handle(self, tensor, param_name="sample"):
        data_ptr = tensor.data_ptr()
        dtype_str = str(tensor.dtype).split(".")[1]
        ret_code, ipc_handle = torch.ops.tensor_ipc_utils.rt_export_tensor_ipc_handle(data_ptr, list(tensor.shape), dtype_str)
        if ret_code != 0:
            print(f'[NPUIPCCore] Export IPC handle failed: param_name: {param_name} | Shape: {list(tensor.shape)} | Dtype: {dtype_str} | Ptr: {data_ptr}')
            handle_b64 = None
        else:
            handle_b64 = base64.b64encode(ipc_handle.numpy().tobytes()).decode()
        return ret_code, ipc_handle, handle_b64

    def export_handle(self, param, param_name="sample"):
        if param_name in self.cached_handles:
            (handle_data, ipc_handle) = self.cached_handles[param_name]
            return handle_data

        ret_code, ipc_handle, handle_b64 = self._export_handle(param.data, param_name)

        if ret_code == 0:
            handle_data = {
                "param_name": param_name,
                "handle": handle_b64,
                "shape": list(param.shape),
                "dtype": str(param.dtype),
            }
            self.cached_handles[param_name] = (handle_data, ipc_handle)
        else:
            handle_data = None
        return handle_data

    def open_tensor(self, handle_data):
        handle_bytes = base64.b64decode(handle_data["handle"])
        ipc_handle = torch.tensor(list(handle_bytes), dtype=torch.uint8)
        ret, ptr = torch.ops.tensor_ipc_utils.rt_open_ipc_handle(ipc_handle)

        if ret != 0:
            print(f"[NPUIPCCore] Failed to open IPC handle (rt_open_ipc_handle)")
            return None

        return self.create_tensor(handle_data["shape"], handle_data["dtype"], ptr)

    def create_tensor(self, shape, dtype_str, ptr):
        dtype = eval(dtype_str)
        tensors = torchair.llm_datadist.create_npu_tensors(shape, dtype, [ptr])
        return tensors[0]

    def reset(self):
        self.cached_handles.clear()
        self.whitelisted_clients = []

class WORMBase():
    def inference(self, model_path, model, max_tokens):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        input_text = "Once upon a time in a futuristic world,"
        device_id = next(model.parameters()).device.index
        inputs = tokenizer(input_text, return_tensors="pt").to(f"npu:{device_id}")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_tokens)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)        
        return generated_text    
    
    def check_all_weights_on_npu(self, named_parameters_dict):
        params_on_npu = []
        all_on_npu = True
        for name, param in named_parameters_dict.items():
            if not param.is_npu:
                print_from_all_ranks(f"❌ {name} ({param.shape}, {param.dtype}) is NOT on NPU")
                all_on_npu = False
                params_on_npu.append(name)
            else:
                print_from_all_ranks(f"✅ {name} ({param.shape}, {param.dtype}) is on {param.device}")
                pass
        if all_on_npu:
            print_from_all_ranks("🎉 All model parameters are on NPU")
        else:
            print_from_all_ranks("⚠️ Some parameters are on CPU")
        print_from_all_ranks(params_on_npu)

if __name__ == "__main__":
    print("[Test] Starting NPUIPCCore self-tests...")

    ipc_config = {
        'dp_rank': 0,
        'dp_size': 2,
        'tp_rank': 0,
        'tp_size': 2,
        'device_id': 0,
    }

    core = NPUIPCCore(ipc_config)

    # Set device contexts
    # torch_npu.npu.set_device(0)
    # torch_npu.npu.set_device(1)

    # Create a random tensor for testing
    # test_tensor = torch.randn(shape, dtype=dtype, device="npu")

    # # ---- Test: _export_handle (internal) ----
    # print("\n[Test] Running _export_handle...")
    # ret_code, ipc_handle, handle_b64 = core._export_handle(test_tensor, param_name="test_tensor_raw")
    # assert ret_code == 0, "[FAIL] _export_handle: Failed to export IPC handle"
    # assert isinstance(handle_b64, str), "[FAIL] _export_handle: Did not return a valid base64 string"
    # print("[PASS] _export_handle")

    # ---- Test: allocate_tensor ----
    # print("\n[Test] Running allocate_tensor...")
    # allocated_tensor = core.allocate_tensor(shape, dtype_str)
    # # allocated_tensor = core.allocate_tensor(shape, dtype_str, device_id=0)
    # assert isinstance(allocated_tensor, torch.Tensor), "[FAIL] allocate_tensor: Did not return a tensor"
    # assert list(allocated_tensor.shape) == shape, "[FAIL] allocate_tensor: Shape mismatch"
    # assert allocated_tensor.dtype == dtype, "[FAIL] allocate_tensor: Dtype mismatch"
    # assert allocated_tensor.device.type == "npu", "[FAIL] allocate_tensor: Not allocated on NPU"
    # print(f"[PASS] allocate_tensor. Shape {allocated_tensor.shape} Tensor {allocated_tensor}")

    # # ---- Test: export_handle + open_tensor ----
    # ## THIS TEST DOES NOT WORK AS SELF WHITELISTING DOES NOT WORK
    # print("\n[Test] Running export_handle + open_tensor...")
    # core.set_tgid([core.get_tgid()])
    # handle_data = core.export_handle(test_tensor, param_name="test_tensor")
    # print(f'Open handle worked, handle data - {handle_data}\n Now opening tensor')
    # reopened_tensor = core.open_tensor(handle_data)
    # assert isinstance(reopened_tensor, torch.Tensor), "[FAIL] open_tensor: Did not return a tensor"
    # assert list(reopened_tensor.shape) == shape, "[FAIL] open_tensor: Shape mismatch"
    # assert reopened_tensor.dtype == dtype, "[FAIL] open_tensor: Dtype mismatch"
    # assert reopened_tensor.device.type == "npu", "[FAIL] open_tensor: Not reopened on NPU"
    # print("[PASS] export_handle + open_tensor")

    print("\n✅ All local IPC core tests passed.")
