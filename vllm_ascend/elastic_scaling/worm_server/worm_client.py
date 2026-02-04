import os
import pickle
import time

import torch_npu
import zmq
from worm_ipc.npu_ipc_core import NPUIPCCore, WORMBase


class WormNPUClient(WORMBase):
    def __init__(self, ipc_config, stamp=""):
        # ipc_config is the server IPC config to connect to
        self.ipc_core = NPUIPCCore(ipc_config)

        self.socket_path = f"/tmp/tensor_ipc_{self.ipc_core.dp_rank}_{self.ipc_core.tp_rank}{stamp}.sock"

        if not os.path.exists(self.socket_path):
            raise FileNotFoundError(f"Socket does not exist: {self.socket_path}")

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.connect(f"ipc://{self.socket_path}")
        print(f"[WormNPUClient] Connected to {self.socket_path}")
        start_time = time.time()
        self.handshake()
        print(f"Handshaking time {round(time.time() - start_time, 5)}s")

    def handshake(self):
        my_tgid = self.ipc_core.get_tgid()
        response = self.send_request("handshake", {"client_tgid": my_tgid, "client_device": self.ipc_core.device_id})
        server_tgid = response["server_tgid"]
        server_device = response["server_device"]
        if server_device != self.ipc_core.device_id:
            print(
                f"""\nWARNING: Cross Device Zero Copy Happening. 
                Server npu:{server_device} -> Client npu:{self.ipc_core.device_id}"""
            )
            # self.ipc_core.initialize_context(server_device)
        self.ipc_core.set_tgid([server_tgid])

    def send_request(self, end_point_name, end_point_args=None):
        if end_point_args is None:
            end_point_args = {}
        self.socket.send(pickle.dumps({"cmd": end_point_name, **end_point_args}))
        return pickle.loads(self.socket.recv())

    def zero_copy(self, name):
        response = self.send_request(end_point_name="zero_copy", end_point_args={"param_name": name})
        if response is not None and response["whitelist_status"]:
            tensor = self.ipc_core.open_tensor(response)
        else:
            tensor = None
        return tensor, response

    def zero_copy_model(self, model, is_clone=False):
        zero_copy_failed = []
        responses = self.send_request("zero_copy_batch")

        for param_name, param in model.named_parameters():
            handle_data = responses[param_name]
            if handle_data is None:
                zero_copy_failed.append(param_name)
                print(f"[WormNPUClient] Zero-copy failed for {param_name}. Could not open {handle_data}")
                continue

            tensor = self.ipc_core.open_tensor(handle_data)
            if tensor is not None:
                if is_clone:
                    param.data = tensor.clone()
                else:
                    param.data = tensor
                print(f"[WormNPUClient] Zero-copy success for {param_name}.")
            else:
                zero_copy_failed.append(param_name)
                print(f"[WormNPUClient] Zero-copy failed for {param_name}. Sum did not match {handle_data}")

        num_tensors = sum(1 for _ in model.named_parameters())
        print(
            f"""[WormNPUClient] | DP {self.ipc_core.dp_rank} TP {self.ipc_core.tp_rank} | Done loading model. 
            NCCL fallback: {len(zero_copy_failed)}/{num_tensors} ({100 * len(zero_copy_failed) / num_tensors:.2f}%)"""
        )

    def zero_copy_kv_caches(self):
        kv_cache_handles = self.send_request("zero_copy_kv_handles")
        kv_caches = {}

        for cache_key, handle_info in kv_cache_handles.items():
            if isinstance(handle_info, tuple):
                cache_tensor0 = self.ipc_core.open_tensor(handle_info[0])
                cache_tensor1 = self.ipc_core.open_tensor(handle_info[1])
                kv_caches[cache_key] = (cache_tensor0, cache_tensor1)
            else:
                cache_tensor = self.ipc_core.open_tensor(handle_info)
                kv_caches[cache_key] = cache_tensor
        return kv_caches

    def npu_grouped_matmul(self):
        response = self.send_request(end_point_name="npu_grouped_matmul")[0]
        if response is not None:
            tensor = self.ipc_core.open_tensor(response)
        else:
            tensor = None

        tensor, response = ipc_engine.npu_grouped_matmul()
        print(response)
        print(tensor)
        print(tensor.is_contiguous())
        import pickle

        with open("simar.pkl", "rb") as f:
            data = pickle.load(f)
        (sorted_hidden_states, w1, expert_tokens) = data
        sorted_hidden_states = sorted_hidden_states.npu()
        expert_tokens = expert_tokens.npu()

        w1 = tensor.npu().contiguous()
        gate_up_out_list = torch_npu.npu_grouped_matmul(
            x=[sorted_hidden_states],
            weight=[w1],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
        )
        print(gate_up_out_list)
        exit()


if __name__ == "__main__":
    import time

    ipc_config = {
        "dp_rank": 0,
        "dp_size": 2,
        "tp_rank": 0,
        "tp_size": 2,
        "device_id": 0,
    }

    ipc_engine = WormNPUClient(ipc_config, stamp="test")
    tensor, response = ipc_engine.zero_copy("sample")
    print(f"Received tensor- {tensor}")
