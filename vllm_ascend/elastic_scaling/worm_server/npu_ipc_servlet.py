import zmq
import os
import pickle
import torch
import threading
from worm_ipc.npu_ipc_core import NPUIPCCore, WORMBase


class NPUIPCServlet(WORMBase):
    def __init__(self, ipc_config, stamp=""):
        self.ipc_core = NPUIPCCore(ipc_config)
        self.ipc_config = ipc_config

        self.socket_path = f"/tmp/tensor_ipc_{self.ipc_core.dp_rank}_{self.ipc_core.tp_rank}{stamp}.sock"

        if os.path.exists(self.socket_path):
            os.remove(self.socket_path)

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REP)
        self.socket.bind(f"ipc://{self.socket_path}")
        print(f"[NPUIPCServlet] Bound to {self.socket_path}")

        self.named_parameters = {}
        self.kv_caches = {}
        self.run_non_blocking()

    def run_non_blocking(self):
        thread = threading.Thread(target=self.run, daemon=True)
        thread.start()

    def run(self):
        while True:
            msg = self.socket.recv()
            request = pickle.loads(msg)
            response = self.handle_request(request)
            self.socket.send(pickle.dumps(response))

    def set_model_params(self, named_parameters_dict):
        self.named_parameters = named_parameters_dict
        # # Preprocess and cache handles
        new_allocations = []
        for param_name, param in self.named_parameters.items():
            handle_info = self.ipc_core.export_handle(param, param_name)
            if handle_info is None: new_allocations.append(param_name)
        print(f'[NPUIPCServlet] Model Parameters Handle Info Cached Successfully. New allocations - {len(new_allocations)}/{len(self.named_parameters)}={round(len(new_allocations)/len(self.named_parameters)*100,2)}%')

    def set_kv_caches(self, kv_caches):
        self.kv_caches = kv_caches
        for cache_key, cache_tensor in self.kv_caches.items():
            if isinstance(cache_tensor, tuple):
                self.ipc_core.export_handle(cache_tensor[0], f"kv0_{cache_key}")
                self.ipc_core.export_handle(cache_tensor[1], f"kv1_{cache_key}")
            else:
                self.ipc_core.export_handle(cache_tensor, f"kv_{cache_key}")

    def handshake(self, client_tgid, client_device):
        server_tgid = self.ipc_core.get_tgid()
        self.ipc_core.set_tgid([client_tgid])
        self.ipc_core.current_device(self.ipc_core.device_id, force=True)
        whitelist_success, whitelist_failed = self.ipc_core.whitelist_cached_handles()
        print(f'\nWhitelist completed. TGIDs {self.ipc_core.whitelisted_clients} Failed percentage {len(whitelist_failed)}/{len(whitelist_success)+len(whitelist_failed)} = {round(len(whitelist_failed)/(len(whitelist_success)+len(whitelist_failed))*100,2)}%')
        return server_tgid

    def handle_request(self, request):
        cmd = request.pop("cmd")

        if cmd == "handshake":
            print(f'tgid {request["client_tgid"]}')
            print(f'client_device {request["client_device"]}')
            server_tgid = self.handshake(request["client_tgid"], request["client_device"])
            return {"server_tgid": server_tgid,
                    "server_device": self.ipc_core.device_id}

        elif cmd == "zero_copy":
            param_name = request["param_name"]
            return self.ipc_core.export_handle(None, param_name) # assumes its already cached

        elif cmd == "zero_copy_batch":
            return {
                name: self.ipc_core.export_handle(tensor, name)
                for name, tensor in self.named_parameters.items()
            }

        elif cmd == "zero_copy_kv_handles":
            kv_cache_handles = {}
            for cache_key, cache_tensor in self.kv_caches.items():
                if isinstance(cache_tensor, tuple):
                    kv_cache_handles[cache_key] = (
                        self.ipc_core.export_handle(cache_tensor[0], f"kv0_{cache_key}"),
                        self.ipc_core.export_handle(cache_tensor[1], f"kv1_{cache_key}")
                    )
                else:
                    kv_cache_handles[cache_key] = self.ipc_core.export_handle(cache_tensor, f"kv_{cache_key}")
            return kv_cache_handles
        
        else:
            print(f'[NPUIPCServlet] Unknown command received! {cmd}')
            return {"error": f"Unknown command: {cmd}"}

    def reset(self):
        self.named_parameters = {}
        self.kv_caches = {}
        self.ipc_core.reset()

if __name__=='__main__':
    ipc_config = {
        'dp_rank': 0,
        'dp_size': 2,
        'tp_rank': 0,
        'tp_size': 2,
        'device_id': 0,
    }
    ipc_engine = NPUIPCServlet(ipc_config, stamp='test')
    sample_tensor = torch.ones((5000,50000), dtype=torch.int8, device=f'npu:{ipc_config["device_id"]}')
    handle_info = ipc_engine.ipc_core.export_handle(sample_tensor, 'sample')
    print(handle_info, sample_tensor)
    input('press any key to exit')
