import os, torch
lib = os.path.join(os.path.dirname(__file__), 'elastic_ipc_utils.cpython-311-aarch64-linux-gnu.so')
torch.ops.load_library(lib) 