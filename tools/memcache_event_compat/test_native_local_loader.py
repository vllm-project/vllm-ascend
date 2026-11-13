import ctypes
import pathlib

lib = ctypes.CDLL(str(pathlib.Path(__file__).with_name("test_native_local_loader.so")), mode=ctypes.RTLD_LOCAL)
lib.check_hash.argtypes = [ctypes.c_char_p]
lib.check_hash.restype = ctypes.c_uint64
h = "3bb61b5f4f104508bfce20107f10324ef835fff2a6bf10015b8fcea4db906503"
expected = int(h, 16) & ((1 << 64) - 1)
for key in [
    h,
    "model@" + h,
    "model@0@" + h + "@0",
    "model@0@" + h + "@3",
    "model@group:0@layer_id:3@" + h,
    "dspark-prefix-v1@identity@pp0@tp0@" + h,
]:
    assert lib.check_hash(key.encode()) == expected, key
for key in ["", "model@partial@request@0@0@127@0", "model@not-a-hash"]:
    assert lib.check_hash(key.encode()) == (1 << 64) - 1, key
print("Python RTLD_LOCAL regression: 9 cases passed", flush=True)
