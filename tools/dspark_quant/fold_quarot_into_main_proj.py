import json, os, glob, shutil, torch
from safetensors.torch import safe_open, save_file
SRC = "/data1/DeepSeek-V4-Flash-DSpark-bf16"
DST = "/data2/DeepSeek-V4-Flash-DSpark-bf16-folded-draft"
QP  = "/data1/DeepSeek-V4-Flash-DSpark-w8a8-draft/optional/quarot.safetensors"
KEY = "mtp.0.main_proj.weight"
idx = json.load(open("/data2/DeepSeek-V4-Flash-DSpark-bf16-draft/model.safetensors.index.json"))
shard = idx["weight_map"][KEY]
print("main_proj in shard:", shard)
with safe_open(os.path.join("/data2/DeepSeek-V4-Flash-DSpark-bf16-draft", shard), framework="pt") as h:
    keys = list(h.keys()); tensors = {k: h.get_tensor(k) for k in keys}
W = tensors[KEY].float()                      # [4096, 12288]
with safe_open(QP, framework="pt") as h:
    Q = h.get_tensor("global_rotation").float()   # [4096, 4096]
h_dim = Q.shape[0]; n = W.shape[1] // h_dim
print("W", tuple(W.shape), "Q", tuple(Q.shape), "blocks", n)
Wf = torch.cat([W[:, k*h_dim:(k+1)*h_dim] @ Q for k in range(n)], dim=1)
# validation: y_old = (x @ Qt blockwise) @ W^T  vs  y_new = x @ Wf^T
x = torch.randn(8, n*h_dim)
xrot = torch.cat([x[:, k*h_dim:(k+1)*h_dim] @ Q.t() for k in range(n)], dim=1)
y_old = xrot @ W.t(); y_new = x @ Wf.t()
cos = torch.nn.functional.cosine_similarity(y_old.flatten(), y_new.flatten(), dim=0)
print("fold validation cosine: %.8f  maxdiff: %.3e" % (cos.item(), (y_old-y_new).abs().max().item()))
assert cos.item() > 0.9999
tensors[KEY] = Wf.to(tensors[KEY].dtype)
os.makedirs(DST, exist_ok=True)
# symlink everything from the existing -draft dir, replace only the fold shard + drop optional/
SRCDRAFT = "/data2/DeepSeek-V4-Flash-DSpark-bf16-draft"
for f in os.listdir(SRCDRAFT):
    if f == "optional": continue
    s = os.path.join(SRCDRAFT, f); d = os.path.join(DST, f)
    if os.path.lexists(d): os.remove(d)
    if f == shard: continue
    if os.path.islink(s): os.symlink(os.readlink(s), d)
    else: shutil.copy(s, d)
save_file(tensors, os.path.join(DST, shard), metadata={"format": "pt"})
print("FOLDED draft dir ready:", DST, "(no optional/ -> quarot no-op)")
