import os, glob, sys, time, torch
from safetensors.torch import safe_open, save_file

N = int(sys.argv[1]); REM = int(sys.argv[2])
t0 = time.time()

FP4_TABLE = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
                          0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], dtype=torch.float32)

def cast_e2m1fn_to_e4m3fn(x, scale):
    assert x.dtype == torch.int8 and x.ndim == 2
    out_dim, in_dim = x.size(); in_dim *= 2
    fp8_bs, fp4_bs = 128, 32
    assert in_dim % fp8_bs == 0 and out_dim % fp8_bs == 0
    assert scale.size(0) == out_dim and scale.size(1) == in_dim // fp4_bs
    x = x.view(torch.uint8)
    low = x & 0x0F; high = (x >> 4) & 0x0F
    x = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(2)
    MAX_OFFSET_BITS = 6
    bOut, bIn = out_dim // fp8_bs, in_dim // fp8_bs
    x = x.view(bOut, fp8_bs, bIn, fp8_bs).transpose(1, 2)
    scale = scale.float().view(bOut, fp8_bs, bIn, -1).transpose(1, 2).flatten(2)
    smob = scale.amax(dim=-1, keepdim=True) / (2**MAX_OFFSET_BITS)
    offset = scale / smob
    offset = offset.unflatten(-1, (fp8_bs, -1)).repeat_interleave(fp4_bs, dim=-1)
    x = (x * offset).transpose(1, 2).reshape(out_dim, in_dim)
    return x.to(torch.float8_e4m3fn).contiguous(), smob.squeeze(-1).to(torch.float8_e8m0fnu).contiguous()

SRC = "/data1/DeepSeek-V4-Flash-DSpark"
DST = "/data1/DeepSeek-V4-Flash-DSpark-fp8-full"
os.makedirs(DST, exist_ok=True)
def is_re(k): return ".experts." in k and "shared_experts" not in k

files = sorted(glob.glob(SRC + "/*.safetensors"))
for fi, f in enumerate(files):
    if fi % N != REM:
        continue
    base = os.path.basename(f)
    outp = os.path.join(DST, base)
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        print(f"[W{REM}] skip existing {base}", flush=True); continue
    with safe_open(f, framework="pt", device="cpu") as h:
        keys = list(h.keys()); raw = {k: h.get_tensor(k) for k in keys}
    out = {}
    for k in keys:
        if is_re(k) and k.endswith(".weight"):
            sc = raw[k[:-7] + ".scale"]
            w8, s8 = cast_e2m1fn_to_e4m3fn(raw[k], sc)
            out[k] = w8; out[k[:-7] + ".scale"] = s8
        elif is_re(k) and k.endswith(".scale"):
            continue
        else:
            out[k] = raw[k]
    save_file(out, outp + ".tmp", metadata={"format": "pt"})
    os.replace(outp + ".tmp", outp)
    print(f"[W{REM} {time.time()-t0:6.1f}s] saved {base}: {len(out)} tensors", flush=True)
print(f"[W{REM}] DONE elapsed={time.time()-t0:.1f}s", flush=True)
