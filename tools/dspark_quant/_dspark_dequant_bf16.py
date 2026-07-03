import os, glob, sys, time, json, struct, torch
from safetensors.torch import safe_open, save_file

N = int(sys.argv[1]); REM = int(sys.argv[2])
t0 = time.time()
SRC = "/data1/DeepSeek-V4-Flash-DSpark"
DST = "/data1/DeepSeek-V4-Flash-DSpark-bf16"
os.makedirs(DST, exist_ok=True)

FP4_TABLE = torch.tensor([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0,
                          0.0,-0.5,-1.0,-1.5,-2.0,-3.0,-4.0,-6.0], dtype=torch.float32)

def deq_fp4(x_int8, scale):
    # x_int8 [out, in/2] packed nibbles; scale [out, in/32] (E8M0 -> .float() = 2^e)
    x = x_int8.view(torch.uint8)
    low = x & 0x0F; high = (x >> 4) & 0x0F
    vals = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)
    out_dim, in_dim = vals.shape
    s = scale.float().repeat_interleave(32, dim=1)[:, :in_dim]
    return vals * s

def deq_fp8(w8, s8):
    # w8 [out,in] e4m3; s8 [out/128, in/128] E8M0 (.float() = 2^e)
    out_dim, in_dim = w8.shape
    s = s8.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:out_dim, :in_dim]
    return w8.float() * s

def hdrdtype(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return {k: v.get("dtype") for k, v in h.items() if isinstance(v, dict) and "dtype" in v}

files = sorted(glob.glob(SRC + "/*.safetensors"))
for fi, f in enumerate(files):
    if fi % N != REM:
        continue
    base = os.path.basename(f)
    outp = os.path.join(DST, base)
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        print(f"[W{REM}] skip {base}", flush=True); continue
    dtypes = hdrdtype(f)
    with safe_open(f, framework="pt", device="cpu") as h:
        keys = list(h.keys()); raw = {k: h.get_tensor(k) for k in keys}
    out = {}
    for k in keys:
        dt = dtypes.get(k)
        if k.endswith(".scale"):
            continue                                  # drop quant scales
        if k.endswith(".weight") and dt in ("I8", "F8_E4M3"):
            scname = k[:-7] + ".scale"
            sc = raw.get(scname)
            if sc is None:                            # unquantized .weight -> keep
                out[k] = raw[k].to(torch.bfloat16); continue
            if dt == "I8":
                out[k] = deq_fp4(raw[k], sc).to(torch.bfloat16)
            else:
                out[k] = deq_fp8(raw[k], sc).to(torch.bfloat16)
        elif dt == "F32":
            out[k] = raw[k]                            # keep norms/params fp32
        else:
            out[k] = raw[k].to(torch.bfloat16) if dt == "BF16" else raw[k]
    save_file(out, outp + ".tmp", metadata={"format": "pt"})
    os.replace(outp + ".tmp", outp)
    print(f"[W{REM} {time.time()-t0:6.1f}s] {base}: {len(out)} tensors", flush=True)
print(f"[W{REM}] DONE elapsed={time.time()-t0:.1f}s", flush=True)
