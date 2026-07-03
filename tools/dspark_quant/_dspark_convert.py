import os, glob, json, time, torch
from safetensors.torch import safe_open, save_file

t0=time.time()
print("torch", torch.__version__,
      "e4m3", hasattr(torch,"float8_e4m3fn"),
      "e8m0", hasattr(torch,"float8_e8m0fnu"), flush=True)

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

def deq_fp4(x_int8, scale):
    x = x_int8.view(torch.uint8)
    low = x & 0x0F; high = (x >> 4) & 0x0F
    vals = torch.stack([FP4_TABLE[low.long()], FP4_TABLE[high.long()]], dim=-1).flatten(1)
    out_dim, in_dim = vals.shape
    s = scale.float().repeat_interleave(32, dim=1)[:, :in_dim]
    return vals * s

def deq_fp8(w8, s8):
    out_dim, in_dim = w8.shape
    s = s8.float().repeat_interleave(128, 0).repeat_interleave(128, 1)[:out_dim, :in_dim]
    return w8.float() * s

SRC="/data1/dspark-draft/weights"
DST="/data1/dspark-draft/weights-fp8"
os.makedirs(DST, exist_ok=True)
def is_re(k): return ".experts." in k and "shared_experts" not in k

weight_map={}; total=0; verified=0; max_err=0.0; n_conv=0
for f in sorted(glob.glob(SRC+"/*.safetensors")):
    base=os.path.basename(f)
    with safe_open(f, framework="pt", device="cpu") as h:
        keys=list(h.keys()); raw={k:h.get_tensor(k) for k in keys}
    out={}
    for k in keys:
        if is_re(k) and k.endswith(".weight"):
            sc=raw[k[:-7]+".scale"]
            w8,s8=cast_e2m1fn_to_e4m3fn(raw[k], sc)
            out[k]=w8; out[k[:-7]+".scale"]=s8; n_conv+=1
            if verified<8:
                e=(deq_fp4(raw[k],sc)-deq_fp8(w8,s8)).abs().max().item()
                max_err=max(max_err,e); verified+=1
        elif is_re(k) and k.endswith(".scale"):
            continue
        else:
            out[k]=raw[k]
    save_file(out, os.path.join(DST, base), metadata={"format":"pt"})
    for k,v in out.items():
        weight_map[k]=base; total += v.numel()*v.element_size()
    print(f"[{time.time()-t0:6.1f}s] saved {base}: {len(out)} tensors", flush=True)

idx={"metadata":{"total_size":int(total)},"weight_map":weight_map}
with open(os.path.join(DST,"model.safetensors.index.json"),"w") as fo:
    json.dump(idx, fo, indent=2)

print(f"converted_experts={n_conv} verify_samples={verified} max|fp4-fp8 dequant|={max_err:.3e}", flush=True)
print("RESULT:", "PASS" if max_err==0.0 else "FAIL", flush=True)
print(f"total_bytes={total/1e9:.2f}GB  elapsed={time.time()-t0:.1f}s", flush=True)
