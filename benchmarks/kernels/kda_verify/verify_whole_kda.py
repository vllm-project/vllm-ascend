"""Replay a captured GLM-5.3-Flash KDA verifier input through its native and fused paths."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace as NS

import torch
import torch_npu
from vllm_ascend.models.glm5next.ops.causal_conv1d import causal_conv1d
from vllm_ascend.ops.triton.kda.recoverssm import kda_recoverssm_verify
from vllm_ascend.ops.triton.kda.kda import rms_norm_gated
from tilelang_whole_kda import WholeKDA
parser=argparse.ArgumentParser()
parser.add_argument("--out",required=True)
parser.add_argument("--trace",default="trace.pt")
parser.add_argument("--batches",default="10")
parser.add_argument("--profile")
parser.add_argument("--mstx",action="store_true")
parser.add_argument("--stress",type=int,default=0)
args=parser.parse_args()
inference_guard=torch.inference_mode()
inference_guard.__enter__()
assert torch.is_inference_mode_enabled()
torch.npu.set_device(0)
trace=torch.load(args.trace,map_location="cpu",weights_only=False)
assert trace["conv_ids"].min().item()>0 and trace["state_ids"].min().item()>0
assert trace["state"].float().abs().max().item()>0
def weight(name):
    w=trace[name].npu()
    fmt=trace[name+"_info"][-1]
    return torch_npu.npu_format_cast(w,fmt) if torch_npu.get_npu_format(w)!=fmt else w
layer=NS(_conv_state_dim_first=False,kda_lower_bound=trace["lower_bound"],num_spec=5,prefix="model.layers.0.self_attn",A_log=trace["A"].npu(),dt_bias=trace["dt"].npu(),o_norm=NS(weight=trace["nw"].npu(),eps=trace["eps"]))
for name in ("in_proj_qkvbfg_a","f_b_proj","g_b_proj","o_proj","q_conv1d","k_conv1d","v_conv1d"):
    setattr(layer,name,NS(weight=weight(name)))
cw=torch.cat([m.weight.reshape(8192,4) for m in (layer.q_conv1d,layer.k_conv1d,layer.v_conv1d)]).T.to(torch.bfloat16).contiguous()
F=torch.nn.functional.linear
print("DEVICE",torch.npu.get_device_name(0),"LAYOUT",trace["layouts"],flush=True)
print("WEIGHTS",[(n,trace[n+"_info"]) for n in ("in_proj_qkvbfg_a","f_b_proj","g_b_proj","o_proj")],flush=True)

a=layer.A_log.reshape(-1).float().contiguous()
dt=layer.dt_bias.reshape(64,128).float().contiguous()
results={}
for b in map(int,args.batches.split(",")):
    pages=2*b+2
    page_bytes=trace["layouts"][0][1][0]*2
    raw=torch.zeros(pages*page_bytes,device="npu",dtype=torch.uint8)
    def alloc(idx,dtype):
        shape,stride=trace["layouts"][idx]
        storage=raw.view(dtype)
        offset=trace["offsets"][idx]-trace["offsets"][0]*2//storage.element_size()
        assert offset>=0
        return storage.as_strided((pages,*shape[1:]),stride,offset)
    conv,state,corr,kg=[alloc(i,dtype) for i,dtype in enumerate((torch.bfloat16,torch.float32,torch.float32,torch.bfloat16))]
    layer.kv_cache=(conv,state,corr,kg)
    ids=(2*torch.arange(b,device="npu",dtype=torch.int32)+1).flip((0,)).contiguous()
    ids64=ids.long()
    def batch_data(key):
        source=trace[key]
        return source[torch.arange(b)%source.shape[0]].contiguous().npu()
    hidden=trace["hidden"].reshape(-1,6,4096)
    x=hidden[torch.arange(b)%len(hidden)].reshape(b*6,4096).contiguous().npu()
    initial=batch_data("conv")
    checkpoint=batch_data("state")
    state.index_copy_(0,ids64,checkpoint)
    conv_out=None if trace["conv_meta"]["output_buffer"][0]=="NoneType" else x.new_empty((b*6,24576))
    accepted=batch_data("accepted").reshape(b).int()
    starts=torch.arange(b+1,device="npu",dtype=torch.int32)*6
    # The serving metadata stores the checkpoint in the first of the per-token columns.
    slots=ids[:,None].expand(b,trace["state_index_shape"][1]).contiguous()
    print("COMPILING", b, flush=True)
    fused=WholeKDA(layer)
    workspace=fused.allocate(x)
    p,y,out=workspace
    metadata=torch.stack((ids,slots[:,0],accepted)).int().contiguous()
    def run():
        return fused(x,metadata,workspace)
    conv.index_copy_(0,ids64,initial)
    p0=F(x,layer.in_proj_qkvbfg_a.weight)
    qkv,beta,fa,ga=p0.split([24576,64,128,128],-1)
    fg=F(fa,layer.f_b_proj.weight).reshape(1,-1,64,128)
    gg=F(ga,layer.g_b_proj.weight).reshape(-1,64,128)
    cv=causal_conv1d(qkv,cw,conv,starts,ids,run_mode=1,num_accepted_tokens=accepted,output_buffer=conv_out)
    q,k,v=[z.reshape(1,-1,64,128) for z in cv.split(8192,-1)]
    core=kda_recoverssm_verify(q,k,v,fg,beta.unsqueeze(0),a,dt,layer.kda_lower_bound,state,corr,kg,starts,slots[:,0],6)
    y0=rms_norm_gated(core,gg,layer.o_norm.weight,None,activation="sigmoid",eps=layer.o_norm.eps).reshape(b*6,8192)
    out0=F(y0,layer.o_proj.weight)
    gold=[t.index_select(0,ids64) for t in (conv,corr,kg)]
    conv.index_copy_(0,ids64,initial)
    for t in (corr,kg):
        t.index_fill_(0,ids64,float("nan"))
    torch.npu.synchronize()
    print("LAUNCH",b,flush=True)
    run()
    torch.npu.synchronize()
    def rel(t,r):
        return ((t.float()-r.float()).norm() / r.float().norm().clamp_min(1e-20)).item()
    live=[t.index_select(0,ids64) for t in (conv,corr,kg)]
    errors={name:rel(t,r) for name,t,r in zip(("in_proj","gated_norm","o_proj","conv","correction","kg"),(p[:b*6],y,out[:b*6],*live),(p0,y0,out0,*gold))}
    results[b]={"errors":errors,"checkpoint_unchanged":torch.equal(state.index_select(0,ids64),checkpoint),"device":torch.npu.get_device_name(0)}
    print("CHECK",b,results[b],flush=True)
    Path(args.out).write_text(json.dumps(results,indent=2))
    assert errors["o_proj"]<0.012 and all(errors[k]<0.003 for k in ("in_proj","conv","correction","kg")) and results[b]["checkpoint_unchanged"]
    assert not any(t.index_select(0,torch.arange(0,2*b+2,2,device="npu")).count_nonzero().item() for t in (conv,state,corr,kg))
    for trial in range(args.stress):
        conv.index_copy_(0,ids64,initial)
        for scratch in workspace:
            scratch.fill_(float("nan"))
        actual=run()
        assert rel(actual,out0)<0.012, (b,trial,"o_proj",rel(actual,out0))
        assert rel(p[:b*6],p0)<0.003 and rel(y,y0)<0.012, (b,trial,"projection workspace")
    if args.profile:
        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.NPU],on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args.profile)) as prof:
            run()
            torch.npu.synchronize()
            prof.step()
    if args.mstx:
        for rep in range(-4,11):
            conv.index_copy_(0,ids64,initial)
            torch.npu.synchronize()
            if rep >= 0:
                mark=torch_npu.npu.mstx.range_start(f"KDATIME_{b}_whole_{rep}")
            run()
            torch.npu.synchronize()
            if rep >= 0:
                torch_npu.npu.mstx.range_end(mark)
