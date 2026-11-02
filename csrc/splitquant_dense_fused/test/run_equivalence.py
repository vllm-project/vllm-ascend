from pathlib import Path
import argparse,json,os,torch,torch_npu
p=argparse.ArgumentParser();p.add_argument('--profile',choices=['generic','30b'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--reference',type=Path);p.add_argument('--profiler',action='store_true');p.add_argument('--profiler-per-case',action='store_true');p.add_argument('--perf-cases',type=Path);a=p.parse_args()
from vllm_ascend.quantization.w4a4_splitquant import load_splitquant_operator
load_splitquant_operator()
out={};metrics=[];calls=[]
def pack(w):return ((w[:,::2].short()&15)|((w[:,1::2].short()&15)<<4)).to(torch.int8)
def add(name,fn):
 with torch.inference_mode():v=fn();torch.npu.synchronize();out[name]=v.cpu()
 calls.append((name,fn))
for g in ([2,16,32,40,136] if a.profile=='generic' else [2,16,32]):
 for m in [1,3,32]:
  for dtype in [torch.float16,torch.bfloat16]:
   torch.manual_seed(751+g+m);n=256;k=g*128;x=torch.randn(m,k,dtype=dtype).npu();w=pack(torch.randint(-8,8,(n,k),dtype=torch.int8)).npu();ws=(torch.rand(n,g)*.02+.001).npu();wo=torch.full((n,g),1.25 if m==3 else 0.).npu();cp=torch.full((g,),.9).npu();cn=torch.full((g,),.7).npu();args=(x,w,ws,wo,cp,cn)
   add(f'single-{g}-{m}-{dtype}',lambda args=args:torch.ops.npu.splitquant_dense_fused(*args))
for g in ([2,16,32,40,136] if a.profile=='generic' else [2,16,32]):
 for m in [1,32]:
  torch.manual_seed(823+g+m);k=g*128;n=512;x=torch.randn(m,k,dtype=torch.bfloat16).npu();w=pack(torch.randint(-8,8,(n,k),dtype=torch.int8)).reshape(n,g,64).permute(1,0,2).contiguous().npu();ws=(torch.rand(g,n)*.02+.001).npu();wo=torch.zeros(g,n).npu();cp=(torch.rand(2,g)*.2+.7).npu();cn=(torch.rand(2,g)*.2+.6).npu();gt=torch.eye(128,dtype=torch.bfloat16).expand(g,-1,-1).contiguous().npu();args=(x,w,ws,wo,cp,cn,gt,[256,256]);add(f'multi-{g}-{m}',lambda args=args:torch.ops.npu.splitquant_dense_fused_multi(*args))
for dtype in [torch.float16,torch.bfloat16,torch.float32]:
 torch.manual_seed(851);base=torch.randn(4*1048576,dtype=dtype).npu();cache=base.as_strided((4,3,10240),(1048576,10240,1));ids=torch.tensor([3,1],device='npu');got=torch.ops.npu.splitquant_state_gather(cache,ids);add('state-gather-'+str(dtype),lambda cache=cache,ids=ids:torch.ops.npu.splitquant_state_gather(cache,ids));src=torch.randn(2,3,10240,dtype=dtype).npu();torch.ops.npu.splitquant_state_scatter(cache,src,ids);add('state-scatter-'+str(dtype),lambda cache=cache:cache.contiguous())
a.out.mkdir(parents=True,exist_ok=True);torch.save(out,a.out/'outputs.pt')
if a.reference:
 ref=torch.load(a.reference,map_location='cpu',weights_only=True);assert set(ref)==set(out)
 for name,v in out.items():
  u=ref[name];d=(v.float().double()-u.float().double()).abs();rel=d/(u.double().abs()+1e-7);eq=torch.equal(u,v);metrics.append(dict(name=name,passed=eq,MaxAbsErr=d.max().item(),MeanAbsErr=d.mean().item(),MaxRelErr=rel.max().item(),MeanRelErr=rel.mean().item(),CosineSim=1. if eq else torch.nn.functional.cosine_similarity(v.double().flatten(),u.double().flatten(),dim=0).item()))
 assert all(x['passed'] for x in metrics),[x for x in metrics if not x['passed']]
if a.profiler or a.profiler_per_case:
 fns=calls[:8]
 if a.profiler_per_case:
  fns += [(n,f) for n,f in calls if n=='state-gather-torch.float32']
  if a.perf_cases:
   assert a.perf_cases.suffix=='.jsonl';specs=[json.loads(x) for x in a.perf_cases.read_text().splitlines()];wanted=[x['case'] for x in specs if x['profile']==a.profile];functions=dict(calls);fns=[(n,functions[n]) for n in wanted]
  batches=[(name,[fn]) for name,fn in fns]
 else:batches=[('combined',[fn for _,fn in fns])]
 for case_name,functions in batches:
  with torch.inference_mode(),torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],schedule=torch_npu.profiler.schedule(wait=0,warmup=5,active=5,repeat=1),on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(a.out/'profiler'/case_name)),experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1),record_shapes=False,profile_memory=False,with_stack=False) as prof:
   for _ in range(10):
    for fn in functions:fn()
    torch.npu.synchronize();prof.step()
library=Path(__import__('vllm_ascend.quantization.w4a4_splitquant',fromlist=['__file__']).__file__).resolve().parents[1]/'lib'/'libvllm_ascend_splitquant_dense.so'
result=dict(status='PASS',profile=a.profile,cases=len(out),metrics=metrics,library=str(library));(a.out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print('EQUIVALENCE_PASS',a.profile,len(out),flush=True)
