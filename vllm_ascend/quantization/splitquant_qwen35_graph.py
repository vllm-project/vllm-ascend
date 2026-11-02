"""Opt-in Qwen3.5 TP1 batch1 decode Graph adapter; existing arithmetic order.

The caller selects this only for the current dense Qwen3.5 text model.
Prefill keeps its existing chunk path. No device scalar is read by Python.
"""
import torch
import torch.nn.functional as F
GRAPH_COUNTS = dict(capture=0,replay=0,prefill_eager=0)

def install_graph_audit():
 from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
 # Ascend runner uses aclgraph_support, generic GDN publishes cudagraph_support.
 if not hasattr(GDNAttentionMetadataBuilder,'aclgraph_support'):
  GDNAttentionMetadataBuilder.aclgraph_support=GDNAttentionMetadataBuilder.cudagraph_support
 if not hasattr(GDNAttentionMetadataBuilder,'build_for_graph_capture'):
  def build_for_graph_capture(self,common_attn_metadata,*args,**kwargs):
   return self.build_for_cudagraph_capture(common_attn_metadata)
  GDNAttentionMetadataBuilder.build_for_graph_capture=build_for_graph_capture
 from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
 from vllm.forward_context import get_forward_context
 if getattr(ACLGraphWrapper,'_qwen35_audited',False):return
 original=ACLGraphWrapper.__call__
 def audited(self,*args,**kwargs):
  config=getattr(self.runnable,'_hf_config',None)
  if not getattr(config,'_splitquant_dense_qwen35',False):
   return original(self,*args,**kwargs)
  context=get_forward_context();mode=context.cudagraph_runtime_mode
  active=mode==self.runtime_mode and str(mode) not in ('NONE','CUDAGraphMode.NONE')
  key=context.batch_descriptor
  replay=active and key in self.concrete_aclgraph_entries and self.concrete_aclgraph_entries[key].aclgraph is not None
  # A new one-token prompt can share the uniform-decode batch descriptor,
  # but PrefillNoCache uses a different ordinary-attention branch. Do not
  # replay a DecodeOnly capture for that first step.
  metadata=context.attn_metadata
  values=metadata.values() if isinstance(metadata,dict) else []
  states=[getattr(getattr(m,'attn_state',None),'name',None) for m in values]
  if replay and any(name is not None and name!='DecodeOnly' for name in states):
   GRAPH_COUNTS['prefill_eager']+=1
   return self.runnable(*args,**kwargs)
  result=original(self,*args,**kwargs)
  if active:GRAPH_COUNTS['replay' if replay else 'capture']+=1
  return result
 ACLGraphWrapper.__call__=audited  # type: ignore[method-assign]
 ACLGraphWrapper._qwen35_audited=True  # type: ignore[attr-defined]

def _gather(cache,indices,use_state):
 if use_state:
  from vllm_ascend.quantization.splitquant_state import gather
  return gather(cache,indices)
 return torch.index_select(cache,0,indices.long())

def _scatter(cache,source,indices,use_state):
 if use_state:
  from vllm_ascend.quantization.splitquant_state import scatter
  return scatter(cache,source,indices)
 cache.index_copy_(0,indices.long(),source.contiguous())
 return cache

def conv_update(x,conv_state,weight,bias=None,activation=None,conv_state_indices=None,use_state=False,**kwargs):
 if x.ndim!=2 or x.shape[0]!=1:raise NotImplementedError('Graph adapter supports batch1 single-token decode')
 indices=conv_state_indices.long() if conv_state_indices is not None else torch.zeros(1,device=x.device,dtype=torch.long)
 base=conv_state.transpose(-1,-2)
 history=_gather(base,indices,use_state).transpose(-1,-2)
 window=torch.cat((history[..., -(weight.shape[1]-1):],x.unsqueeze(-1)),-1)
 out=(window.float()*weight.float()).sum(-1)
 if bias is not None:out=out+bias.float()
 if activation in (True,'silu','swish'):out=F.silu(out)
 source=window[..., -conv_state.shape[-1]:].to(conv_state.dtype)
 _scatter(base,source.transpose(-1,-2).contiguous(),indices,use_state)
 return out.to(x.dtype)

def recurrent(q,k,v,g,beta,initial_state,inplace_final_state=True,cu_seqlens=None,ssm_state_indices=None,use_qk_l2norm_in_kernel=False,use_state=False,**kwargs):
 if kwargs.get('num_accepted_tokens') is not None:raise NotImplementedError('No speculative decode')
 if q.shape[1]!=1 or len(cu_seqlens)!=2:raise NotImplementedError('Graph adapter supports exactly one single-token sequence')
 indices=ssm_state_indices[:1] if ssm_state_indices is not None else torch.zeros(1,device=q.device,dtype=torch.long)
 state=_gather(initial_state,indices,use_state).float()
 query,key=q[:,0:1].float(),k[:,0:1].float()
 if use_qk_l2norm_in_kernel:
  query=F.normalize(query,p=2,dim=-1);key=F.normalize(key,p=2,dim=-1)
 ratio=v.shape[2]//k.shape[2]
 query=query.repeat_interleave(ratio,dim=2)*q.shape[-1]**-0.5
 key=key.repeat_interleave(ratio,dim=2)
 kt=key[:,0]
 state=state*g[:,0].float().exp()[...,None,None]
 residual=v[:,0].float()-(state*kt[...,None]).sum(-2)
 state=state+kt[...,None]*(residual*beta[:,0].float()[...,None])[...,None,:]
 out=(state*query[:,0,:,:,None]).sum(-2).unsqueeze(1).to(v.dtype)
 if inplace_final_state:_scatter(initial_state,state.to(initial_state.dtype),indices,use_state)
 return out,initial_state
