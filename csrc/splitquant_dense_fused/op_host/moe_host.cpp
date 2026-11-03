// SPDX-License-Identifier: BSD-3-Clause
#include "torch_kernel_helper.h"
#include "weight_scale_cache.h"
#include "raw_gate_cache.h"
#include "../ops.h"
#include <ATen/core/dispatch/Dispatcher.h>
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"
#include <map>
#include <tuple>
#include <algorithm>
#include <cstdlib>
#include "aclrtlaunch_moe_quant_bf16.h"
#include "aclrtlaunch_moe_quant_fp16.h"
#include "aclrtlaunch_moe_gate.h"
#include "aclrtlaunch_moe_gate_nz.h"
#include "aclrtlaunch_moe_gate_const_nz.h"
#include "aclrtlaunch_moe_gate_raw_nz.h"
#include "aclrtlaunch_moe_post_bf16.h"
#include "aclrtlaunch_moe_post_compact_bf16.h"
#include "aclrtlaunch_moe_post_paged_bf16.h"
namespace splitquant {at::Tensor compact_lut(const at::Tensor&);at::Tensor paged_lut(const at::Tensor&);}
#include "aclrtlaunch_moe_post_fp16.h"
#include "aclrtlaunch_moe_rotate_bf16.h"
#include "aclrtlaunch_moe_rotate_fp16.h"
#include "aclrtlaunch_moe_down_bf16.h"
#include "aclrtlaunch_moe_down_nz_bf16.h"
#include "aclrtlaunch_moe_down_nz_fp16.h"
#include "aclrtlaunch_moe_down_fp16.h"
#include "aclrtlaunch_moe_scatter_bf16.h"
#include "aclrtlaunch_moe_scatter_fp16.h"
namespace ascend_kernel {
namespace {
at::Tensor Basic(const at::Tensor& x){
 static auto format=c10::Dispatcher::singleton().findSchemaOrThrow("npu::get_npu_format", "").typed<int64_t(const at::Tensor&)>();auto f=format.call(x);if(f==0||f==2)return x;
 static auto cast=c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_format_cast", "").typed<at::Tensor(const at::Tensor&,int64_t,c10::optional<int64_t>)>();return cast.call(x,2,c10::nullopt);
}
int64_t GateTile(){
 int64_t tn=512;if(const char* v=std::getenv("SPLITQUANT_MOE_TILE_N"))tn=std::atoi(v);
 TORCH_CHECK(tn==128||tn==256||tn==512||tn==768||tn==1024||tn==1536," gate tileN invalid");return tn;
}
int64_t GateGroup(){int64_t gp=16;if(const char* v=std::getenv("SPLITQUANT_MOE_GROUP_TILE"))gp=std::atoi(v);TORCH_CHECK(gp==8||gp==16," group tile must be8/16");return gp;}
int64_t DownTile(){int64_t n=512;if(const char* v=std::getenv("SPLITQUANT_MOE_DOWN_TILE_N"))n=std::atoi(v);TORCH_CHECK(n==128||n==256||n==512||n==1024," down tile invalid");return n;}
int64_t RotateTile(int64_t r){int64_t n=64;if(const char* v=std::getenv("SPLITQUANT_MOE_ROTATE_TILE_N"))n=std::atoi(v);TORCH_CHECK(n==64||n==128," rotate tile invalid");return r<=8?n:128;}
int64_t GateBaseK(){int64_t v=128;if(const char* p=std::getenv("SPLITQUANT_MOE_BASE_K"))v=std::atoi(p);TORCH_CHECK(v==128||v==256||v==512," MoE baseK invalid");return v;}
int64_t GateBaseN(){int64_t v=512;if(const char* p=std::getenv("SPLITQUANT_MOE_BASE_N"))v=std::atoi(p);TORCH_CHECK(v==128||v==256||v==512," MoE baseN invalid");return v;}
at::Tensor Tiling(const at::Tensor& x,int kind,int64_t m,int64_t h,int64_t inter,int64_t r){
 using Key=std::tuple<int64_t,int,int64_t,int64_t,int64_t,int64_t,int64_t,c10::StreamId,int64_t,int64_t>;
 static thread_local std::map<Key,at::Tensor> cache;
 Key key{kind+GateTile()*10+GateGroup()*100000+DownTile()*10000000+RotateTile(r)*10000000000LL,int(x.scalar_type()),m,h,inter,r,x.device().index(),c10_npu::getCurrentNPUStream(x.device().index()).id(),GateBaseK(),GateBaseN()};auto it=cache.find(key);if(it!=cache.end())return it->second;
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform," platform unavailable");
 using namespace matmul_tiling;MatmulApiTiling api(*platform);
 const bool gate=kind==0||kind==4;
 auto dtype=gate?DataType::DT_INT4:(x.scalar_type()==at::kBFloat16?DataType::DT_BF16:DataType::DT_FLOAT16);
 int64_t n=gate?2*inter:kind==1?128:h,tn=gate?GateTile():kind==1?RotateTile(r):DownTile(),kk=gate?(m==1?GateGroup()*128:128):kind==1?128:inter,orgm=kind==1?r:1,orgk=gate?(m==1?GateGroup()*128:128):inter;
 TORCH_CHECK(api.SetAType(TPosition::GM,CubeFormat::ND,dtype,false)==0 && api.SetBType(TPosition::GM,(kind==3||kind==4)?CubeFormat::NZ:CubeFormat::ND,dtype,kind!=1&&kind!=3&&kind!=4)==0 && api.SetCType(TPosition::GM,CubeFormat::ND,gate?DataType::DT_INT32:dtype)==0," tiling types");
 TORCH_CHECK(api.SetShape(16,tn,kk)==0 && api.SetOrgShape(orgm,n,orgk)==0 && api.SetFixSplit(16,gate?std::min<int64_t>(GateBaseN(),tn):kind>=2?std::min<int64_t>(256,tn):tn,gate?std::min<int64_t>(kk,GateBaseK()):128)==0," tiling shape");
 optiling::TCubeTiling tile{};TORCH_CHECK(api.GetTiling(tile)==0," tiling failed, stage=",kind,", N=",n,", K=",kk);tile.set_M(16);tile.set_N(tn);tile.set_Ka(kk);tile.set_Kb(kk);
 auto cpu=at::empty({int64_t(tile.GetDataSize())},at::TensorOptions().dtype(at::kByte).device(at::kCPU));tile.SaveToBuffer(cpu.data_ptr(),tile.GetDataSize());auto dev=cpu.to(x.device(),false,true);cache.emplace(key,dev);return dev;
}
std::vector<at::Tensor> Run(const at::Tensor& input,const at::Tensor& up,const at::Tensor& ws,const at::Tensor& dw,const at::Tensor& cp,const at::Tensor& cn,const at::Tensor& hp,const at::Tensor& hn,const at::Tensor& gt,const at::Tensor& ids,const at::Tensor& weights,const at::Tensor& lut){
 TORCH_CHECK(input.device().type()==c10::DeviceType::PrivateUse1 && input.dim()==2 && input.is_contiguous() && (input.scalar_type()==at::kHalf||input.scalar_type()==at::kBFloat16)," requires contiguous NPU FP16/BF16 x[M,H]");
 int64_t m=input.size(0),h=input.size(1);TORCH_CHECK((up.dim()==4||up.dim()==5) && ids.dim()==2," weight/ids ranks");int64_t experts=up.size(0),inter=(up.dim()==5?up.size(1)*32:up.size(1)/2),k=ids.size(1),g=h/128,r=m*k;
 TORCH_CHECK(m>=1&&m<=32&&h>=128&&h<=4096&&h%128==0&&inter>=128&&inter<=1024&&inter%128==0&&experts>=1&&experts<=128&&k>=1&&k<=8&&k<=experts," dimensions out of supported range");
 TORCH_CHECK(((up.dim()==4&&up.size(1)==2*inter&&up.size(2)==g&&up.size(3)==64)||(up.dim()==5&&up.size(1)==2*inter/64&&up.size(2)==h/16&&up.size(3)==16&&up.size(4)==32))&&(up.scalar_type()==at::kByte||up.scalar_type()==at::kChar)," packed up shape/dtype");
 for(const auto* t:{&up,&ws,&dw,&cp,&cn,&hp,&hn,&gt,&ids,&weights,&lut})TORCH_CHECK(t->device()==input.device()&&t->is_contiguous()," tensors must be contiguous on one device");
 TORCH_CHECK(ws.sizes()==at::IntArrayRef({experts,g,2*inter})&&ws.scalar_type()==at::kFloat," up scale shape");
 TORCH_CHECK((dw.sizes()==at::IntArrayRef({experts,h,inter})||dw.sizes()==at::IntArrayRef({experts,h/16,inter/16,16,16}))&&dw.scalar_type()==input.scalar_type()," down cache shape");
 TORCH_CHECK(gt.sizes()==at::IntArrayRef({inter/128,128,128})&&gt.scalar_type()==input.scalar_type()," rotation shape");
 TORCH_CHECK(ids.size(0)==m&&ids.scalar_type()==at::kLong&&weights.sizes()==ids.sizes()&&weights.scalar_type()==input.scalar_type()," route shapes");
 TORCH_CHECK(lut.dim()==1&&lut.numel()==65536&&lut.scalar_type()==input.scalar_type()," SiLU LUT");
 for(const auto* t:{&cp,&cn})TORCH_CHECK(t->scalar_type()==at::kFloat&&t->numel()==g," input clips");for(const auto* t:{&hp,&hn})TORCH_CHECK(t->scalar_type()==at::kFloat&&t->numel()==inter/128," hidden clips");
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform," platform unavailable");uint64_t ub=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);TORCH_CHECK(ub>=172128," fused post requires 172128 bytes UB");
 bool nz_down=dw.dim()==5,nz_up=up.dim()==5;bool compact=std::getenv("SPLITQUANT_COMPACT_SILU")&&std::atoi(std::getenv("SPLITQUANT_COMPACT_SILU"))==1&&input.scalar_type()==at::kBFloat16;bool paged=std::getenv("SPLITQUANT_PAGED_SILU")&&std::atoi(std::getenv("SPLITQUANT_PAGED_SILU"))==1&&input.scalar_type()==at::kBFloat16;auto ltable=paged?splitquant::paged_lut(lut):compact?splitquant::compact_lut(lut):lut;if(paged&&!ltable.defined()){paged=false;compact=false;ltable=lut;}
 auto x=Basic(input);auto wt=nz_down?dw:Basic(dw);auto rot=Basic(gt);auto u=Basic(up);auto sc=Basic(ws);auto routing=Basic(weights);
 int64_t gp=GateGroup();
 auto a=(m==1?at::empty({(g+gp-1)/gp,16,gp*64},x.options().dtype(at::kByte)):at::empty({g,m,64},x.options().dtype(at::kByte)));auto scale=at::empty({m,g,8},x.options().dtype(at::kFloat));auto partial=at::empty({r,g,2*inter},x.options().dtype(at::kInt));auto hidden=at::empty({r,inter},x.options());auto rotated=at::empty({r,inter},x.options());auto down=at::empty({r,h},x.options());auto out=at::empty({m,h},x.options());
 auto tg=Tiling(x,nz_up?4:0,m,h,inter,r),tr=Tiling(x,1,m,h,inter,r),td=Tiling(x,nz_down?3:2,m,h,inter,r);
 auto work=at::empty({int64_t(platform->GetLibApiWorkSpaceSize())+4096},x.options().dtype(at::kByte));
 uint32_t mu=m,hu=h,iu=inter,ku=k,eu=experts,ru=r,gtn=GateTile(),gpu=gp,dtn=DownTile(),rtn=RotateTile(r);
 uint32_t scalarRound=(x.scalar_type()==at::kBFloat16&&std::getenv("SPLITQUANT_SCALAR_BF16_ROUND")&&std::atoi(std::getenv("SPLITQUANT_SCALAR_BF16_ROUND"))==1)?1:0;
 if(std::getenv("SPLITQUANT_BROADCAST_SCALE")&&std::atoi(std::getenv("SPLITQUANT_BROADCAST_SCALE"))==1)scalarRound|=2;
 if(std::getenv("SPLITQUANT_CACHE_WEIGHT_SCALE")&&std::atoi(std::getenv("SPLITQUANT_CACHE_WEIGHT_SCALE"))==1){sc=splitquant::rounded_weight_scale(sc,x.scalar_type());scalarRound|=4;}
 uint32_t lutBytes=paged?ltable.numel():0;
 auto aiv=[&](int64_t tasks){return uint32_t(std::min<int64_t>(platform->GetCoreNumAiv(),tasks));};auto aic=[&](int64_t tasks){return uint32_t(std::min<int64_t>(platform->GetCoreNumAic(),tasks));};
 uint32_t bq=aiv(m*g),bg=aic(m==1?r*((g+gp-1)/gp)*((2*inter+gtn-1)/gtn):r*g*((2*inter+gtn-1)/gtn)),bp=aiv(r*(inter/128)),br=aic((inter/128)*((r+15)/16)*(128/rtn)),bd=aic(r*((h+dtn-1)/dtn)),bs=aiv(m*(h/64));
 bool rawGate=std::getenv("SPLITQUANT_RAW_GATE")&&std::atoi(std::getenv("SPLITQUANT_RAW_GATE"))==1&&nz_up&&m==1&&h==2048&&inter==768&&gp==16;
 at::Tensor rawUp;
 if(rawGate){
  uint64_t l1=0,l0a=0,l0b=0,l0c=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);
  TORCH_CHECK(l1>=81920&&l0a>=16384&&l0b>=65536&&l0c>=4096," raw gate exceeds physical memory");
  rawUp=splitquant::raw_gate_weight(u,h,2*inter);bg=aic(k*(2*inter/64));
 }
 bool fixedGate=std::getenv("SPLITQUANT_CONST_MATMUL")&&std::atoi(std::getenv("SPLITQUANT_CONST_MATMUL"))==1&&nz_up&&m==1&&h==2048&&inter==768&&gp==16&&gtn==768&&GateBaseN()==256&&GateBaseK()==512;
 if(x.scalar_type()==at::kBFloat16){
  EXEC_KERNEL_CMD(moe_quant_bf16,bq,x,cp,cn,a,scale,mu,hu,gpu,scalarRound);
  if(rawGate){EXEC_KERNEL_CMD(moe_gate_raw_nz,bg,a,rawUp,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  else if(fixedGate){EXEC_KERNEL_CMD(moe_gate_const_nz,bg,a,u,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  else if(nz_up){EXEC_KERNEL_CMD(moe_gate_nz,bg,a,u,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  else{EXEC_KERNEL_CMD(moe_gate,bg,a,u,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  if(paged){EXEC_KERNEL_CMD(moe_post_paged_bf16,bp,partial,sc,scale,hidden,hp,hn,ids,ltable,mu,hu,iu,ku,eu,scalarRound,lutBytes);}
  else if(compact){EXEC_KERNEL_CMD(moe_post_compact_bf16,bp,partial,sc,scale,hidden,hp,hn,ids,ltable,mu,hu,iu,ku,eu,scalarRound);}
  else{EXEC_KERNEL_CMD(moe_post_bf16,bp,partial,sc,scale,hidden,hp,hn,ids,lut,mu,hu,iu,ku,eu,scalarRound);}
  EXEC_KERNEL_CMD(moe_rotate_bf16,br,hidden,rot,ids,rotated,ru,hu,iu,eu,rtn,work,tr);
  if(nz_down){EXEC_KERNEL_CMD(moe_down_nz_bf16,bd,rotated,wt,ids,down,ru,hu,iu,eu,dtn,work,td);}
  else{EXEC_KERNEL_CMD(moe_down_bf16,bd,rotated,wt,ids,down,ru,hu,iu,eu,dtn,work,td);}
  EXEC_KERNEL_CMD(moe_scatter_bf16,bs,down,ids,routing,out,mu,hu,ku);
 }else{
  EXEC_KERNEL_CMD(moe_quant_fp16,bq,x,cp,cn,a,scale,mu,hu,gpu,scalarRound);
  if(rawGate){EXEC_KERNEL_CMD(moe_gate_raw_nz,bg,a,rawUp,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  else if(fixedGate){EXEC_KERNEL_CMD(moe_gate_const_nz,bg,a,u,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  else if(nz_up){EXEC_KERNEL_CMD(moe_gate_nz,bg,a,u,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  else{EXEC_KERNEL_CMD(moe_gate,bg,a,u,ids,partial,mu,hu,iu,ku,eu,gtn,gpu,work,tg);}
  EXEC_KERNEL_CMD(moe_post_fp16,bp,partial,sc,scale,hidden,hp,hn,ids,lut,mu,hu,iu,ku,eu,scalarRound);
  EXEC_KERNEL_CMD(moe_rotate_fp16,br,hidden,rot,ids,rotated,ru,hu,iu,eu,rtn,work,tr);
  if(nz_down){EXEC_KERNEL_CMD(moe_down_nz_fp16,bd,rotated,wt,ids,down,ru,hu,iu,eu,dtn,work,td);}
  else{EXEC_KERNEL_CMD(moe_down_fp16,bd,rotated,wt,ids,down,ru,hu,iu,eu,dtn,work,td);}
  EXEC_KERNEL_CMD(moe_scatter_fp16,bs,down,ids,routing,out,mu,hu,ku);
 }
 return {a,scale,partial,hidden,rotated,down,out};
}
}
at::Tensor splitquant_moe_fused(const at::Tensor& x,const at::Tensor& up,const at::Tensor& ws,const at::Tensor& dw,const at::Tensor& cp,const at::Tensor& cn,const at::Tensor& hp,const at::Tensor& hn,const at::Tensor& gt,const at::Tensor& ids,const at::Tensor& w,const at::Tensor& lut){return Run(x,up,ws,dw,cp,cn,hp,hn,gt,ids,w,lut).back();}
std::vector<at::Tensor> splitquant_moe_fused_debug(const at::Tensor& x,const at::Tensor& up,const at::Tensor& ws,const at::Tensor& dw,const at::Tensor& cp,const at::Tensor& cn,const at::Tensor& hp,const at::Tensor& hn,const at::Tensor& gt,const at::Tensor& ids,const at::Tensor& w,const at::Tensor& lut){return Run(x,up,ws,dw,cp,cn,hp,hn,gt,ids,w,lut);}
}
