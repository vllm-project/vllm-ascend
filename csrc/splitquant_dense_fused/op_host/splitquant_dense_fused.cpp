#include "torch_kernel_helper.h"
#include "static_metadata.h"
#include "aclrtlaunch_splitquant_epi_zero_fp16.h"
#include "aclrtlaunch_splitquant_epi_zero_bf16.h"
#include <ATen/core/dispatch/Dispatcher.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "tiling/platform/platform_ascendc.h"
#include "aclrtlaunch_splitquant_quant_fp16.h"
#include "aclrtlaunch_splitquant_quant_bf16.h"
#include "aclrtlaunch_splitquant_quant_stable_bf16.h"
#include "aclrtlaunch_splitquant_quant_rotate_fp16.h"
#include "aclrtlaunch_splitquant_quant_rotate_bf16.h"
#include <cstdlib>
#include <string>
#include "aclrtlaunch_splitquant_quant_tile_fp16.h"
#include "aclrtlaunch_splitquant_quant_tile_bf16.h"
#include "aclrtlaunch_splitquant_epi_fp16.h"
#include "aclrtlaunch_splitquant_epi_bf16.h"
#include <c10/core/InferenceMode.h>
#include <map>
#include <tuple>
#include <algorithm>
#include <cstdint>
namespace splitquant {
at::Tensor decode(const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,const c10::optional<at::Tensor>&,const at::Tensor&);
at::Tensor nd_storage(const at::Tensor&);
at::Tensor grouped_rotation(const at::Tensor&,const at::Tensor&);
at::Tensor sparse_cube_epi(const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,int64_t,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t);

at::Tensor splitquant_grouped_cube(const at::Tensor&, const at::Tensor&);
static at::Tensor splitquant_pipeline(bool use_catlass,const at::Tensor& x,const at::Tensor& w,
 const at::Tensor& ws,const at::Tensor& wo,const at::Tensor& cp,const at::Tensor& cn,
 const c10::optional<at::Tensor>& bias,int64_t group_size,
 const c10::optional<at::Tensor>& gt,const c10::optional<at::Tensor>& rt,const c10::optional<at::Tensor>& lt){
 TORCH_CHECK(x.dim()==2 && (x.scalar_type()==at::kHalf || x.scalar_type()==at::kBFloat16) && x.is_contiguous(),"Cube SplitQuant requires contiguous FP16/BF16 [M,K]");
 int64_t m=x.size(0),k=x.size(1),g=k/128,n=w.size(0);
 TORCH_CHECK(group_size==128 && k>0 && k%128==0 && w.dim()==2 && w.size(1)==k/2 && n%16==0 && n>0,"invalid Cube SplitQuant packed shape");
 TORCH_CHECK(ws.dim()==2 && ws.size(0)==n && ws.size(1)==g && wo.sizes()==ws.sizes() && cp.numel()==g && cn.numel()==g,"invalid SplitQuant metadata shape");
 for(const auto* t:{&w,&ws,&wo,&cp,&cn})TORCH_CHECK(t->device()==x.device() && t->is_contiguous(),"SplitQuant tensors must be contiguous on same device");
 TORCH_CHECK(ws.scalar_type()==at::kFloat && wo.scalar_type()==at::kFloat && cp.scalar_type()==at::kFloat && cn.scalar_type()==at::kFloat,"SplitQuant metadata must be FP32");
 TORCH_CHECK(w.scalar_type()==at::kChar || w.scalar_type()==at::kByte,"packed W4 requires byte storage");
 if(bias)TORCH_CHECK(bias->dim()==1 && bias->numel()==n && bias->scalar_type()==x.scalar_type() && bias->device()==x.device() && bias->is_contiguous(),"invalid SplitQuant bias");
 using Key=std::tuple<const void*,uint32_t,const void*,uint32_t>;
 using InferenceKey=std::tuple<const void*,const void*>;
 struct Prepared {at::Tensor wsref,woref,scales,offsets;};
 static thread_local std::map<Key,Prepared> prepared;
 static thread_local std::map<InferenceKey,Prepared> inference_prepared;
 at::Tensor scales, offsets;
 if(c10::InferenceMode::is_enabled() || ws.is_inference() || wo.is_inference()) {
  InferenceKey key{ws.unsafeGetTensorImpl(),wo.unsafeGetTensorImpl()};
  auto it=inference_prepared.find(key);
  if(it==inference_prepared.end()) {
   it=inference_prepared.emplace(key,Prepared{ws,wo,ws.t().contiguous(),wo.to(at::kInt).to(at::kFloat).t().contiguous()}).first;
  }
  scales=it->second.scales; offsets=it->second.offsets;
 } else {
  auto version=[](const at::Tensor& t){return t.unsafeGetTensorImpl()->version_counter().current_version();};
  Key key{ws.unsafeGetTensorImpl(),version(ws),wo.unsafeGetTensorImpl(),version(wo)};
  auto it=prepared.find(key);
  if(it==prepared.end()){
  TORCH_CHECK(at::isfinite(ws).all().item<bool>() && at::isfinite(wo).all().item<bool>() && wo.abs().max().item<float>()<=8192,"invalid scale/offset (offset magnitude must be <=8192)");
  it=prepared.emplace(key,Prepared{ws,wo,ws.t().contiguous(),wo.to(at::kInt).to(at::kFloat).t().contiguous()}).first;
  }
  scales=it->second.scales; offsets=it->second.offsets;
 }
 const bool vector_tile=std::getenv("SPLITQUANT_FUSED_ROTATION") && std::string(std::getenv("SPLITQUANT_FUSED_ROTATION"))=="tile";
 const bool stable_rotation=x.scalar_type()==at::kBFloat16 && gt && !rt && !lt && !std::getenv("SPLITQUANT_DENSE_ROTATION");
 const bool fused_rotation=use_catlass && gt && !rt && !lt && std::getenv("SPLITQUANT_FUSED_ROTATION");
 auto h=x;
 if(gt && !fused_rotation && !stable_rotation){
  TORCH_CHECK(gt->dim()==3 && gt->size(0)==g && gt->size(1)==128 && gt->size(2)==128,"invalid group transform");
  // The exported rotations are block diagonal in the flattened K space.
  // Materialize that matrix once per immutable transform and use one GEMM;
  // quantization still reduces each original 128-wide group independently.
  using TransformKey=std::tuple<const void*,int,int,uint32_t>;
  struct TransformEntry {at::Tensor source, merged;};
  static thread_local std::map<TransformKey,TransformEntry> merged_transforms;
  auto gt_cast=gt->to(x.scalar_type());
  TransformKey transform_key{gt->unsafeGetTensorImpl(),static_cast<int>(x.scalar_type()),static_cast<int>(x.device().index()),gt->is_inference()?0:gt->unsafeGetTensorImpl()->version_counter().current_version()};
  auto merged_it=merged_transforms.find(transform_key);
  if(merged_it==merged_transforms.end()) {
   auto merged=at::zeros({k,k},x.options());
   for(int64_t i=0;i<g;++i)
    merged.narrow(0,i*128,128).narrow(1,i*128,128).copy_(gt_cast.select(0,i));
   merged_it=merged_transforms.emplace(transform_key,TransformEntry{*gt,merged.contiguous()}).first;
  }
  h=at::matmul(h,merged_it->second.merged);
 }
 if(rt){TORCH_CHECK(rt->dim()==2 && rt->size(0)==k && rt->size(1)==k,"invalid right transform");h=at::matmul(h,rt->to(x.scalar_type()));}
 if(lt){
  int64_t d=lt->size(0);TORCH_CHECK(lt->dim()==2 && lt->size(1)==d && d>0 && k%d==0,"invalid left transform");
  using LK=std::tuple<const void*,uint32_t,int,int,c10::StreamId>;struct LE{at::Tensor source,transposed;};static thread_local std::map<LK,LE> left_cache;
  LK key{lt->unsafeGetTensorImpl(),lt->is_inference()?0:lt->unsafeGetTensorImpl()->version_counter().current_version(),int(x.scalar_type()),x.device().index(),c10_npu::getCurrentNPUStream(x.device().index()).id()};auto li=left_cache.find(key);
  if(li==left_cache.end())li=left_cache.emplace(key,LE{*lt,nd_storage(lt->to(x.scalar_type()).t().contiguous())}).first;
  h=at::matmul(li->second.transposed,h.view({m,d,k/d})).reshape({m,k});
 }
 if(stable_rotation)h=nd_storage(h);
 h=h.contiguous();
 const bool fused_pack=m==1 && std::getenv("SPLITQUANT_FUSED_PACK") && !std::getenv("SPLITQUANT_DISABLE_SPARSE");
 auto a=fused_pack?at::empty({1,(g+15)/16,16,1024},x.options().dtype(at::kByte)):at::empty({g,m,64},x.options().dtype(at::kByte));
 auto s=at::empty({m,g,8},x.options().dtype(at::kFloat));auto asum=at::empty_like(s);
 auto *platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"cannot query platform");
 uint64_t ub_size=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub_size);
 TORCH_CHECK(ub_size>=((m>=4 && n>=1024 && n%128==0)?182272:((stable_rotation && m>=4 && g>=32)?172224:138464))," tiling exceeds device UB");
 uint32_t mu=m,quant_mu=static_cast<uint32_t>(m)|(fused_pack?0x80000000U:0),ku=k,nu=n,gu=g,hb=bias?1:0;
 uint32_t quant_blocks=std::max<int64_t>(1,std::min<int64_t>(platform->GetCoreNumAiv(),(stable_rotation && m>=4 && g>=32)?g*((m+7)/8):m*g));
 if(fused_rotation || stable_rotation){
  auto rotation=gt->to(x.scalar_type()).contiguous();
  if(vector_tile || stable_rotation){
   using TileKey=std::tuple<const void*,int,uint32_t>;
   struct TileTransform {at::Tensor source, value;};
   static thread_local std::map<TileKey,TileTransform> tile_transforms;
   TileKey key{gt->unsafeGetTensorImpl(),static_cast<int>(x.scalar_type()),gt->is_inference()?0:gt->unsafeGetTensorImpl()->version_counter().current_version()};
   auto it=tile_transforms.find(key);
   if(it==tile_transforms.end()) it=tile_transforms.emplace(key,TileTransform{*gt,nd_storage(rotation.transpose(1,2).contiguous())}).first;
   rotation=it->second.value;
   if(stable_rotation){EXEC_KERNEL_CMD(splitquant_quant_stable_bf16,quant_blocks,h,rotation,cp,cn,a,s,asum,quant_mu,ku);}
   else if(x.scalar_type()==at::kHalf){EXEC_KERNEL_CMD(splitquant_quant_tile_fp16,quant_blocks,h,rotation,cp,cn,a,s,asum,quant_mu,ku);}
   else {EXEC_KERNEL_CMD(splitquant_quant_tile_bf16,quant_blocks,h,rotation,cp,cn,a,s,asum,quant_mu,ku);}
  }
  else if(x.scalar_type()==at::kHalf){EXEC_KERNEL_CMD(splitquant_quant_rotate_fp16,quant_blocks,h,rotation,cp,cn,a,s,asum,quant_mu,ku);}
  else {EXEC_KERNEL_CMD(splitquant_quant_rotate_bf16,quant_blocks,h,rotation,cp,cn,a,s,asum,quant_mu,ku);}
 }
 else if(x.scalar_type()==at::kHalf){EXEC_KERNEL_CMD(splitquant_quant_fp16,quant_blocks,h,cp,cn,a,s,asum,quant_mu,ku);}
 else {EXEC_KERNEL_CMD(splitquant_quant_bf16,quant_blocks,h,cp,cn,a,s,asum,quant_mu,ku);}
 // The exact vector GEMV prototype is slower than Cube on measured 910B3
 // model projections. Keep it available for experiments, never select it by default.
 const char* gemv=std::getenv("SPLITQUANT_GEMV");
 if(m==1 && gemv && std::string(gemv)=="1")return decode(a,w,scales,offsets,s,asum,bias,x);
 if(m==1&&x.scalar_type()==at::kBFloat16&&fused_pack&&g<=32&&n%128==0&&std::getenv("SPLITQUANT_DENSE_NZ")&&std::getenv("SPLITQUANT_MIX_DENSE_EPI")&&std::atoi(std::getenv("SPLITQUANT_MIX_DENSE_EPI"))==1&&use_zero_epilogue(offsets,m)){
  auto yy=at::empty({m,n},x.options());auto bb=bias?*bias:at::empty({1},x.options());return sparse_cube_epi(a,w,1,n,0,0,scales,s,bb,yy,hb);
 }
 auto p=splitquant_grouped_cube(a,w);auto y=at::empty({m,n},x.options());auto b=bias?*bias:at::empty({1},x.options());
 uint32_t epi_blocks=std::max<int64_t>(1,std::min<int64_t>(platform->GetCoreNumAiv(),((m>=4 && n>=1024 && n%128==0)?((m+3)/4):m)*((n+127)/128)));
 if(use_zero_epilogue(offsets,m)){
  if(x.scalar_type()==at::kHalf){EXEC_KERNEL_CMD(splitquant_epi_zero_fp16,epi_blocks,p,scales,offsets,s,asum,b,y,mu,nu,gu,hb);}
  else{EXEC_KERNEL_CMD(splitquant_epi_zero_bf16,epi_blocks,p,scales,offsets,s,asum,b,y,mu,nu,gu,hb);}
 }
 else if(x.scalar_type()==at::kHalf){EXEC_KERNEL_CMD(splitquant_epi_fp16,epi_blocks,p,scales,offsets,s,asum,b,y,mu,nu,gu,hb);}
 else {EXEC_KERNEL_CMD(splitquant_epi_bf16,epi_blocks,p,scales,offsets,s,asum,b,y,mu,nu,gu,hb);}
 return y;
}
at::Tensor splitquant_dense_fused(const at::Tensor& x,const at::Tensor& w,const at::Tensor& ws,const at::Tensor& wo,
 const at::Tensor& cp,const at::Tensor& cn,const c10::optional<at::Tensor>& bias,int64_t group_size,
 const c10::optional<at::Tensor>& gt,const c10::optional<at::Tensor>& rt,const c10::optional<at::Tensor>& lt) {
 return splitquant_pipeline(true, x, w, ws, wo, cp, cn, bias, group_size, gt, rt, lt);
}

}
