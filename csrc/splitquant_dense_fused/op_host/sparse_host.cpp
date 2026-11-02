// BSD-3-Clause. Independent  sparse decode scheduling and immutable weight cache.
#include "torch_kernel_helper.h"
#include "weight_scale_cache.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"
#include "aclrtlaunch_splitquant_sparse_pack.h"
#include "aclrtlaunch_splitquant_sparse_cube.h"
#include "aclrtlaunch_splitquant_sparse_cube_nz.h"
#include "aclrtlaunch_splitquant_sparse_cube_const_nz.h"
#include "aclrtlaunch_splitquant_sparse_raw_nz.h"
#include "aclrtlaunch_splitquant_sparse_mix_bf16.h"
#include <map>
#include <tuple>
#include <cstdlib>
namespace splitquant {
static at::Tensor sparse_impl(const at::Tensor& a,const at::Tensor& w,int64_t projections,int64_t n,int64_t n0,int64_t n1,const at::Tensor* ws,const at::Tensor* scale,const at::Tensor* bias,const at::Tensor* output,uint32_t hb){
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform," platform missing");
 int64_t g=projections>1?w.size(0):w.size(1)*2/128,chunks=(g+15)/16;uint32_t gu=g,pu=projections,nu=n,n0u=n0,n1u=n1;
 uint32_t tn=512; if(const char* v=std::getenv("SPLITQUANT_TILE_N"))tn=std::atoi(v);
 TORCH_CHECK(tn==128||tn==256||tn==512||tn==1024," TILE_N must be128/256/512/1024");
 if(output){tn=g>16?128:256;if(g<=16){if(const char* v=std::getenv("SPLITQUANT_MIX_QKV_TILE_N"))tn=std::atoi(v);}TORCH_CHECK(tn==128||tn==256,"MIX tileN must be128/256");}
 uint32_t bn=tn,bk=128;if(const char* v=std::getenv("SPLITQUANT_BASE_N"))bn=std::min<uint32_t>(tn,std::atoi(v));if(const char* v=std::getenv("SPLITQUANT_BASE_K"))bk=std::atoi(v);
 TORCH_CHECK((bn==128||bn==256||bn==512||bn==1024)&&(bk==128||bk==256||bk==512)," dense base split invalid");
 auto sparse=a.dim()==4?a:at::empty({projections,chunks,16,1024},a.options());
 uint32_t qb=std::min<int64_t>(platform->GetCoreNumAiv(),g*projections);
 if(a.dim()!=4){EXEC_KERNEL_CMD(splitquant_sparse_pack,qb,a,sparse,gu,pu);}
 const bool nz_weight=std::getenv("SPLITQUANT_DENSE_NZ") && n%64==0;
 const bool raw=std::getenv("SPLITQUANT_RAW_CUBE")&&std::atoi(std::getenv("SPLITQUANT_RAW_CUBE"))==1&&!output&&nz_weight&&(g==16||g==32)&&tn==256&&n<=65536&&(projections==1?n%256==0:(n0%256==0&&n1%256==0&&(n-n0-n1)%256==0));
 if(raw){uint64_t l1=0,l0a=0,l0b=0,l0c=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::L1,l1);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_A,l0a);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_B,l0b);platform->GetCoreMemSize(platform_ascendc::CoreMemType::L0_C,l0c);TORCH_CHECK(l1>=81920&&l0a>=16384&&l0b>=65536&&l0c>=4096," raw Cube exceeds physical local memory");}
 at::Tensor weight=w;
 if(projections>1 || nz_weight){
  using Key=std::tuple<const void*,uint32_t,int,c10::StreamId,bool,bool>;
  struct Entry{at::Tensor source,value;};static thread_local std::map<Key,Entry> weights;
  Key key{w.unsafeGetTensorImpl(),w.is_inference()?0:w.unsafeGetTensorImpl()->version_counter().current_version(),w.device().index(),c10_npu::getCurrentNPUStream(w.device().index()).id(),nz_weight,raw};
  auto wi=weights.find(key);
  if(wi==weights.end()){
   auto row=(projections>1?w.permute({1,0,2}).contiguous().reshape({n,g*64}):w).to(at::kByte);
   at::Tensor value=row;
   if(raw){value=row.reshape({n/64,4,16,g*2,32}).permute({0,3,1,2,4}).contiguous();}
   else if(nz_weight){
    auto codes=at::stack({at::bitwise_and(row,15),at::bitwise_right_shift(row,4)},-1).reshape({n,g*128}).t().contiguous();
    auto packed=at::bitwise_or(codes.slice(1,0,n,2),at::bitwise_left_shift(codes.slice(1,1,n,2),4));
    value=packed.reshape({g*8,16,n/64,32}).permute({2,0,1,3}).contiguous();
   }
   wi=weights.emplace(key,Entry{w,value}).first;
  }
  weight=wi->second.value;
 }
 using Key=std::tuple<int64_t,int64_t,int64_t,bool,int,c10::StreamId,int64_t,int64_t>;static thread_local std::map<Key,at::Tensor> cache;
 Key key{g,tn,n,nz_weight,a.device().index(),c10_npu::getCurrentNPUStream(a.device().index()).id(),bn,bk};auto it=cache.find(key);
 if(it==cache.end()){
  matmul_tiling::MatmulApiTiling api(*platform);using namespace matmul_tiling;
  TORCH_CHECK(api.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_INT4,false)==0&&api.SetBType(TPosition::GM,nz_weight?CubeFormat::NZ:CubeFormat::ND,DataType::DT_INT4,!nz_weight)==0&&api.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_INT32)==0," sparse types");
  TORCH_CHECK(api.SetShape(16,tn,2048)==0&&api.SetOrgShape(16,n,2048)==0&&api.SetFixSplit(16,bn,bk)==0," sparse tiling shape");
  optiling::TCubeTiling tile{};TORCH_CHECK(api.GetTiling(tile)==0," sparse tiling failed");tile.set_M(16);tile.set_N(tn);tile.set_Ka(2048);tile.set_Kb(2048);
  auto cpu=at::empty({int64_t(tile.GetDataSize())},at::TensorOptions().dtype(at::kByte).device(at::kCPU));tile.SaveToBuffer(cpu.data_ptr(),tile.GetDataSize());it=cache.emplace(key,cpu.to(a.device(),false,true)).first;
 }
 auto tiling=it->second;auto partial=at::empty({g,1,n},a.options().dtype(at::kInt));auto work=at::empty({int64_t(platform->GetLibApiWorkSpaceSize())+4096},a.options());
 int64_t tasks=chunks*(projections==1?(n+tn-1)/tn:((n0+tn-1)/tn+(n1+tn-1)/tn+(n-n0-n1+tn-1)/tn));uint32_t cb=std::min<int64_t>(platform->GetCoreNumAic(),tasks);
 if(output){
  TORCH_CHECK(nz_weight&&a.dim()==4,"MIX requires NZ and fused packed A");
  int64_t mix_tasks=projections==1?(n+tn-1)/tn:((n0+tn-1)/tn+(n1+tn-1)/tn+(n-n0-n1+tn-1)/tn);
  uint32_t cores=std::min<int64_t>(platform->GetCoreNumAic(),mix_tasks);auto sw=*ws,ss=*scale,bb=*bias,yy=*output;
  if(std::getenv("SPLITQUANT_CACHE_WEIGHT_SCALE")&&std::atoi(std::getenv("SPLITQUANT_CACHE_WEIGHT_SCALE"))==1){sw=rounded_weight_scale(sw,at::kBFloat16);hb|=2;}
  EXEC_KERNEL_CMD(splitquant_sparse_mix_bf16,cores,sparse,weight,partial,sw,ss,bb,yy,nu,gu,pu,n0u,n1u,tn,cores,hb,work,tiling);return yy;
 }
 bool fixed=std::getenv("SPLITQUANT_CONST_MATMUL")&&std::atoi(std::getenv("SPLITQUANT_CONST_MATMUL"))==1&&nz_weight&&(g==16||g==32)&&tn==256&&bn==128&&bk==256&&(projections==1?n%256==0:(n0%256==0&&n1%256==0&&(n-n0-n1)%256==0));
 if(raw){cb=std::min<int64_t>(platform->GetCoreNumAic(),chunks*(n/64));EXEC_KERNEL_CMD(splitquant_sparse_raw_nz,cb,sparse,weight,partial,nu,gu,pu,n0u,n1u,tn,work,tiling);}
 else if(fixed){EXEC_KERNEL_CMD(splitquant_sparse_cube_const_nz,cb,sparse,weight,partial,nu,gu,pu,n0u,n1u,tn,work,tiling);}
 else if(nz_weight){EXEC_KERNEL_CMD(splitquant_sparse_cube_nz,cb,sparse,weight,partial,nu,gu,pu,n0u,n1u,tn,work,tiling);}
 else{EXEC_KERNEL_CMD(splitquant_sparse_cube,cb,sparse,weight,partial,nu,gu,pu,n0u,n1u,tn,work,tiling);}
 return partial;
}
at::Tensor sparse_cube(const at::Tensor& a,const at::Tensor& w,int64_t p,int64_t n,int64_t n0,int64_t n1){return sparse_impl(a,w,p,n,n0,n1,nullptr,nullptr,nullptr,nullptr,0);}
at::Tensor sparse_cube_epi(const at::Tensor& a,const at::Tensor& w,int64_t p,int64_t n,int64_t n0,int64_t n1,const at::Tensor& ws,const at::Tensor& s,const at::Tensor& bias,const at::Tensor& output,uint32_t hb){return sparse_impl(a,w,p,n,n0,n1,&ws,&s,&bias,&output,hb);}

at::Tensor raw_cube_debug(const at::Tensor& a,const at::Tensor& b,int64_t n,int64_t g){
 TORCH_CHECK(a.device().is_privateuseone()&&b.device()==a.device()&&a.is_contiguous()&&b.is_contiguous()&&a.scalar_type()==at::kByte&&b.scalar_type()==at::kByte,"raw debug requires contiguous same-device uint8 inputs");
 TORCH_CHECK((g==16||g==32)&&n>0&&n<=65536&&n%256==0&&a.numel()==g/16*16384&&b.numel()==n*g*64,"raw debug geometry");
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"raw debug platform");
 auto out=at::empty({g,n},a.options().dtype(at::kInt));auto work=at::empty({32},a.options());auto tiling=work;
 uint32_t nu=n,gu=g,pu=1,n0=0,n1=0,tn=256,cb=std::min<int64_t>(platform->GetCoreNumAic(),(g/16)*(n/64));
 if(const char* v=std::getenv("SPLITQUANT_RAW_DEBUG_CHECKPOINT"))tn=std::atoi(v);
 EXEC_KERNEL_CMD(splitquant_sparse_raw_nz,cb,a,b,out,nu,gu,pu,n0,n1,tn,work,tiling);return out;
}

}
