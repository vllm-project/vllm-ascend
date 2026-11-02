// BSD-3-Clause. Public prepared-bundle ABI, no changes to previous operator schemas.
#include "torch_kernel_helper.h"
#include "static_metadata.h"
#include "aclrtlaunch_splitquant_multi_epi_zero.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"
#include "aclrtlaunch_splitquant_multi_quant.h"
#include "aclrtlaunch_splitquant_multi_quant_plain.h"
#include "aclrtlaunch_splitquant_multi_cube.h"
#include "aclrtlaunch_splitquant_multi_epi.h"
#include <map>
#include <tuple>
#include <algorithm>
#include <cstdlib>
namespace splitquant {
at::Tensor sparse_cube(const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,int64_t);
at::Tensor sparse_cube_epi(const at::Tensor&,const at::Tensor&,int64_t,int64_t,int64_t,int64_t,const at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,uint32_t);

at::Tensor nd_storage(const at::Tensor&);
at::Tensor grouped_rotation(const at::Tensor&,const at::Tensor&);
at::Tensor multi(const at::Tensor& input,const at::Tensor& w,const at::Tensor& ws,const at::Tensor& wo,const at::Tensor& cp,const at::Tensor& cn,const at::Tensor& gt,at::IntArrayRef splits){
 TORCH_CHECK(input.dim()==2 && input.scalar_type()==at::kBFloat16 && input.is_contiguous(),"multi requires contiguous BF16 [M,K]");
 int64_t m=input.size(0),k=input.size(1),g=k/128,p=splits.size(),n=0;
 TORCH_CHECK(m>0 && k>0 && k%128==0 && (p==2 || p==3),"invalid M/K/projection count");
 for(auto size:splits){TORCH_CHECK(size>0 && size%128==0,"split must be positive multiple of128");n+=size;}
 TORCH_CHECK(w.dim()==3 && w.size(0)==g && w.size(1)==n && w.size(2)==64 && (w.scalar_type()==at::kChar || w.scalar_type()==at::kByte),"W must be INT4 bytes [G,N,64]");
 TORCH_CHECK(ws.dim()==2 && ws.size(0)==g && ws.size(1)==n && wo.sizes()==ws.sizes(),"invalid grouped scale/offset");
 TORCH_CHECK(cp.dim()==2 && cp.size(0)==p && cp.size(1)==g && cn.sizes()==cp.sizes(),"invalid independent clip[P,G]");
 TORCH_CHECK(gt.dim()==3 && gt.size(0)==g && gt.size(1)==128 && gt.size(2)==128 && gt.scalar_type()==at::kBFloat16,"invalid transposed rotation");
 for(const auto* t:{&w,&ws,&wo,&cp,&cn,&gt})TORCH_CHECK(t->device()==input.device() && t->is_contiguous(),"multi tensors must be contiguous on one device");
 for(const auto* t:{&ws,&wo,&cp,&cn})TORCH_CHECK(t->scalar_type()==at::kFloat,"multi metadata requires FP32");
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"missing platform");
 uint64_t ub=0;platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB,ub);TORCH_CHECK(ub>=((m>=4 && n>=1024)?182272:((m>=4 && g>=32)?172224:138464)),"insufficient UB");
 auto x=nd_storage(input);
 const bool dense_rotation=std::getenv("SPLITQUANT_DENSE_ROTATION");
 if(dense_rotation){
  const bool grouped_cube=m==1 && std::getenv("SPLITQUANT_GROUPED_CUBE_ROTATION");
  using RK=std::tuple<const void*,uint32_t,int,c10::StreamId,bool>;struct RE{at::Tensor source,merged;};static thread_local std::map<RK,RE> rotations;
  RK rk{gt.unsafeGetTensorImpl(),gt.is_inference()?0:gt.unsafeGetTensorImpl()->version_counter().current_version(),x.device().index(),c10_npu::getCurrentNPUStream(x.device().index()).id(),grouped_cube};auto ri=rotations.find(rk);
  if(ri==rotations.end()){
   auto original=gt.transpose(1,2).contiguous();at::Tensor merged;
   if(grouped_cube)merged=original;
   else {merged=at::zeros({k,k},x.options());for(int64_t j=0;j<g;++j)merged.narrow(0,j*128,128).narrow(1,j*128,128).copy_(original.select(0,j));}
   ri=rotations.emplace(rk,RE{gt,merged.contiguous()}).first;
  }
  x=grouped_cube?grouped_rotation(x,ri->second.merged):at::matmul(x,ri->second.merged).contiguous();
 }

 using Key=std::tuple<int64_t,int64_t,int64_t,int64_t,int,c10::StreamId>;
 static thread_local std::map<Key,at::Tensor> tiles;
 Key key{m,n,g,p,input.device().index(),c10_npu::getCurrentNPUStream(input.device().index()).id()};
 auto it=tiles.find(key);
 if(it==tiles.end()){
  matmul_tiling::MatmulApiTiling api(*platform);
  TORCH_CHECK(api.SetAType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT4,false)==0 && api.SetBType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT4,true)==0 && api.SetCType(matmul_tiling::TPosition::GM,matmul_tiling::CubeFormat::ND,matmul_tiling::DataType::DT_INT32)==0 && api.SetShape(16,1024,128)==0 && api.SetOrgShape(p*m,n,128)==0 && api.SetFixSplit(16,1024,128)==0,"multi tiling setup failed");
  optiling::TCubeTiling cube{};TORCH_CHECK(api.GetTiling(cube)==0,"multi tiling failed");cube.set_M(16);cube.set_N(1024);cube.set_Ka(128);cube.set_Kb(128);
  auto cpu=at::empty({static_cast<int64_t>(cube.GetDataSize())},at::TensorOptions().dtype(at::kByte).device(at::kCPU));cube.SaveToBuffer(cpu.data_ptr(),cube.GetDataSize());
  it=tiles.emplace(key,cpu.to(input.device(),false,true)).first;
 }
 auto tiling=it->second;
 const bool fused_pack=m==1 && std::getenv("SPLITQUANT_FUSED_PACK") && !std::getenv("SPLITQUANT_DISABLE_SPARSE");
 auto a=fused_pack?at::empty({p,(g+15)/16,16,1024},x.options().dtype(at::kByte)):at::empty({g,p*m,64},x.options().dtype(at::kByte));auto s=at::empty({p*m,g,8},x.options().dtype(at::kFloat));auto sum=at::empty_like(s);
 auto partial=at::empty({g,m,n},x.options().dtype(at::kInt));auto y=at::empty({m,n},x.options());auto bias=at::empty({1},x.options());
 auto workspace=at::empty({static_cast<int64_t>(platform->GetLibApiWorkSpaceSize())+4096},x.options().dtype(at::kByte));
 uint32_t mu=m,quant_mu=static_cast<uint32_t>(m)|(fused_pack?0x80000000U:0),ku=k,nu=n,gu=g,pu=p,n0=splits[0],n1=splits[1],hb=0;
 uint32_t qb=std::min<int64_t>(platform->GetCoreNumAiv(),(m>=4 && g>=32)?g*((m+7)/8):m*g),eb=std::min<int64_t>(platform->GetCoreNumAiv(),((m>=4 && n>=1024)?((m+3)/4):m)*((n+127)/128)),cb=std::min<int64_t>(platform->GetCoreNumAic(),g*((m+15)/16)*((n+1023)/1024+p-1));
 if(dense_rotation){EXEC_KERNEL_CMD(splitquant_multi_quant_plain,qb,x,gt,cp,cn,a,s,sum,quant_mu,ku,pu);}
 else {EXEC_KERNEL_CMD(splitquant_multi_quant,qb,x,gt,cp,cn,a,s,sum,quant_mu,ku,pu);}
 if(m==1&&a.dim()==4&&g<=32&&std::getenv("SPLITQUANT_DENSE_NZ")&&std::getenv("SPLITQUANT_MIX_DENSE_EPI")&&std::atoi(std::getenv("SPLITQUANT_MIX_DENSE_EPI"))==1&&use_zero_epilogue(wo,m))return sparse_cube_epi(a,w,p,n,n0,n1,ws,s,bias,y,0);
 if(m==1 && !std::getenv("SPLITQUANT_DISABLE_SPARSE"))partial=sparse_cube(a,w,p,n,n0,n1);
 else {EXEC_KERNEL_CMD(splitquant_multi_cube,cb,a,w,partial,mu,nu,gu,pu,n0,n1,workspace,tiling);}
 if(use_zero_epilogue(wo,m)){EXEC_KERNEL_CMD(splitquant_multi_epi_zero,eb,partial,ws,wo,s,sum,bias,y,mu,nu,gu,hb,n0,n1);}
 else{EXEC_KERNEL_CMD(splitquant_multi_epi,eb,partial,ws,wo,s,sum,bias,y,mu,nu,gu,hb,n0,n1);}
 return y;
}
}
