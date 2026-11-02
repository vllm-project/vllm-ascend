#include "torch_kernel_helper.h"
#include <ATen/core/dispatch/Dispatcher.h>
#include "tiling/platform/platform_ascendc.h"
#include "tiling/matrix/matmul_tiling.h"
#include "aclrtlaunch_splitquant_rotate_bf16.h"
#include <map>
#include <tuple>
#include <algorithm>
namespace splitquant {
at::Tensor nd_storage(const at::Tensor& value) {
 static auto format=c10::Dispatcher::singleton().findSchemaOrThrow("npu::get_npu_format", "").typed<int64_t(const at::Tensor&)>();
 int64_t f=format.call(value);
 if(f==0 || f==2)return value;
 static auto cast=c10::Dispatcher::singleton().findSchemaOrThrow("npu::npu_format_cast", "").typed<at::Tensor(const at::Tensor&,int64_t,c10::optional<int64_t>)>();
 return cast.call(value,2,c10::nullopt);
}
at::Tensor grouped_rotation(const at::Tensor& input,const at::Tensor& rotation) {
 auto x=nd_storage(input);
 TORCH_CHECK(x.dim()==2 && x.scalar_type()==at::kBFloat16 && x.is_contiguous() && x.size(0)>0 && x.size(1)>0 && x.size(1)%128==0,"V3 rotation requires contiguous BF16 [M,128G]");
 const int64_t m=x.size(0),k=x.size(1),g=k/128;
 TORCH_CHECK(rotation.dim()==3 && rotation.size(0)==g && rotation.size(1)==128 && rotation.size(2)==128 && rotation.device()==x.device(),"invalid group transform");
 auto r=rotation.to(x.scalar_type()).contiguous();
 r=nd_storage(r);
 auto* platform=platform_ascendc::PlatformAscendCManager::GetInstance();TORCH_CHECK(platform,"cannot query platform");
 using Key=std::tuple<int64_t,int64_t,int,c10::StreamId>;
 static thread_local std::map<Key,at::Tensor> cache;
 Key key{m,k,x.device().index(),c10_npu::getCurrentNPUStream(x.device().index()).id()};
 auto it=cache.find(key);
 if(it==cache.end()) {
  using namespace matmul_tiling;
  MatmulApiTiling api(*platform);
  TORCH_CHECK(api.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_BF16,false)==0 &&
   api.SetBType(TPosition::GM,CubeFormat::ND,DataType::DT_BF16,false)==0 &&
   api.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_BF16)==0 &&
   api.SetShape(16,128,128)==0 && api.SetOrgShape(m,128,k)==0 && api.SetFixSplit(16,128,128)==0,"cannot configure grouped rotation");
  optiling::TCubeTiling cube{};TORCH_CHECK(api.GetTiling(cube)==0,"cannot tile grouped rotation");
  cube.set_M(16);cube.set_N(128);cube.set_Ka(128);cube.set_Kb(128);
  auto cpu=at::empty({static_cast<int64_t>(cube.GetDataSize())},at::TensorOptions().dtype(at::kByte).device(at::kCPU));
  cube.SaveToBuffer(cpu.data_ptr(),cube.GetDataSize());it=cache.emplace(key,cpu.to(x.device(),false,true)).first;
 }
 auto y=at::empty_like(x);auto tiling=it->second;
 auto workspace=at::empty({static_cast<int64_t>(platform->GetLibApiWorkSpaceSize())+4096},x.options().dtype(at::kByte));
 uint32_t mu=m,ku=k,blocks=std::min<int64_t>(platform->GetCoreNumAic(),g*((m+15)/16));
 EXEC_KERNEL_CMD(splitquant_rotate_bf16,blocks,x,r,y,mu,ku,workspace,tiling);
 return y;
}
}
TORCH_LIBRARY_FRAGMENT(npu,m) {m.def("splitquant_rotation(Tensor x, Tensor rotation) -> Tensor");}
TORCH_LIBRARY_IMPL(npu,PrivateUse1,m) {m.impl("splitquant_rotation",TORCH_FN(splitquant::grouped_rotation));}
