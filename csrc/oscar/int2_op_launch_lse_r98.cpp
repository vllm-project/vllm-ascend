// R98 LSE test launcher: calls the SIBLING op OscarInt2FusedAttnInt2Lse which
// outputs both attention_out AND softmax_lse. Verifies LSE on device.
#include <torch/extension.h>
#include <acl/acl.h>
#include <vector>
#include <cstring>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <c10/core/DeviceGuard.h>
#include "torch_npu/csrc/core/npu/NPUGuard.h"
#include "aclnn_oscar_int2_fused_attn_int2_lse.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
namespace py = pybind11;
static aclDataType acl_dtype(at::ScalarType t){ switch(t){
    case at::ScalarType::BFloat16:return ACL_BF16; case at::ScalarType::Char: return ACL_INT8;
    case at::ScalarType::Byte: return ACL_UINT8; case at::ScalarType::Int: return ACL_INT32;
    case at::ScalarType::Bool: return ACL_BOOL; case at::ScalarType::Float: return ACL_FLOAT;
    default: return ACL_DT_UNDEFINED;}}
static aclTensor* make_acl(const at::Tensor& t){
    auto s=t.sizes().vec(); auto st=t.strides().vec();
    return aclCreateTensor(s.data(),s.size(),acl_dtype(t.scalar_type()),st.data(),(int64_t)0,ACL_FORMAT_ND,(const int64_t*)nullptr,(uint64_t)0,t.data_ptr());}
struct LseDiag {
    int64_t getws_ret; int64_t ws_size; bool exec_nonnull;
    int64_t launch_ret; int64_t sync_ret;
    int64_t out_changed; int64_t lse_changed; int64_t lse_finite;
    torch::Tensor out_cpu; torch::Tensor lse_cpu;
};
static py::dict lse_diag_to_dict(const LseDiag& d){
    py::dict o;
    o["getws_ret"]=(int64_t)d.getws_ret; o["workspace_size"]=(int64_t)d.ws_size;
    o["executor_nonnull"]=d.exec_nonnull; o["launch_ret"]=(int64_t)d.launch_ret;
    o["sync_ret"]=(int64_t)d.sync_ret; o["out_changed"]=(int64_t)d.out_changed;
    o["lse_changed"]=(int64_t)d.lse_changed; o["lse_finite"]=(int64_t)d.lse_finite;
    if(d.out_cpu.defined()) o["out_cpu"]=d.out_cpu; else o["out_cpu"]=py::none();
    if(d.lse_cpu.defined()) o["lse_cpu"]=d.lse_cpu; else o["lse_cpu"]=py::none();
    o["output_valid"]=(d.getws_ret==0 && d.launch_ret==0 && d.sync_ret==0 && d.out_changed>0);
    return o;
}
static py::dict launch_lse(
        torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor bt,
        torch::Tensor kScale,torch::Tensor kOff,torch::Tensor vScale,torch::Tensor vOff,
        torch::Tensor out, torch::Tensor lse,
        int64_t numHeads,double scale,int64_t numKvHeads,std::string layout,int64_t blockSize,
        int64_t keyAntiqMode,int64_t valAntiqMode, std::vector<int64_t> askv,
        torch::Tensor attenMask, int64_t sparseMode,
        int64_t preTokens, int64_t nextTokens, bool launchKernel) {
    const c10::DeviceGuard device_guard(q.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(q.device().index()));
    LseDiag d{0,0,false,-1,-1,0,0,0,torch::Tensor(),torch::Tensor()};
    aclTensor *a_q=make_acl(q),*a_k=make_acl(k),*a_v=make_acl(v),*a_bt=make_acl(bt);
    aclTensor *a_ks=make_acl(kScale),*a_ko=make_acl(kOff),*a_vs=make_acl(vScale),*a_vo=make_acl(vOff);
    aclTensor *a_o=make_acl(out),*a_lse=make_acl(lse);
    aclTensor *a_mask=nullptr;
    if(attenMask.numel()>0) a_mask=make_acl(attenMask);
    int64_t B=q.size(0); std::vector<int64_t> asq(B,q.size(2));
    aclIntArray *a_asq=aclCreateIntArray(asq.data(),B),*a_askv=aclCreateIntArray(askv.data(),B);
    std::vector<char> lb(layout.begin(),layout.end()); lb.push_back('\0');
    aclrtStream kstream=c10_npu::getCurrentNPUStream(q.device().index()).stream(false);
    c10_npu::npuSynchronizeDevice(true);
    // Pre-fill lse with sentinel to detect write
    lse.fill_(torch::Scalar(-9999.0));
    aclrtSynchronizeStream(kstream);
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus r=aclnnOscarInt2FusedAttnInt2LseGetWorkspaceSize(
        a_q,a_k,a_v,/*pse*/nullptr,a_mask,a_asq,a_askv,
        /*deq1*/nullptr,/*qs1*/nullptr,/*deq2*/nullptr,/*qs2*/nullptr,/*qo2*/nullptr,
        /*antiquantScale*/nullptr,/*antiquantOffset*/nullptr,
        a_bt,/*qpad*/nullptr,/*kvpad*/nullptr,
        a_ks,a_ko,a_vs,a_vo,
        numHeads,scale,preTokens,nextTokens,lb.data(),numKvHeads,
        sparseMode,/*innerPrecise*/0,blockSize,/*antiquantMode*/1,
        keyAntiqMode,valAntiqMode,
        a_o,a_lse,&ws,&ex);
    d.getws_ret=(int64_t)r; d.ws_size=(int64_t)ws; d.exec_nonnull=(ex!=nullptr);
    fprintf(stderr,"[lse_launch] getws=%d ws=%lu exec=%d askv=[",(int)r,(unsigned long)ws,(int)(ex!=nullptr));
    for(size_t i=0;i<askv.size();i++) fprintf(stderr,"%ld%s",(long)askv[i],i+1<askv.size()?",":"");
    fprintf(stderr,"] sparse=%ld mask=%s\n",(long)sparseMode,a_mask?"PROVIDED":"nullptr");
    if(r==0 && launchKernel){
        at::Tensor wsh; void* ws_p=nullptr;
        if(ws>0){
            try {
                wsh=torch::empty({(int64_t)ws},torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
                ws_p=wsh.data_ptr();
            } catch(...) {
                aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
                aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
                aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
                if(a_mask) aclDestroyTensor(a_mask);
                aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
                throw;
            }
        }
        aclrtStream s=kstream;
        aclnnStatus lr=aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);
        aclError sr=aclrtSynchronizeStream(s);
        d.launch_ret=(int64_t)lr; d.sync_ret=(int64_t)sr;
        fprintf(stderr,"[lse_launch] launch=%d sync=%d\n",(int)lr,(int)sr);
    }
    if(r==0 && launchKernel){
        d.out_cpu=out.to(torch::kCPU).contiguous();
        d.lse_cpu=lse.to(torch::kCPU).contiguous();
        d.out_changed=(int64_t)torch::ne(d.out_cpu, torch::full_like(d.out_cpu,-11333.0)).sum().item<int64_t>();
        d.lse_changed=(int64_t)torch::ne(d.lse_cpu, torch::full_like(d.lse_cpu,-9999.0)).sum().item<int64_t>();
        d.lse_finite=(int64_t)torch::isfinite(d.lse_cpu).sum().item<int64_t>();
    }
    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
    if(a_mask) aclDestroyTensor(a_mask);
    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
    return lse_diag_to_dict(d);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("r98_lse_version",[](){return std::string("r98.lse.v1");});
    m.def("launch_lse",&launch_lse,"R98 LSE: calls sibling op with softmax_lse output");
}
