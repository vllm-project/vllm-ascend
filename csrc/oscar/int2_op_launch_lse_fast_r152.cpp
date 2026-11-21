// R112 fast launcher: same op call sequence as int2_op_launch_lse_r110ws.cpp but
// ZERO device syncs, ZERO D2H, ZERO .item(), cached workspace, no fprintf in steady state.
// Caller-side contract: outputs consumed on-stream by the merge; status returned as ints.
// Optional verify=1 -> does the R110-style readback (sync) for correctness legs only.
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <acl/acl.h>
#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <string>
#include <filesystem>
#include <system_error>
#include <pybind11/pybind11.h>
#include <map>
#include <utility>
#include <pybind11/stl.h>
// R144: model-path counter dump. Gated by R144_COUNTER_DUMP env (default OFF).
// ONLY for the counter diagnostic .o (7e354b85, FORCE_GROUPED=1+COUNTER_MODE=1)
// which writes a per-core dequant/member table to attentionOutGm[:640].
// The counter corrupts the real attention output -> COUNTER LEG ISOLATED:
// never judge token/numerical correctness from a run with R144_COUNTER_DUMP=1.
// Adds ONE aclrtSynchronizeStream per call (diagnostic sync; this leg only).
static int64_t g_r144_call = 0;
#include "aclnn_oscar_int2_fused_attn_int2_lse.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "torch_npu/csrc/core/npu/NPUGuard.h"

// ===== R218-P7 (2026-10-25): torch_npu task-queue commit boundary =====
// torch_npu 2.10.0.post7 (git 3363ea59) submits ALL op_api/ATen host work
// through a per-device Repository task queue drained by a consumer thread
// (OpCommand.cpp RunOpApi -> enCurrentNPUStream -> Repository::Enqueue,
// NPUQueue.cpp ReadQueue executes + only then advances read_idx). A DIRECT
// aclnn* call from this extension submits immediately on the calling thread
// and can therefore OVERTAKE ops the consumer has not yet submitted -- both
// the caller's operand staging and the lse sentinel fill in this file.
// NPUStream::stream(false) (NPUStream.cpp:838) returns the raw handle with NO
// flush; the implicit operator aclrtStream() conversion and stream(true) call
// MakeSureQueueEmpty first (NPUQueue.cpp:243): the official host-side commit
// boundary. r218_qbound_flush establishes exactly that boundary: on return,
// every op enqueued before the call has been SUBMITTED to the stream
// (submission order only -- NOT a device synchronization).
#include <atomic>
#include <chrono>
static std::atomic<int64_t> g_qb_calls{0};
static std::atomic<int64_t> g_qb_ns{0};
static std::atomic<int64_t> g_qb_max_ns{0};
static std::atomic<int> g_qb_seen[8];
static inline aclrtStream r218_qbound_flush(c10::DeviceIndex dev){
    auto t0 = std::chrono::steady_clock::now();
    aclrtStream s = c10_npu::getCurrentNPUStream(dev).stream(true);
    int64_t dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - t0).count();
    g_qb_calls.fetch_add(1, std::memory_order_relaxed);
    g_qb_ns.fetch_add(dt, std::memory_order_relaxed);
    int64_t prev = g_qb_max_ns.load(std::memory_order_relaxed);
    while (dt > prev && !g_qb_max_ns.compare_exchange_weak(prev, dt, std::memory_order_relaxed)) {}
    return s;
}
static inline void r218_qbound_seen(int slot, const char* entry){
    if (!g_qb_seen[slot].exchange(1)) {
        std::fprintf(stderr, "[R218-QBOUND] entry=%s slot=%d ACTIVE\n", entry, slot);
        std::fflush(stderr);
    }
}
// ===== end R218-P7 helper =====
namespace py = pybind11;
static aclDataType acl_dtype(at::ScalarType t){ switch(t){
    case at::ScalarType::BFloat16:return ACL_BF16; case at::ScalarType::Char: return ACL_INT8;
    case at::ScalarType::Byte: return ACL_UINT8; case at::ScalarType::Int: return ACL_INT32;
    case at::ScalarType::Bool: return ACL_BOOL; case at::ScalarType::Float: return ACL_FLOAT;
    default: return ACL_DT_UNDEFINED;}}
static aclTensor* make_acl(const at::Tensor& t){
    auto s=t.sizes().vec(); auto st=t.strides().vec();
    return aclCreateTensor(s.data(),s.size(),acl_dtype(t.scalar_type()),st.data(),(int64_t)0,ACL_FORMAT_ND,(const int64_t*)nullptr,(uint64_t)0,t.data_ptr());}
static at::Tensor g_ws;                 // cached workspace (grown on demand)
static int64_t g_ws_hits = 0;
// R192 PREDUMP (DIAGNOSTIC_ONLY / NEVER_PROMOTE): dump the EXACT at::Tensor
// operands of OscarInt2FusedAttnInt2Lse RIGHT BEFORE the aclnn Run, so the
// operands of a FAULTING call survive on disk (R144_OPDUMP fires only after a
// clean lr==0 and syncs post-launch — the 507015 fault kills that path).
// Gated by R192_PREDUMP=1 (default OFF). Eager/diagnostic legs only: the D2H
// copies sync the current stream and are NOT graph-capture-safe. Whole block
// try/catch'd — a dump failure must never break the launch path.
static int64_t g_r192_call = 0;
static void r192_predump(const std::string& site, const at::Tensor* ts, const char* const* nm,
                         int64_t nb, int64_t mid_len, int64_t blockSize,
                         double scale, int64_t nq, int64_t nkv){
    try {
        const char* e = std::getenv("R192_PREDUMP");
        if(!e || std::string(e)!="1") return;
        int64_t mx = 4;
        if(const char* m = std::getenv("R192_PREDUMP_MAX")) mx = std::atoll(m);
        if(g_r192_call >= mx) return;
        std::string odir = std::getenv("R192_PREDUMP_DIR") ? std::getenv("R192_PREDUMP_DIR") : "/tmp/r192_livedump";
        std::error_code _fs_ec;
        std::filesystem::create_directories(odir, _fs_ec);
        auto to_cpu = [](const at::Tensor& t){ return t.contiguous().to(at::kCPU).contiguous(); };
        at::Tensor cpu[10];
        for(int i=0;i<10;i++) cpu[i] = to_cpu(ts[i]);
        std::string meta = "{\"call\":" + std::to_string(g_r192_call) + ",\"site\":\"" + site + "\"";
        meta += ",\"numHeads\":" + std::to_string(nq);
        meta += ",\"scale\":" + std::to_string(scale);
        meta += ",\"numKvHeads\":" + std::to_string(nkv);
        meta += ",\"blockSize\":" + std::to_string(blockSize);
        meta += ",\"mid_len\":" + std::to_string(mid_len);
        meta += ",\"nb\":" + std::to_string(nb);
        meta += ",\"ptrs\":[";
        char buf[96];
        for(int i=0;i<10;i++){
            uint64_t p = (uint64_t)ts[i].data_ptr();
            std::snprintf(buf,sizeof(buf),"%s{\"name\":\"%s\",\"ptr\":\"0x%lx\",\"m2\":%lu,\"a32\":%lu,\"a512\":%lu}",
                i?",":"",nm[i],(unsigned long)p,(unsigned long)(p%(2UL<<20)),
                (unsigned long)(p%32UL),(unsigned long)(p%512UL));
            meta += buf;
        }
        meta += "],\"tensors\":[";
        std::string path = odir + "/r192_pre_call" + std::to_string(g_r192_call) + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        uint64_t off = 0;
        for(int i=0;i<10;i++){
            at::Tensor c = cpu[i];
            if(i) meta += ",";
            std::string sh = "["; auto ss = c.sizes();
            for(int64_t j=0;j<(int)ss.size();j++){ if(j) sh+=","; sh+=std::to_string(ss[j]); }
            sh += "]";
            std::string stj = "["; auto sg = c.strides();
            for(int64_t j=0;j<(int)sg.size();j++){ if(j) stj+=","; stj+=std::to_string(sg[j]); }
            stj += "]";
            meta += "{\"name\":\""; meta += nm[i]; meta += "\",\"offset\":";
            meta += std::to_string(off); meta += ",\"nbytes\":";
            meta += std::to_string(c.nbytes()); meta += ",\"dtype\":\"";
            meta += std::string(c.dtype().name()); meta += "\",\"shape\":"; meta += sh;
            meta += ",\"stride\":"; meta += stj; meta += "}";
            if(f) std::fwrite(c.data_ptr(), 1, c.nbytes(), f);
            off += c.nbytes();
        }
        meta += "]}";
        if(f) std::fclose(f);
        std::string mp = odir + "/r192_pre_call" + std::to_string(g_r192_call) + ".json";
        FILE* fm = std::fopen(mp.c_str(), "w");
        if(fm){ std::fprintf(fm,"%s",meta.c_str()); std::fclose(fm); }
        std::fprintf(stderr,"[R192-PREDUMP] call=%lld site=%s nb=%lld mid=%lld -> %s\n",
            (long long)g_r192_call,site.c_str(),(long long)nb,(long long)mid_len,path.c_str());
        std::fflush(stderr);
        g_r192_call++;
    } catch (...) {
        // never break the launch path on a diagnostic failure
    }
}
static py::object launch_lse_fast(
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
    aclTensor *a_q=make_acl(q),*a_k=make_acl(k),*a_v=make_acl(v),*a_bt=make_acl(bt);
    aclTensor *a_ks=make_acl(kScale),*a_ko=make_acl(kOff),*a_vs=make_acl(vScale),*a_vo=make_acl(vOff);
    aclTensor *a_o=make_acl(out),*a_lse=make_acl(lse);
    aclTensor *a_mask=nullptr;
    if(attenMask.numel()>0) a_mask=make_acl(attenMask);
    int64_t B=q.size(0); std::vector<int64_t> asq(B,q.size(2));
    aclIntArray *a_asq=aclCreateIntArray(asq.data(),B),*a_askv=aclCreateIntArray(askv.data(),B);
    std::vector<char> lb(layout.begin(),layout.end()); lb.push_back('\0');
    aclrtStream s=c10_npu::getCurrentNPUStream(q.device().index()).stream(false);
    {   // R192 PREDUMP site: pre-launch operand capture (see r192_predump)
        at::Tensor _d[10] = {q,k,v,bt,kScale,kOff,vScale,vOff,out,lse};
        const char* _n[10] = {"q","k","v","bt","ks","ko","vs","vo","out","lse"};
        int64_t _nb = bt.dim()>=2 ? bt.size(1) : bt.size(0);
        int64_t _ml = askv.empty() ? 0 : askv[0];
        r192_predump("launch_lse_fast", _d, _n, _nb, _ml, blockSize, scale, numHeads, numKvHeads);
    }
    lse.fill_(torch::Scalar(-9999.0));   // async on same stream, ordered before the op

    // R218-P7 commit boundary: host task-queue flush AFTER the sentinel fill
    // (and any caller staging enqueued before this call), BEFORE the direct
    // ACLNN submit. Submission-order guarantee only; NOT a device sync.
    s = r218_qbound_flush(q.device().index());
    TORCH_CHECK(s != nullptr, "R218-P7: task-queue flush failed dev=", (int)q.device().index());
    r218_qbound_seen(0, "launch_lse_fast");
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
    TORCH_CHECK(r==0, "R218-P7: GetWorkspaceSize failed r=",(int64_t)r);
    TORCH_CHECK(ex!=nullptr, "R218-P7: GetWorkspaceSize returned null executor");
    int64_t lr=-1;
    if(launchKernel){
        void* ws_p=nullptr;
        if(ws>0){
            if(!g_ws.defined() || (uint64_t)g_ws.numel()<ws){
                try {
                    g_ws=torch::empty({(int64_t)ws},torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
                    g_ws_hits=0;
                } catch(...) {
                    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
                    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
                    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
                    if(a_mask) aclDestroyTensor(a_mask);
                    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
                    throw;
                }
            } else { g_ws_hits++; }
            ws_p=g_ws.data_ptr();
        }
        lr=(int64_t)aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);
        TORCH_CHECK(lr==0, "R218-P7: aclnn launch failed lr=",lr);
    }
    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
    if(a_mask) aclDestroyTensor(a_mask);
    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
    py::tuple res=py::make_tuple((int64_t)r,(int64_t)ws,(int64_t)lr,g_ws_hits);
    return res;
}
// R113: pre-grow g_ws OUTSIDE any aclgraph capture / task-update region.
// torch::empty inside those regions deadlocks/errors (allocator offset bump);
// replay updates with larger askv need a bigger workspace than the
// capture-time placeholder call sized. Call once during the eager warmup.
static py::object warm_ws(torch::Tensor dev_ref, int64_t bytes){
    const c10::DeviceGuard device_guard(dev_ref.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(dev_ref.device().index()));
    if(!g_ws.defined() || g_ws.numel()<bytes){
        g_ws=torch::empty({bytes},torch::TensorOptions().dtype(torch::kUInt8).device(dev_ref.device()));
        g_ws_hits=0;
    }
    return py::make_tuple((int64_t)g_ws.numel());
}
// R113 Fallback-1: single-call middle operand prep + launch. Replicates the
// exact eager op sequence (advanced-index gather, permute+contig+int8 view,
// scale tight-pack, neg-mul offsets, identity bt) with ATen calls INSIDE C++ —
// same kernels => bit-exact vs the Python path, but ONE pybind crossing
// instead of ~35 Python dispatches per layer per decode step.
static std::map<std::pair<int64_t,int64_t>, at::Tensor> g_btcache;
static py::object oscar_middle_prep(
        torch::Tensor block_table, int64_t b, int64_t blk_lo, int64_t nb, int64_t mid_len,
        torch::Tensor key_cache, torch::Tensor value_cache,
        torch::Tensor k_scale_cache, torch::Tensor k_zero_cache,
        torch::Tensor v_scale_cache, torch::Tensor v_zero_cache,
        torch::Tensor q, torch::Tensor out, torch::Tensor lse,
        int64_t numHeads, double scale, int64_t numKvHeads, int64_t blockSize){
    const c10::DeviceGuard device_guard(q.device());   // R113: see oscar_quant_kv
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(q.device().index()));
    auto dev = q.device();
    auto key = std::make_pair(nb, (int64_t)dev.index());
    auto it = g_btcache.find(key);
    if(it == g_btcache.end()){
        auto bt_t = torch::arange(nb, torch::TensorOptions().dtype(torch::kInt32).device(dev))
                       .view({1, -1}).contiguous();
        it = g_btcache.emplace(key, bt_t).first;
    }
    at::Tensor bt = it->second;
    at::Tensor idx = block_table.narrow(1, blk_lo, nb).select(0, b);
    int64_t S = nb * blockSize;
    // R113: index_select keeps indices ON DEVICE (advanced .index host-reads
    // them => a D2H sync per gather, 6x per layer per step); gathered VALUES
    // are identical => outputs remain bit-exact.
    at::Tensor idx64 = idx.to(at::kLong);
    at::Tensor k_r91 = at::index_select(key_cache, 0, idx64).permute({0,2,1,3}).contiguous().view(torch::kInt8);
    at::Tensor v_r91 = at::index_select(value_cache, 0, idx64).permute({0,2,1,3}).contiguous().view(torch::kInt8);
    // R152 P0-b: universal stride contract. The r146 stride .o derives
    // seqInnerSize = nb*blockSize from the K/bt tensor shapes it is given; a
    // tight [1,nkv,mid_len] scale staging has row pitch mid_len != S => OOB
    // scale reads => NaN o_h/l_h (R151 B1 root cause — fixed here at the true
    // site). Pitch is derived from the PHYSICAL gathered block span S (shape-
    // derived, batch-agnostic; NO B==1 special case). The base .o
    // (R146_STRIDE=0) requires scale dim-2 == max(actualSeqKv) (tight), so the
    // staging follows the ACTIVE op contract via env — never via batch size.
    // In-range values and math order are bit-identical to the tight staging.
    const char* r152_stride_env = std::getenv("R146_STRIDE");
    const bool r152_cap_pitch = (r152_stride_env != nullptr)
                             && (std::strcmp(r152_stride_env, "1") == 0);
    auto stage_scales = [&](const at::Tensor& scale_cache) {
        at::Tensor g = at::index_select(scale_cache, 0, idx64).reshape({S, numKvHeads})
                         .narrow(0, 0, mid_len);
        if (!r152_cap_pitch) {
            return g.t().unsqueeze(0).to(at::kBFloat16).contiguous();
        }
        at::Tensor t = torch::zeros({1, numKvHeads, S},
            torch::TensorOptions().dtype(at::kBFloat16).device(dev));
        t.narrow(2, 0, mid_len).copy_(g.t().to(at::kBFloat16));
        return t;
    };
    at::Tensor ks = stage_scales(k_scale_cache);
    at::Tensor kz = stage_scales(k_zero_cache);
    at::Tensor ko = ((-kz) * ks).contiguous();
    at::Tensor vs = stage_scales(v_scale_cache);
    at::Tensor vz = stage_scales(v_zero_cache);
    at::Tensor vo = ((-vz) * vs).contiguous();
    at::Tensor empty_mask = torch::zeros(0, torch::TensorOptions().dtype(torch::kUInt8).device(dev));
    // ---- verbatim launch path (identical to launch_lse_fast) ----
    {   // R192 PREDUMP site: pre-launch operand capture (see r192_predump)
        at::Tensor _d[10] = {q,k_r91,v_r91,bt,ks,ko,vs,vo,out,lse};
        const char* _n[10] = {"q","k","v","bt","ks","ko","vs","vo","out","lse"};
        r192_predump("oscar_middle_prep", _d, _n, nb, mid_len, blockSize, scale, numHeads, numKvHeads);
    }
    aclTensor *a_q=make_acl(q),*a_k=make_acl(k_r91),*a_v=make_acl(v_r91),*a_bt=make_acl(bt);
    aclTensor *a_ks=make_acl(ks),*a_ko=make_acl(ko),*a_vs=make_acl(vs),*a_vo=make_acl(vo);
    aclTensor *a_o=make_acl(out),*a_lse=make_acl(lse);
    int64_t B=q.size(0); std::vector<int64_t> asq(B,q.size(2));
    std::vector<int64_t> askv_v(B, mid_len);
    aclIntArray *a_asq=aclCreateIntArray(asq.data(),B),*a_askv=aclCreateIntArray(askv_v.data(),B);
    std::vector<char> lb{'B','N','S','D','\0'};
    aclrtStream s=c10_npu::getCurrentNPUStream(q.device().index()).stream(false);
    lse.fill_(torch::Scalar(-9999.0));

    // R218-P7 commit boundary: host task-queue flush AFTER the sentinel fill
    // (and any caller staging enqueued before this call), BEFORE the direct
    // ACLNN submit. Submission-order guarantee only; NOT a device sync.
    s = r218_qbound_flush(q.device().index());
    TORCH_CHECK(s != nullptr, "R218-P7: task-queue flush failed dev=", (int)q.device().index());
    r218_qbound_seen(1, "oscar_middle_prep");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus r=aclnnOscarInt2FusedAttnInt2LseGetWorkspaceSize(
        a_q,a_k,a_v,/*pse*/nullptr,/*mask*/nullptr,a_asq,a_askv,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        a_bt,nullptr,nullptr,
        a_ks,a_ko,a_vs,a_vo,
        numHeads,scale,/*pre*/0,/*next*/0,lb.data(),numKvHeads,
        /*sparse*/0,/*innerPrecise*/0,blockSize,/*antiquantMode*/1,
        /*keyAntiq*/1,/*valAntiq*/1,
        a_o,a_lse,&ws,&ex);
    TORCH_CHECK(r==0, "R218-P7: GetWorkspaceSize failed r=",(int64_t)r);
    TORCH_CHECK(ex!=nullptr, "R218-P7: GetWorkspaceSize returned null executor");
    int64_t lr=-1;
    {
        void* ws_p=nullptr;
        if(ws>0){
            if(!g_ws.defined() || (uint64_t)g_ws.numel()<ws){
                try {
                    g_ws=torch::empty({(int64_t)ws},torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
                    g_ws_hits=0;
                } catch(...) {
                    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
                    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
                    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
                    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
                    throw;
                }
            } else { g_ws_hits++; }
            ws_p=g_ws.data_ptr();
        }
        lr=(int64_t)aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);
        TORCH_CHECK(lr==0, "R218-P7: aclnn launch failed lr=",lr);
    }
    // --- R144 G1 OPERAND DUMP (capture gathered middle operands for A/B/C replay) ---
    // Gated by R144_OPDUMP=1. Saves the EXACT gathered tensors that enter the aclnn
    // call (q, k_r91, v_r91, bt, ks, ko, vs, vo, out, lse) + meta, so a controlled
    // A(2xB1)/B(1xB2 base)/C(1xB2 grouped) comparison can replay with IDENTICAL
    // operands across schedules. Operands are schedule-INDEPENDENT (inputs only).
    // One diagnostic sync/call; values unchanged (operand dump is read-only).
    if (std::getenv("R144_OPDUMP") && std::string(std::getenv("R144_OPDUMP"))=="1" && lr == 0) {
        int64_t opmax = 1;
        if (std::getenv("R144_OPDUMP_MAX")) opmax = std::atoll(std::getenv("R144_OPDUMP_MAX"));
        if (g_r144_call < opmax) {
        aclrtSynchronizeStream(s);
        std::string odir = std::getenv("R144_OPDIR") ? std::getenv("R144_OPDIR") : "/tmp";
        auto to_cpu = [](const at::Tensor& t){ return t.contiguous().to(at::kCPU).contiguous(); };
        at::Tensor ts_arr[10] = {to_cpu(q),to_cpu(k_r91),to_cpu(v_r91),to_cpu(bt),
            to_cpu(ks),to_cpu(ko),to_cpu(vs),to_cpu(vo),to_cpu(out),to_cpu(lse)};
        const char* nm_arr[10] = {"q","k","v","bt","ks","ko","vs","vo","out","lse"};
        std::string meta = "{\"call\":" + std::to_string(g_r144_call);
        meta += ",\"numHeads\":" + std::to_string(numHeads);
        meta += ",\"scale\":" + std::to_string(scale);
        meta += ",\"numKvHeads\":" + std::to_string(numKvHeads);
        meta += ",\"blockSize\":" + std::to_string(blockSize);
        meta += ",\"mid_len\":" + std::to_string(mid_len);
        meta += ",\"nb\":" + std::to_string(nb);
        meta += ",\"tensors\":[";
        std::string path = odir + "/r144_op_call" + std::to_string(g_r144_call) + ".bin";
        FILE* f = std::fopen(path.c_str(), "wb");
        uint64_t off = 0;
        for (int i=0;i<10;i++){
            at::Tensor c = ts_arr[i].contiguous();
            if (i) meta += ",";
            std::string sh = "["; auto ss = c.sizes();
            for (int64_t j=0;j<(int)ss.size();j++){ if(j) sh+=","; sh+=std::to_string(ss[j]); }
            sh += "]";
            meta += "{\"name\":\""; meta += nm_arr[i]; meta += "\",\"offset\":";
            meta += std::to_string(off); meta += ",\"nbytes\":";
            meta += std::to_string(c.nbytes()); meta += ",\"dtype\":\"";
            meta += std::string(c.dtype().name()); meta += "\",\"shape\":"; meta += sh; meta += "}";
            if (f) std::fwrite(c.data_ptr(), 1, c.nbytes(), f);
            off += c.nbytes();
        }
        meta += "]}";
        if (f) std::fclose(f);
        std::string mp = odir + "/r144_op_call" + std::to_string(g_r144_call) + ".json";
        FILE* fm = std::fopen(mp.c_str(), "w");
        if (fm) { std::fprintf(fm, "%s", meta.c_str()); std::fclose(fm); }
        std::fprintf(stderr, "[R144-OPDUMP] call=%lld mid=%lld nb=%lld -> %s\n",
            (long long)g_r144_call,(long long)mid_len,(long long)nb,path.c_str());
        std::fflush(stderr);
        }
    }
    // --- R144 COUNTER DUMP (model path = correct tiling) ---
    // counter .o writes attentionOutGm[:640] = 16 bf16/core x 40 cores:
    // [0]=dK(dequantK calls) [1]=dV [2]=mC(memberCount) [3]=maxSlot [4]=maxGroupMembers.
    // Gated by R144_COUNTER_DUMP; diagnostic sync on THIS stream only. COUNTER LEG ISOLATED.
    if (std::getenv("R144_COUNTER_DUMP") && std::string(std::getenv("R144_COUNTER_DUMP"))=="1" && lr == 0) {
        aclrtSynchronizeStream(s);
        int64_t B = q.size(0), NQ = numHeads, NKV = numKvHeads, Dm = q.size(3);
        int64_t G = (NKV > 0) ? (NQ / NKV) : 0;
        at::Tensor cnt = out.reshape(-1).narrow(0, 0, 640).to(at::kFloat).to(at::kCPU).contiguous();
        float* p = cnt.data_ptr<float>();
        double dK = 0, dV = 0, mC = 0; int64_t nz = 0; float maxSlot = 0, maxGrp = 0;
        std::string jc = "[";
        for (int c = 0; c < 40; ++c) {
            float dk=p[c*16+0], dv=p[c*16+1], mc=p[c*16+2], ms=p[c*16+3], mg=p[c*16+4];
            dK+=dk; dV+=dv; mC+=mc; if(dk>0)++nz; if(ms>maxSlot)maxSlot=ms; if(mg>maxGrp)maxGrp=mg;
            if(c) jc+=","; jc+="["+std::to_string(dk)+","+std::to_string(dv)+","+std::to_string(mc)+","+std::to_string(ms)+","+std::to_string(mg)+"]";
        }
        jc += "]";
        double kp = (mC>0)?(dK/mC):0.0, vp = (mC>0)?(dV/mC):0.0;
        std::string dir = std::getenv("R144_COUNTER_DIR") ? std::getenv("R144_COUNTER_DIR") : "/tmp";
        std::string path = dir + "/r144_counter_call" + std::to_string(g_r144_call) + ".json";
        FILE* f = std::fopen(path.c_str(), "w");
        if (f) {
            std::fprintf(f, "{\"call\":%lld,\"B\":%lld,\"NQ\":%lld,\"NKV\":%lld,\"G\":%lld,\"D\":%lld,\"actual_kv_len\":%lld,\"blockSize\":%lld,\"nb\":%lld,\"sInnerTileCount_est_mCdivNQ\":%.4f,\"groupWorkItemCount_dK\":%.0f,\"memberCount_mC\":%.0f,\"K_dequant_calls\":%.0f,\"V_dequant_calls\":%.0f,\"base_expected_calls_NQxtiles_mC\":%.0f,\"grouped_expected_calls_NKVxtiles\":%.4f,\"K_dequant_per_member\":%.6f,\"V_dequant_per_member\":%.6f,\"nonzero_core_count\":%lld,\"maxSlot\":%.0f,\"maxGroupMembers\":%.0f,\"cores\":[%s]}",
                (long long)g_r144_call,(long long)B,(long long)NQ,(long long)NKV,(long long)G,(long long)Dm,(long long)mid_len,(long long)blockSize,(long long)nb,
                (mC>0&&NQ>0)?(mC/(double)NQ):0.0, dK,mC,dK,dV, (mC>0?mC:0.0), (mC>0&&NQ>0)?(mC*(double)NKV/(double)NQ):0.0,
                kp,vp,(long long)nz,maxSlot,maxGrp,jc.c_str());
            std::fclose(f);
        }
        std::fprintf(stderr,"[R144-CNT] call=%lld B=%lld NQ=%lld NKV=%lld G=%lld D=%lld kvlen=%lld dK=%.0f dV=%.0f mC=%.0f K/mC=%.4f V/mC=%.4f nz=%lld maxSlot=%.0f maxGrp=%.0f\n",
            (long long)g_r144_call,(long long)B,(long long)NQ,(long long)NKV,(long long)G,(long long)Dm,(long long)mid_len,dK,dV,mC,kp,vp,(long long)nz,maxSlot,maxGrp);
        std::fflush(stderr);
    }
    // --- R144 CORRECTNESS DUMP (counter OFF; reads the REAL out/lse) ---
    // Gated by R144_CORRECT_DUMP. One sync/call (diagnostic; values unchanged).
    // Dumps out (bf16 bytes) + lse (fp32 bytes) + meta (actual_len/bt/scale)
    // per middle-op call for base-vs-grouped bit-exact / max_abs / NRMSE compare.
    if (std::getenv("R144_CORRECT_DUMP") && std::string(std::getenv("R144_CORRECT_DUMP"))=="1" && lr == 0) {
        aclrtSynchronizeStream(s);
        int64_t call = g_r144_call;
        std::string cdir = std::getenv("R144_CORRECT_DIR") ? std::getenv("R144_CORRECT_DIR") : "/tmp";
        // out: [B,NQ,Sq,D] bf16 -> raw bytes; lse: [B,NQ,Sq] fp32 -> raw bytes
        at::Tensor oc = out.contiguous().to(at::kCPU).contiguous();
        at::Tensor lc = lse.contiguous().to(at::kCPU).contiguous();
        std::string ob = cdir + "/c" + std::to_string(call) + "_out.bin";
        std::string lb = cdir + "/c" + std::to_string(call) + "_lse.bin";
        FILE* fo = std::fopen(ob.c_str(), "wb");
        if (fo) { std::fwrite(oc.data_ptr(), 1, oc.nbytes(), fo); std::fclose(fo); }
        FILE* fl = std::fopen(lb.c_str(), "wb");
        if (fl) { std::fwrite(lc.data_ptr(), 1, lc.nbytes(), fl); std::fclose(fl); }
        // bt summary (the gathered block indices): sha of block_table[b, blk_lo:blk_lo+nb]
        at::Tensor bt_idx = block_table.narrow(1, blk_lo, nb).select(0, b).contiguous().to(torch::kInt32).to(at::kCPU).contiguous();
        uint64_t bt_h = 1469598103934665603ULL;
        for (int64_t i = 0; i < bt_idx.numel(); ++i) { int32_t v = bt_idx.data_ptr<int32_t>()[i];
            bt_h ^= (uint64_t)v; bt_h *= 1099511628211ULL; }
        // scale summaries (min/max of ks,ko,vs,vo as bf16 floats)
        auto f2 = [](const at::Tensor& t){ auto f = t.to(at::kFloat).to(at::kCPU); float mn, mx;
            mn = f.min().item<float>(); mx = f.max().item<float>(); return std::make_pair(mn, mx); };
        auto ks_ = f2(ks), ko_ = f2(ko), vs_ = f2(vs), vo_ = f2(vo);
        std::string jp = cdir + "/c" + std::to_string(call) + "_meta.json";
        FILE* fm = std::fopen(jp.c_str(), "w");
        if (fm) {
            std::fprintf(fm, "{\"call\":%lld,\"B\":%lld,\"NQ\":%lld,\"NKV\":%lld,\"G\":%lld,\"D\":%lld,\"actual_kv_len\":%lld,\"blockSize\":%lld,\"nb\":%lld,\"blk_lo\":%lld,\"bt_fnv1a\":\"%016llx\",\"ks_min\":%.6f,\"ks_max\":%.6f,\"ko_min\":%.6f,\"ko_max\":%.6f,\"vs_min\":%.6f,\"vs_max\":%.6f,\"vo_min\":%.6f,\"vo_max\":%.6f,\"out_nbytes\":%lld,\"lse_nbytes\":%lld}",
                (long long)call, (long long)q.size(0), (long long)numHeads, (long long)numKvHeads,
                (long long)(numKvHeads>0?numHeads/numKvHeads:0), (long long)q.size(3),
                (long long)mid_len, (long long)blockSize, (long long)nb, (long long)blk_lo,
                (unsigned long long)bt_h, ks_.first, ks_.second, ko_.first, ko_.second,
                vs_.first, vs_.second, vo_.first, vo_.second, (long long)oc.nbytes(), (long long)lc.nbytes());
            std::fclose(fm);
        }
        std::fprintf(stderr, "[R144-COR] call=%lld kvlen=%lld bt=%016llx out_b=%lld lse_b=%lld\n",
            (long long)call, (long long)mid_len, (unsigned long long)bt_h, (long long)oc.nbytes(), (long long)lc.nbytes());
        std::fflush(stderr);
    }
    ++g_r144_call;
    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
    py::tuple res=py::make_tuple((int64_t)r,(int64_t)ws,(int64_t)lr,g_ws_hits);
    return res;
}
// R215 GM capture state (one-shot, env-gated)
static bool _r215_gm_post = false;
static std::string _r215_gm_dir;
static at::Tensor _r215_gm_out, _r215_gm_lse;
// R144 G1: REAL batched middle op — general runtime-B (NOT hardcoded B=2).
// Merges B per-batch B=1 middle calls into ONE aclnn launch with op input B=runtime.
// q_stack=[B,NQ,1,D], out_stack=[B,NQ,1,D], lse_stack=[B,NQ,1]; gathered K/V/scale
// stacked to [B,nkv,max_mid,D/4] / [B,nkv,max_mid]. Per-row actualSeqKv/block_table/
// scale-offset; identity bt=[B,max_nb] padded. All shapes runtime-derived (max_nb/
// max_mid swept across rows -> supports hetero length + non-contig bt). B=2 is just
// the first test config, NOT an implementation limit (supports 1/2/4/8).
// NOT a Python loop wrapping Bx B=1 aclnn calls. Caller-gated (framework R144_BATCHED_MIDDLE).
// R144 G1: REAL batched middle op — general runtime-B (NOT hardcoded B=2).
// Merges B per-batch B=1 middle calls into ONE aclnn launch with op input B=runtime.
// Per-row blk_lo (mid_start=pref varies per request) + per-row nb/mid_len/actualSeqKv.
// K contract = real paged (proven in op-level ABC): K=[total_blocks,nkv,block,D/4]
// (concatenated gathered blocks, NOT stacked-per-batch), bt=[B,max_nb] with row i ->
// [offset_i..offset_i+nb_i-1] (global indices into K, padded w/ last valid), askv=
// mid_len_list. Scales=[B,nkv,max_mid] (per-row, padded). All shapes runtime-derived.
// B=1/2/4/8 share ONE impl. NOT a Python loop wrapping Bx B=1. Framework R144_BATCHED_MIDDLE.
static py::object oscar_middle_prep_batched(
        torch::Tensor block_table,
        std::vector<int64_t> b_list, std::vector<int64_t> blk_lo_list,
        std::vector<int64_t> nb_list, std::vector<int64_t> mid_len_list,
        torch::Tensor key_cache, torch::Tensor value_cache,
        torch::Tensor k_scale_cache, torch::Tensor k_zero_cache,
        torch::Tensor v_scale_cache, torch::Tensor v_zero_cache,
        torch::Tensor q_stack, torch::Tensor out_stack, torch::Tensor lse_stack,
        int64_t numHeads, double scale, int64_t numKvHeads, int64_t blockSize) {
    const c10::DeviceGuard device_guard(q_stack.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(q_stack.device().index()));
    auto dev = q_stack.device();
    int64_t B = q_stack.size(0);
    int64_t D = q_stack.size(3);
    int64_t nkv = numKvHeads;
    int64_t max_nb = 0, max_mid = 0;
    for (int64_t i=0;i<B;i++){
        if (nb_list[i] > max_nb) max_nb = nb_list[i];
        if (mid_len_list[i] > max_mid) max_mid = mid_len_list[i];
    }
    if (max_nb < 1) max_nb = 1;  // guard: avoid 0-size bt
    // gather per-row blocks: [nb_i, nkv, block, D/4] (full blocks, NOT narrowed;
    // the op indexes blocks via bt, askv caps actual length within the blocks)
    auto gather_kv = [&](const at::Tensor& cache, const at::Tensor& idx, int64_t nb) {
        auto g = at::index_select(cache, 0, idx).permute({0,2,1,3}).contiguous().view(torch::kInt8);
        return g;  // [nb, nkv, block, D/4]
    };
    auto gather_s = [&](const at::Tensor& cache, const at::Tensor& idx, int64_t nb, int64_t mid_len) {
        int64_t cap = nb*blockSize; int64_t m = std::min(mid_len, cap);
        auto g = at::index_select(cache, 0, idx).reshape({cap, nkv}).narrow(0,0,std::max((int64_t)0,m));
        auto gp = at::zeros({nkv, max_mid}, g.options().dtype(at::kBFloat16));
        if (m>0) gp.narrow(1,0,m).copy_(g.t());
        return gp.unsqueeze(0);  // [1, nkv, max_mid]
    };
    std::vector<at::Tensor> kv, vv, ksv, kzv, vsv, vzv;
    std::vector<int64_t> row_offsets(B, 0);
    int64_t acc = 0;
    for (int64_t i=0;i<B;i++){
        auto idx = block_table.narrow(1, blk_lo_list[i], nb_list[i]).select(0, b_list[i]).to(at::kLong);
        kv.push_back(gather_kv(key_cache, idx, nb_list[i]));
        vv.push_back(gather_kv(value_cache, idx, nb_list[i]));
        ksv.push_back(gather_s(k_scale_cache, idx, nb_list[i], mid_len_list[i]));
        kzv.push_back(gather_s(k_zero_cache, idx, nb_list[i], mid_len_list[i]));
        vsv.push_back(gather_s(v_scale_cache, idx, nb_list[i], mid_len_list[i]));
        vzv.push_back(gather_s(v_zero_cache, idx, nb_list[i], mid_len_list[i]));
        row_offsets[i] = acc;
        acc += nb_list[i];
    }
    // K = concatenated gathered blocks: [total_blocks, nkv, block, D/4]
    at::Tensor k_b = (kv.size()==1) ? kv[0].contiguous() : at::cat(kv, 0).contiguous();
    at::Tensor v_b = (vv.size()==1) ? vv[0].contiguous() : at::cat(vv, 0).contiguous();
    // scales: cat (NOT stack) — each gather_s returns [1,nkv,max_mid]; cat dim0 -> [B,nkv,max_mid] (3D).
    // stack would add a spurious dim -> [B,1,nkv,max_mid] (4D) -> op 561002.
    at::Tensor ks_b = (ksv.size()==1) ? ksv[0].contiguous() : at::cat(ksv, 0).contiguous();
    at::Tensor kz_b = (kzv.size()==1) ? kzv[0].contiguous() : at::cat(kzv, 0).contiguous();
    at::Tensor vs_b = (vsv.size()==1) ? vsv[0].contiguous() : at::cat(vsv, 0).contiguous();
    at::Tensor vz_b = (vzv.size()==1) ? vzv[0].contiguous() : at::cat(vzv, 0).contiguous();
    at::Tensor ko_b = ((-kz_b)*ks_b).contiguous(), vo_b = ((-vz_b)*vs_b).contiguous();
    // bt=[B, max_nb] int32: row i -> [offset_i .. offset_i+nb_i-1], pad w/ 0 (proven
    // via op-level probe: 0-pad getws=0; last-valid-pad -> duplicate bt -> 561002.
    // askv caps actual len so the padded block 0 entries are never read for short rows)
    at::Tensor bt_b_cpu = at::zeros({B, max_nb}, at::TensorOptions().dtype(torch::kInt32));
    auto bt_a = bt_b_cpu.accessor<int32_t,2>();
    for (int64_t i=0;i<B;i++){
        int64_t o = row_offsets[i];
        for (int64_t j=0;j<max_nb;j++){
            int64_t v = (j < nb_list[i]) ? (o + j) : 0;  // pad with 0
            bt_a[i][j] = (int32_t)v;
        }
    }
    at::Tensor bt_b = bt_b_cpu.to(dev);
    aclTensor *a_q=make_acl(q_stack),*a_k=make_acl(k_b),*a_v=make_acl(v_b),*a_bt=make_acl(bt_b);
    aclTensor *a_ks=make_acl(ks_b),*a_ko=make_acl(ko_b),*a_vs=make_acl(vs_b),*a_vo=make_acl(vo_b);
    aclTensor *a_o=make_acl(out_stack),*a_lse=make_acl(lse_stack);
    std::vector<int64_t> asq(B,1);
    aclIntArray *a_asq=aclCreateIntArray(asq.data(),B),*a_askv=aclCreateIntArray(mid_len_list.data(),B);
    if (std::getenv("R144_BATCH_DBG") && std::string(std::getenv("R144_BATCH_DBG"))=="1") {
        auto ps=[&](const at::Tensor& t){ std::string s="["; for(auto x:t.sizes()){ s+=std::to_string(x); s+=","; } s+="]"; return s; };
        std::fprintf(stderr,"[R144-BD] q=%s k=%s v=%s bt=%s ks=%s ko=%s vs=%s vo=%s out=%s lse=%s askv=[",
            ps(q_stack).c_str(), ps(k_b).c_str(), ps(v_b).c_str(), ps(bt_b).c_str(), ps(ks_b).c_str(),
            ps(ko_b).c_str(), ps(vs_b).c_str(), ps(vo_b).c_str(), ps(out_stack).c_str(), ps(lse_stack).c_str());
        for(auto x:mid_len_list) std::fprintf(stderr,"%lld,",(long long)x);
        std::fprintf(stderr,"] k.dt=%d bt.dt=%d ks.dt=%d\n",(int)k_b.scalar_type(),(int)bt_b.scalar_type(),(int)ks_b.scalar_type());
        std::fflush(stderr);
    }
    // R215 GROUP-M batched in-vivo capture (env R215_GMDUMP=1, default OFF,
    // ONE-shot, B>=R215_GMDUMP_MINB): the stacked operands PRE-launch and the
    // product's own out/lse POST-launch — real mixed-step GROUP-M ground truth.
    static int r215_gm_done = 0;
    {
        const char* r215e = std::getenv("R215_GMDUMP");
        int64_t r215minb = 4;
        if (const char* mb = std::getenv("R215_GMDUMP_MINB")) r215minb = std::atoll(mb);
        if (r215e && r215e[0]=='1' && !r215_gm_done && (int64_t)q_stack.size(0) >= r215minb) {
            r215_gm_done = 1;
            std::string odir = std::getenv("R215_GMDUMP_DIR") ? std::getenv("R215_GMDUMP_DIR") : "/tmp/r215_gm";
            std::error_code _fs_ec;
        std::filesystem::create_directories(odir, _fs_ec);
            try {
                torch::Tensor _pre[10] = {q_stack,k_b,v_b,bt_b,ks_b,ko_b,vs_b,vo_b,out_stack.clone(),lse_stack.clone()};
                const char* _pn[10] = {"q","k","v","bt","ks","ko","vs","vo","out_pre","lse_pre"};
                for (int i=0;i<10;i++) torch::save(_pre[i], odir + "/r215gm_" + _pn[i] + ".pt");
                std::fprintf(stderr,"[R215-GM] PRE dumped B=%lld nb_total=%lld max_mid=%lld -> %s\n",
                    (long long)q_stack.size(0),(long long)k_b.size(0),(long long)max_mid,odir.c_str());
                std::fflush(stderr);
                _r215_gm_post = true;
                _r215_gm_dir = odir;
                _r215_gm_out = out_stack;
                _r215_gm_lse = lse_stack;
            } catch (const std::exception& ex215) {
                std::fprintf(stderr,"[R215-GM] PRE failed: %s\n", ex215.what());
            }
        }
    }
    std::vector<char> lb{'B','N','S','D','\0'};
    aclrtStream s=c10_npu::getCurrentNPUStream(q_stack.device().index()).stream(false);
    lse_stack.fill_(torch::Scalar(-9999.0));

    // R218-P7 commit boundary: host task-queue flush AFTER the sentinel fill
    // (and any caller staging enqueued before this call), BEFORE the direct
    // ACLNN submit. Submission-order guarantee only; NOT a device sync.
    s = r218_qbound_flush(q_stack.device().index());
    TORCH_CHECK(s != nullptr, "R218-P7: task-queue flush failed dev=", (int)q_stack.device().index());
    r218_qbound_seen(2, "oscar_middle_prep_batched");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus r=aclnnOscarInt2FusedAttnInt2LseGetWorkspaceSize(
        a_q,a_k,a_v,nullptr,nullptr,a_asq,a_askv,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        a_bt,nullptr,nullptr,
        a_ks,a_ko,a_vs,a_vo,
        numHeads,scale,0,0,lb.data(),numKvHeads,
        0,0,blockSize,1,1,1,
        a_o,a_lse,&ws,&ex);
    TORCH_CHECK(r==0, "R218-P7: GetWorkspaceSize failed r=",(int64_t)r);
    TORCH_CHECK(ex!=nullptr, "R218-P7: GetWorkspaceSize returned null executor");
    int64_t lr=-1;
    {
        void* ws_p=nullptr;
        if(ws>0){
            if(!g_ws.defined() || (uint64_t)g_ws.numel()<ws){
                g_ws=torch::empty({(int64_t)ws},torch::TensorOptions().dtype(torch::kUInt8).device(q_stack.device())); g_ws_hits=0;
            } else { g_ws_hits++; }
            ws_p=g_ws.data_ptr();
        }
        lr=(int64_t)aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);
        TORCH_CHECK(lr==0, "R218-P7: aclnn launch failed lr=",lr);
    }
    if (_r215_gm_post && (int64_t)r == 0) {
        aclrtSynchronizeStream(s);
        try {
            torch::save(_r215_gm_out, _r215_gm_dir + "/r215gm_out_post.pt");
            torch::save(_r215_gm_lse, _r215_gm_dir + "/r215gm_lse_post.pt");
            std::fprintf(stderr,"[R215-GM] POST dumped lse rows=%lld\n",(long long)_r215_gm_lse.size(0));
        } catch (const std::exception& ex215) { std::fprintf(stderr,"[R215-GM] POST failed: %s\n", ex215.what()); }
        _r215_gm_post = false;
    }
    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
    py::tuple res=py::make_tuple((int64_t)r,(int64_t)ws,(int64_t)lr,g_ws_hits);
    return res;
}
// R144 G1: B=1 replay from PRE-GATHERED operands (captured from the model path via
// R144_OPDUMP). Same aclnn call as oscar_middle_prep; gather skipped. Lets the
// controlled A(2xB1)/B(1xB2 base)/C(1xB2 grouped) arms reuse IDENTICAL operands
// across schedules (only the loaded .o / tiling differs). Operands are device tensors.
static py::object oscar_middle_replay(
        torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor bt,
        torch::Tensor kScale, torch::Tensor kOff, torch::Tensor vScale, torch::Tensor vOff,
        torch::Tensor out, torch::Tensor lse,
        int64_t numHeads, double scale, int64_t numKvHeads, int64_t blockSize,
        int64_t mid_len) {
    const c10::DeviceGuard device_guard(q.device());
    auto dev = q.device();
    auto mk = [&](const at::Tensor& t){ return (t.device()==dev)?t:t.to(dev); };
    at::Tensor qd=mk(q),kd=mk(k),vd=mk(v),btd=mk(bt);
    at::Tensor ksd=mk(kScale),kod=mk(kOff),vsd=mk(vScale),vod=mk(vOff);
    at::Tensor od=mk(out),ld=mk(lse);
    aclTensor *a_q=make_acl(qd),*a_k=make_acl(kd),*a_v=make_acl(vd),*a_bt=make_acl(btd);
    aclTensor *a_ks=make_acl(ksd),*a_ko=make_acl(kod),*a_vs=make_acl(vsd),*a_vo=make_acl(vod);
    aclTensor *a_o=make_acl(od),*a_lse=make_acl(ld);
    int64_t B=qd.size(0); std::vector<int64_t> asq(B,qd.size(2));
    std::vector<int64_t> askv_v(B, mid_len);
    aclIntArray *a_asq=aclCreateIntArray(asq.data(),B),*a_askv=aclCreateIntArray(askv_v.data(),B);
    std::vector<char> lb{'B','N','S','D','\0'};
    aclrtStream s=c10_npu::getCurrentNPUStream(qd.device().index()).stream(false);
    ld.fill_(torch::Scalar(-9999.0));

    // R218-P7 commit boundary: host task-queue flush AFTER the sentinel fill
    // (and any caller staging enqueued before this call), BEFORE the direct
    // ACLNN submit. Submission-order guarantee only; NOT a device sync.
    s = r218_qbound_flush(qd.device().index());
    TORCH_CHECK(s != nullptr, "R218-P7: task-queue flush failed dev=", (int)qd.device().index());
    r218_qbound_seen(3, "oscar_middle_replay");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus r=aclnnOscarInt2FusedAttnInt2LseGetWorkspaceSize(
        a_q,a_k,a_v,nullptr,nullptr,a_asq,a_askv,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        a_bt,nullptr,nullptr,
        a_ks,a_ko,a_vs,a_vo,
        numHeads,scale,0,0,lb.data(),numKvHeads,
        0,0,blockSize,1,1,1,
        a_o,a_lse,&ws,&ex);
    TORCH_CHECK(r==0, "R218-P7: GetWorkspaceSize failed r=",(int64_t)r);
    TORCH_CHECK(ex!=nullptr, "R218-P7: GetWorkspaceSize returned null executor");
    int64_t lr=-1;
    {
        void* ws_p=nullptr;
        if(ws>0){
            if(!g_ws.defined() || (uint64_t)g_ws.numel()<ws){
                g_ws=torch::empty({(int64_t)ws},torch::TensorOptions().dtype(torch::kUInt8).device(qd.device())); g_ws_hits=0;
            } else { g_ws_hits++; }
            ws_p=g_ws.data_ptr();
        }
        lr=(int64_t)aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);
        TORCH_CHECK(lr==0, "R218-P7: aclnn launch failed lr=",lr);
    }
    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
    py::tuple res=py::make_tuple((int64_t)r,(int64_t)ws,(int64_t)lr,g_ws_hits);
    return res;
}
// R144 G1: B-agnostic batched replay from PRE-GATHERED stacked operands. The single
// aclnn launch with op input B=runtime (1/2/4/8). mid_len_list per row (hetero length).
// bt_stack=[B,max_nb] identity; scale/offset=[B,nkv,max_mid]. No gather, no Python loop.
static py::object oscar_middle_replay_batched(
        torch::Tensor q_stack, torch::Tensor k_stack, torch::Tensor v_stack,
        torch::Tensor bt_stack,
        torch::Tensor kScale, torch::Tensor kOff, torch::Tensor vScale, torch::Tensor vOff,
        torch::Tensor out_stack, torch::Tensor lse_stack,
        int64_t numHeads, double scale, int64_t numKvHeads, int64_t blockSize,
        std::vector<int64_t> mid_len_list) {
    const c10::DeviceGuard device_guard(q_stack.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(q_stack.device().index()));
    auto dev = q_stack.device();
    int64_t B = q_stack.size(0);
    auto mk = [&](const at::Tensor& t){ return (t.device()==dev)?t:t.to(dev); };
    at::Tensor qd=mk(q_stack),kd=mk(k_stack),vd=mk(v_stack),btd=mk(bt_stack);
    at::Tensor ksd=mk(kScale),kod=mk(kOff),vsd=mk(vScale),vod=mk(vOff);
    at::Tensor od=mk(out_stack),ld=mk(lse_stack);
    aclTensor *a_q=make_acl(qd),*a_k=make_acl(kd),*a_v=make_acl(vd),*a_bt=make_acl(btd);
    aclTensor *a_ks=make_acl(ksd),*a_ko=make_acl(kod),*a_vs=make_acl(vsd),*a_vo=make_acl(vod);
    aclTensor *a_o=make_acl(od),*a_lse=make_acl(ld);
    std::vector<int64_t> asq(B,1);
    aclIntArray *a_asq=aclCreateIntArray(asq.data(),B),*a_askv=aclCreateIntArray(mid_len_list.data(),B);
    std::vector<char> lb{'B','N','S','D','\0'};
    aclrtStream s=c10_npu::getCurrentNPUStream(qd.device().index()).stream(false);
    ld.fill_(torch::Scalar(-9999.0));

    // R218-P7 commit boundary: host task-queue flush AFTER the sentinel fill
    // (and any caller staging enqueued before this call), BEFORE the direct
    // ACLNN submit. Submission-order guarantee only; NOT a device sync.
    s = r218_qbound_flush(qd.device().index());
    TORCH_CHECK(s != nullptr, "R218-P7: task-queue flush failed dev=", (int)qd.device().index());
    r218_qbound_seen(4, "oscar_middle_replay_batched");
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus r=aclnnOscarInt2FusedAttnInt2LseGetWorkspaceSize(
        a_q,a_k,a_v,nullptr,nullptr,a_asq,a_askv,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        a_bt,nullptr,nullptr,
        a_ks,a_ko,a_vs,a_vo,
        numHeads,scale,0,0,lb.data(),numKvHeads,
        0,0,blockSize,1,1,1,
        a_o,a_lse,&ws,&ex);
    TORCH_CHECK(r==0, "R218-P7: GetWorkspaceSize failed r=",(int64_t)r);
    TORCH_CHECK(ex!=nullptr, "R218-P7: GetWorkspaceSize returned null executor");
    int64_t lr=-1;
    {
        void* ws_p=nullptr;
        if(ws>0){
            if(!g_ws.defined() || (uint64_t)g_ws.numel()<ws){
                g_ws=torch::empty({(int64_t)ws},torch::TensorOptions().dtype(torch::kUInt8).device(qd.device())); g_ws_hits=0;
            } else { g_ws_hits++; }
            ws_p=g_ws.data_ptr();
        }
        lr=(int64_t)aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);
        TORCH_CHECK(lr==0, "R218-P7: aclnn launch failed lr=",lr);
    }
    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
    py::tuple res=py::make_tuple((int64_t)r,(int64_t)ws,(int64_t)lr,g_ws_hits);
    return res;
}
// R144 G1: read the counter table from a (counter .o) output's first 640 bf16.
// Returns tuple (dK,dV,mC,nz,maxSlot,maxGrp,K_per_member,V_per_member,G,base_exp,grp_exp).
// Counter .o (7e354b85, FORCE_GROUPED=1+COUNTER_MODE=1) writes per-core dequant/member
// counts to attentionOutGm[:640] = 16 bf16/core x 40 cores. For B=2 grouped, dK=B*NKV*tiles,
// mC=B*NQ*tiles => K/mC stays 1/G=0.25 (proof grouped active at B>1). Counter corrupts output.
static py::object r144_read_counter(torch::Tensor out_gm, int64_t B, int64_t NQ, int64_t NKV) {
    const c10::DeviceGuard device_guard(out_gm.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(out_gm.device().index()));
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(out_gm.device().index()).stream(false));
    at::Tensor cnt = out_gm.reshape(-1).narrow(0,0,640).to(at::kFloat).to(at::kCPU).contiguous();
    float* p = cnt.data_ptr<float>();
    double dK=0,dV=0,mC=0; int64_t nz=0; float maxSlot=0,maxGrp=0;
    for (int c=0;c<40;c++){
        float dk=p[c*16+0],dv=p[c*16+1],mc=p[c*16+2],ms=p[c*16+3],mg=p[c*16+4];
        dK+=dk;dV+=dv;mC+=mc; if(dk>0)++nz; if(ms>maxSlot)maxSlot=ms; if(mg>maxGrp)maxGrp=mg;
    }
    double kp=(mC>0)?(dK/mC):0.0, vp=(mC>0)?(dV/mC):0.0;
    int64_t G = (NKV>0)?(NQ/NKV):0;
    double base_exp = (double)B*NQ, grp_exp = (double)B*NKV;
    py::tuple res = py::make_tuple((double)dK,(double)dV,(double)mC,(int64_t)nz,
        (double)maxSlot,(double)maxGrp,kp,vp,(int64_t)G,(double)base_exp,(double)grp_exp);
    return res;
}
// R113 Fallback-1b: C++ port of oscar_quantize_int2 (both K and V in one
// crossing). Replicates the Python op sequence with ATen (abs/sort/threshold-
// clip via maximum+minimum — clamp(Tensor,Tensor) unavailable; values are
// EXACTLY equal: clamp(x,lo,hi) == min(max(x,lo),hi) for non-NaN). Pack uses
// +4/+16/+64 instead of |<<2/<<4/<<6: 2-bit lanes are disjoint, so add==or
// bit-for-bit. ~58 Python dispatches per layer collapse to one call.
static std::vector<at::Tensor> r113_quant_one(const at::Tensor& x, double clip_ratio){
    auto T = x.size(0), H = x.size(1), D = x.size(2);
    int64_t clip_idx = (int64_t)(clip_ratio * (double)D);
    clip_idx = clip_idx < 1 ? 1 : (clip_idx > (int64_t)D ? (int64_t)D : clip_idx);
    auto abs_x = at::abs(x);
    // R113: kthvalue instead of full sort — same threshold VALUE (k-th smallest
    // == ascending-sorted index k-1) => bit-exact, but avoids the AICPU sort
    // kernel whose completion is not reliably stream-ordered under back-to-back
    // C++ dispatch (deterministic stale-read garble; device-wide sync masks).
    auto threshold = std::get<0>(at::kthvalue(abs_x, /*k=*/clip_idx, /*dim=*/-1, /*keepdim=*/true));
    auto neg_thr = at::neg(threshold);
    auto x_clipped = at::minimum(at::maximum(x, neg_thr), threshold);
    auto x_min = at::amin(x_clipped, /*dims=*/{-1}, /*keepdim=*/true);
    auto x_max = at::amax(x_clipped, /*dims=*/{-1}, /*keepdim=*/true);
    auto scale = at::clamp((x_max - x_min), /*min=*/1e-8) / 3.0;
    auto zero = at::div(at::neg(x_min), scale);
    auto q = at::clamp(at::round((x_clipped / scale) + zero), /*min=*/0, /*max=*/3).to(at::kByte);
    auto qv = q.view({T, H, 4, D / 4});
    auto packed = qv.select(2,0) + 4*qv.select(2,1) + 16*qv.select(2,2) + 64*qv.select(2,3);
    return {packed, scale.squeeze(-1), zero.squeeze(-1)};
}
// R113 Fallback-1c: single-call paged scatter (6 index_put writes -> ONE
// crossing; msprof: aclnnIndexPutImpl + GetWS = ~19ms/step host at 216/step).
// Same at::index_put_ kernels, values identical.
static py::object oscar_scatter_int2(
        torch::Tensor k_cache, torch::Tensor v_cache,
        torch::Tensor ks_cache, torch::Tensor kz_cache,
        torch::Tensor vs_cache, torch::Tensor vz_cache,
        torch::Tensor slots, int64_t num_tokens,
        torch::Tensor k_packed, torch::Tensor v_packed,
        torch::Tensor k_scale, torch::Tensor k_zero,
        torch::Tensor v_scale, torch::Tensor v_zero){
    const c10::DeviceGuard device_guard(k_cache.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(k_cache.device().index()));
    int64_t num_blocks = k_cache.size(0);
    int64_t block_size = k_cache.size(1);
    auto k_flat = k_cache.view({num_blocks * block_size, k_cache.size(2), k_cache.size(3)});
    auto v_flat = v_cache.view({num_blocks * block_size, v_cache.size(2), v_cache.size(3)});
    auto ks_flat = ks_cache.view({num_blocks * block_size, -1});
    auto kz_flat = kz_cache.view({num_blocks * block_size, -1});
    auto vs_flat = vs_cache.view({num_blocks * block_size, -1});
    auto vz_flat = vz_cache.view({num_blocks * block_size, -1});
    auto sl = slots.narrow(0, 0, num_tokens);
    k_flat.index_put_({sl}, k_packed.narrow(0, 0, num_tokens));
    v_flat.index_put_({sl}, v_packed.narrow(0, 0, num_tokens));
    ks_flat.index_put_({sl}, k_scale.narrow(0, 0, num_tokens));
    kz_flat.index_put_({sl}, k_zero.narrow(0, 0, num_tokens));
    vs_flat.index_put_({sl}, v_scale.narrow(0, 0, num_tokens));
    vz_flat.index_put_({sl}, v_zero.narrow(0, 0, num_tokens));
    // device-level barrier (same class the quant chain requires; keeps the
    // consumer reads ordered; one per layer instead of six)
    aclrtSynchronizeDevice();
    return py::make_tuple((int64_t)0);
}
// R113: keep the last quant outputs ALIVE (per shape) — if the garble is the
// caching allocator reusing freed C++-local storage for the next consumer
// while writes are in flight, holding the references removes the free.
static std::map<int64_t, std::vector<at::Tensor>> g_quant_keepalive;
static py::object oscar_quant_kv(torch::Tensor k, torch::Tensor v, double kclip, double vclip){
    // R113 FIX: python-side ops carry a DeviceGuard per dispatch; a raw pybind
    // entry does NOT — without it, multi-stream internal kernels (sort et al.)
    // can land on the wrong device context (deterministic garble; proved by
    // probe-mask A/B: extra python recompute -> correct tokens).
    const c10::DeviceGuard device_guard(k.device());
    // R113 FIX (stream anchor): python-side ops pick the current stream through
    // the dispatcher's TLS; a raw pybind entry could resolve a DIFFERENT stream
    // without this guard — consumers then read stale quant outputs (deterministic
    // garble; a post-call torch.npu.synchronize() masks it, proving the class).
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(k.device().index()));
    auto a = r113_quant_one(k, kclip);
    auto b = r113_quant_one(v, vclip);
    int64_t sz = k.size(0);
    g_quant_keepalive[sz] = {a[0], a[1], a[2], b[0], b[1], b[2]};
    // R113: stream-level barrier — without it, back-to-back C++ dispatch lets
    // the consumer (paged-cache scatter) read the pack output before its write
    // is visible (deterministic garble; device-wide torch.npu.synchronize()
    // masks it; probe-recompute masks it; values verified bit-exact at ALL T).
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(k.device().index()).stream(false));
    py::tuple res = py::make_tuple(a[0], a[1], a[2], b[0], b[1], b[2]);
    return res;
}
// correctness-leg readback (explicit sync): returns out/lse on CPU + changed/finite counts
static py::dict verify_lse(torch::Tensor out, torch::Tensor lse){
    c10_npu::npuSynchronizeDevice(true);
    torch::Tensor oc=out.to(torch::kCPU).contiguous(), lc=lse.to(torch::kCPU).contiguous();
    py::dict d;
    d["out_changed"]=(int64_t)torch::ne(oc, torch::full_like(oc,-11333.0)).sum().item<int64_t>();
    d["lse_changed"]=(int64_t)torch::ne(lc, torch::full_like(lc,-9999.0)).sum().item<int64_t>();
    d["lse_finite"]=(int64_t)torch::isfinite(lc).sum().item<int64_t>();
    d["out_cpu"]=oc; d["lse_cpu"]=lc;
    return d;
}
// R144 P1 sha1 bit-exact debug: SAME-PROCESS shadow. From the SAME source caches
// + metadata, build V1 operands (original index_select+permute+contiguous+cat
// path) AND compare byte-by-byte against the V2 persistent carriers. Finds
// FIRST_WRONG_OPERAND (physical contiguous-byte hash, not just logical). Gated
// by R144_V2_SHADOW=1; runs ONCE (static counter). Diagnostic leg: allows D2H
// sync for the comparison. V1 operands are DISCARDED (not used for generation).
static int64_t g_shadow_calls = 0;
static void shadow_v1v2(
        torch::Tensor block_table,
        std::vector<int64_t> b_list, std::vector<int64_t> blk_lo_list,
        std::vector<int64_t> nb_list, std::vector<int64_t> mid_len_list,
        torch::Tensor key_cache, torch::Tensor value_cache,
        torch::Tensor k_scale_cache, torch::Tensor k_zero_cache,
        torch::Tensor v_scale_cache, torch::Tensor v_zero_cache,
        torch::Tensor q_stack,
        torch::Tensor K_contig, torch::Tensor V_contig,
        torch::Tensor bt_v2, torch::Tensor ks_v2, torch::Tensor ko_v2,
        torch::Tensor vs_v2, torch::Tensor vo_v2,
        torch::Tensor out_v2, torch::Tensor lse_v2,
        int64_t nkv, int64_t blockSize, int64_t step, int64_t numHeads, double scale) {
    int64_t B = (int64_t)b_list.size();
    int64_t max_nb=0, max_mid=0, acc=0;
    for(int64_t i=0;i<B;i++){ if(nb_list[i]>max_nb)max_nb=nb_list[i]; if(mid_len_list[i]>max_mid)max_mid=mid_len_list[i]; }
    if(max_nb<1)max_nb=1;
    auto dev = key_cache.device();
    std::vector<at::Tensor> kv, vv, ksv, kzv, vsv, vzv;
    std::vector<int64_t> row_offsets(B,0);
    for(int64_t i=0;i<B;i++){
        at::Tensor idx = block_table.narrow(1,blk_lo_list[i],nb_list[i]).select(0,b_list[i]).to(at::kLong);
        kv.push_back(at::index_select(key_cache,0,idx).permute({0,2,1,3}).contiguous().view(torch::kInt8));
        vv.push_back(at::index_select(value_cache,0,idx).permute({0,2,1,3}).contiguous().view(torch::kInt8));
        int64_t cap=nb_list[i]*blockSize; int64_t m=std::min(mid_len_list[i],cap);
        auto gs=[&](at::Tensor cache)->at::Tensor{
            at::Tensor g=at::index_select(cache,0,idx).reshape({cap,nkv}).narrow(0,0,std::max((int64_t)0,m));
            at::Tensor gp=at::zeros({nkv,max_mid},g.options().dtype(at::kBFloat16));
            if(m>0) gp.narrow(1,0,m).copy_(g.t());
            return gp.unsqueeze(0);
        };
        ksv.push_back(gs(k_scale_cache)); kzv.push_back(gs(k_zero_cache));
        vsv.push_back(gs(v_scale_cache)); vzv.push_back(gs(v_zero_cache));
        row_offsets[i]=acc; acc+=nb_list[i];
    }
    int64_t total_nb=acc;
    at::Tensor k1=(kv.size()==1)?kv[0].contiguous():at::cat(kv,0).contiguous();
    at::Tensor v1=(vv.size()==1)?vv[0].contiguous():at::cat(vv,0).contiguous();
    at::Tensor ks1=(ksv.size()==1)?ksv[0].contiguous():at::cat(ksv,0).contiguous();
    at::Tensor kz1=(kzv.size()==1)?kzv[0].contiguous():at::cat(kzv,0).contiguous();
    at::Tensor vs1=(vsv.size()==1)?vsv[0].contiguous():at::cat(vsv,0).contiguous();
    at::Tensor vz1=(vzv.size()==1)?vzv[0].contiguous():at::cat(vzv,0).contiguous();
    at::Tensor ko1=((-kz1)*ks1).contiguous(), vo1=((-vz1)*vs1).contiguous();
    at::Tensor bt1_cpu=at::zeros({B,max_nb},at::TensorOptions().dtype(torch::kInt32));
    auto ba=bt1_cpu.accessor<int32_t,2>();
    for(int64_t i=0;i<B;i++){ int64_t o=row_offsets[i]; for(int64_t j=0;j<max_nb;j++) ba[i][j]=(j<nb_list[i])?(int32_t)(o+j):0; }
    at::Tensor bt1=bt1_cpu.to(dev);
    FILE* f=std::fopen("/tmp/r144_v2_shadow.log","a");
    if(!f) return;
    std::fprintf(f,"[SHADOW step=%lld] B=%lld total_nb=%lld max_nb=%lld max_mid=%lld\n",(long long)step,(long long)B,(long long)total_nb,(long long)max_nb,(long long)max_mid);
    auto pstr=[&](const at::Tensor& t){ std::string s="["; for(auto x:t.sizes()){s+=std::to_string(x);s+=",";} s+="]st["; for(auto x:t.strides()){s+=std::to_string(x);s+=",";} s+="]dt="; s+=std::to_string((int)t.scalar_type()); s+=" off="; s+=std::to_string(t.storage_offset()); s+=" cont="; s+=std::to_string((int)t.is_contiguous()); return s; };
    // compare: shape/dtype/stride/storage_offset + physical contiguous-byte hash + first diff + max_abs
    auto cmp=[&](const char* name, at::Tensor v1t, at::Tensor v2t){
        at::Tensor c1=v1t.contiguous().cpu(), c2=v2t.contiguous().cpu();
        bool shp=(c1.sizes()==c2.sizes())&&(c1.scalar_type()==c2.scalar_type())&&(c1.numel()==c2.numel());
        int64_t bytes=c1.numel()*c1.element_size();
        const char* p1=(const char*)c1.data_ptr(); const char* p2=(const char*)c2.data_ptr();
        int64_t diff_byte=-1;
        for(int64_t k=0;k<bytes;k++){ if(p1[k]!=p2[k]){diff_byte=k;break;} }
        // sha-like hash (sum of bytes) + max_abs for float types
        uint64_t h1=1469598103934665603ULL, h2=h1;
        for(int64_t k=0;k<bytes;k++){ h1^=(uint64_t)(unsigned char)p1[k]; h1*=1099511628211ULL; h2^=(uint64_t)(unsigned char)p2[k]; h2*=1099511628211ULL; }
        std::fprintf(f,"  %-8s shape_match=%d v1=%s\n           v2=%s\n           bytes=%lld h1=%016llx h2=%016llx hash_match=%d FIRST_DIFF_BYTE=%lld\n",
            name,(int)shp,pstr(v1t).c_str(),pstr(v2t).c_str(),(long long)bytes,(unsigned long long)h1,(unsigned long long)h2,(int)(h1==h2),(long long)diff_byte);
        std::fflush(f);
    };
    cmp("K", k1, K_contig.narrow(0,0,total_nb).view(torch::kInt8));
    cmp("V", v1, V_contig.narrow(0,0,total_nb).view(torch::kInt8));
    cmp("bt", bt1, bt_v2.narrow(1,0,max_nb));
    cmp("kScale", ks1, ks_v2.narrow(2,0,max_mid));
    cmp("kOffset", ko1, ko_v2.narrow(2,0,max_mid));
    cmp("vScale", vs1, vs_v2.narrow(2,0,max_mid));
    cmp("vOffset", vo1, vo_v2.narrow(2,0,max_mid));
    // ---- OUTPUT comparison: test address-sensitivity. The op ran into out_v2
    // (carrier addr). Now: (a) V1 op into out1 (FRESH addr, zeros); (b) V1 op
    // into out_v2 (CARRIER addr, zeros). Compare both to the saved V2 result.
    // If (b)==saved but (a)!=saved -> op is OUTPUT-ADDRESS-SENSITIVE (carrier
    // stable addr vs fresh addr -> different PFA reduction). If both !=saved ->
    // op is non-deterministic (contradicts v2 run1==run2).
    aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(q_stack.device().index()).stream(false)); // V2 op is async -> sync BEFORE cloning its result
    at::Tensor out_v2_saved = out_v2.clone();      // V2 result (carrier addr)
    at::Tensor lse_v2_saved = lse_v2.clone();
    auto v1_op_call=[&](at::Tensor kk, at::Tensor vv, at::Tensor btt, at::Tensor kks, at::Tensor kko, at::Tensor vvs, at::Tensor vvo, at::Tensor o, at::Tensor l, const char* tag){
        aclTensor *a_q1=make_acl(q_stack),*a_k1=make_acl(kk),*a_v1=make_acl(vv),*a_bt1=make_acl(btt);
        aclTensor *a_ks1=make_acl(kks),*a_ko1=make_acl(kko),*a_vs1=make_acl(vvs),*a_vo1=make_acl(vvo);
        aclTensor *a_o1=make_acl(o),*a_lse1=make_acl(l);
        std::vector<int64_t> asq1(B,1);
        aclIntArray *a_asq1=aclCreateIntArray(asq1.data(),B),*a_askv1=aclCreateIntArray(mid_len_list.data(),B);
        std::vector<char> lb1{'B','N','S','D','\0'};
        aclrtStream s1=c10_npu::getCurrentNPUStream(q_stack.device().index()).stream(false);
        l.fill_(torch::Scalar(-9999.0)); o.zero_();
        uint64_t ws1=0; aclOpExecutor* ex1=nullptr;
        aclnnStatus r1=aclnnOscarInt2FusedAttnInt2LseGetWorkspaceSize(
            a_q1,a_k1,a_v1,nullptr,nullptr,a_asq1,a_askv1,
            nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
            a_bt1,nullptr,nullptr, a_ks1,a_ko1,a_vs1,a_vo1,
            numHeads,scale,0,0,lb1.data(),nkv, 0,0,blockSize,1,1,1, a_o1,a_lse1,&ws1,&ex1);
        int64_t lr1=-1;
        if(r1==0 && ex1!=nullptr){ void* wsp=nullptr; if(ws1>0){ at::Tensor fws=torch::empty({(int64_t)ws1},torch::TensorOptions().dtype(torch::kUInt8).device(q_stack.device())); wsp=fws.data_ptr(); /*FRESH ws, not g_ws*/ } lr1=(int64_t)aclnnOscarInt2FusedAttnInt2Lse(wsp,ws1,ex1,s1); aclrtSynchronizeStream(s1); }
        std::fprintf(f,"  [V1-OP %s] r1=%lld lr1=%lld\n",tag,(long long)r1,(long long)lr1);
        aclDestroyTensor(a_q1);aclDestroyTensor(a_k1);aclDestroyTensor(a_v1);aclDestroyTensor(a_bt1);
        aclDestroyTensor(a_ks1);aclDestroyTensor(a_ko1);aclDestroyTensor(a_vs1);aclDestroyTensor(a_vo1);
        aclDestroyTensor(a_o1);aclDestroyTensor(a_lse1); aclDestroyIntArray(a_asq1);aclDestroyIntArray(a_askv1);
    };
    // (a) V1 op into FRESH out1, V1 operands (per-call shapes [max_mid],[max_nb])
    at::Tensor out1 = at::zeros_like(out_v2);
    at::Tensor lse1 = at::zeros_like(lse_v2);
    v1_op_call(k1, v1, bt1, ks1, ko1, vs1, vo1, out1, lse1, "V1shapes_fresh");
    cmp("out_V1shape_vs_V2", out1, out_v2_saved);
    // (b) V1 op with V2 scale/bt SHAPES (ks_b [4096], bt_b [32]) + V1 K/V.
    // If result == V2 saved -> the SCALE/BT SHAPE (not non-det) causes the diff.
    at::Tensor out2 = at::zeros_like(out_v2);
    at::Tensor lse2 = at::zeros_like(lse_v2);
    v1_op_call(k1, v1, bt_v2, ks_v2, ko_v2, vs_v2, vo_v2, out2, lse2, "V2scalebt");
    cmp("out_V2scalebt_vs_V2", out2, out_v2_saved);
    // (c) V1 op into CARRIER addr (out_v2), V1 operands
    v1_op_call(k1, v1, bt1, ks1, ko1, vs1, vo1, out_v2, lse_v2, "V1shapes_carrier");
    cmp("out_V1shape_carrier_vs_V2", out_v2, out_v2_saved);
    std::fclose(f);
}
// R144 P1 StageV2: batched middle prep+launch into PERSISTENT device carriers
// (Python OscarBatchedCarrierAdapter-owned, passed as out params). Eliminates:
// - per-row index_select+contiguous alloc -> at::index_select_out into carrier (alloc=0)
// - at::cat -> carrier is contiguous, gathered at the right offset (cat=0)
// - at::zeros scales -> persistent bf16 scale carriers (zeros=0)
// - CPU bt + .to(dev) H2D -> persistent DEVICE int32 bt_carrier (CPU-bt=0, H2D=0)
// - ko/vo neg*mul alloc -> in-place into carriers
// NO C++ static/global tensor cache; carriers are caller-owned. Gated by the
// Python adapter (R144_BATCHED_STAGE_V2=1). Graph-capturable: addresses stable.
static py::object oscar_middle_prep_batched_v2(
        torch::Tensor block_table,
        std::vector<int64_t> b_list, std::vector<int64_t> blk_lo_list,
        std::vector<int64_t> nb_list, std::vector<int64_t> mid_len_list,
        torch::Tensor key_cache, torch::Tensor value_cache,
        torch::Tensor k_scale_cache, torch::Tensor k_zero_cache,
        torch::Tensor v_scale_cache, torch::Tensor v_zero_cache,
        torch::Tensor q_stack, torch::Tensor out_stack, torch::Tensor lse_stack,
        torch::Tensor K_carrier, torch::Tensor V_carrier,
        torch::Tensor K_contig, torch::Tensor V_contig,
        torch::Tensor kS_carrier, torch::Tensor kO_carrier,
        torch::Tensor vS_carrier, torch::Tensor vO_carrier,
        torch::Tensor bt_carrier,
        int64_t numHeads, double scale, int64_t numKvHeads, int64_t blockSize) {
    const c10::DeviceGuard device_guard(q_stack.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(q_stack.device().index()));
    auto dev = q_stack.device();
    int64_t B = q_stack.size(0);
    int64_t D = q_stack.size(3);
    int64_t nkv = numKvHeads;
    int64_t max_nb = 0, max_mid = 0;
    for (int64_t i=0;i<B;i++){
        if (nb_list[i] > max_nb) max_nb = nb_list[i];
        if (mid_len_list[i] > max_mid) max_mid = mid_len_list[i];
    }
    if (max_nb < 1) max_nb = 1;
    // bt/scales: pass CONTIGUOUS dim-0 slices (narrow(0,0,B) of a contiguous
    // carrier is contiguous). The op kernel rejects strided (dim-1/2 narrow)
    // views at launch (561000). askv caps the actual length so the larger
    // carrier dim (max_nb_cap / max_mid_cap) is fine as long as the stale tail
    // is zeroed. Zero the full active region first.
    // bt: PER-CALL device [B, max_nb] (op is SHAPE-SENSITIVE: the carrier's fixed
    // max_nb_cap -> different tiling -> different output; must match v1's per-call
    // max_nb). Device alloc + device fill (no CPU accessor, no H2D).
    at::Tensor bt_b = at::zeros({B, max_nb}, at::TensorOptions().dtype(torch::kInt32).device(dev));
    int64_t acc = 0;
    for (int64_t i=0;i<B;i++){
        int64_t nb_i = nb_list[i];
        if (nb_i > 0) {
            at::Tensor rng = at::arange(acc, acc + nb_i,
                at::TensorOptions().dtype(torch::kInt32).device(dev));
            bt_b.select(0, i).narrow(0, 0, nb_i).copy_(rng);
        }
        acc += nb_i;
    }
    int64_t total_nb = acc;
    // K/V gather into persistent carriers via index_select_out (alloc=0).
    // carrier is [Bcap*max_nb_cap, nkv, block, D/4] int8 (native cache layout);
    // the op reads a PERMUTED VIEW (permute{0,2,1,3}) -> no contiguous copy.
    int64_t off = 0;
    for (int64_t i=0;i<B;i++){
        int64_t nb_i = nb_list[i];
        at::Tensor idx = block_table.narrow(1, blk_lo_list[i], nb_i).select(0, b_list[i]).to(at::kLong);
        if (nb_i > 0) {
            at::Tensor kc = K_carrier.narrow(0, off, nb_i);
            at::Tensor vc = V_carrier.narrow(0, off, nb_i);
            at::index_select_out(kc, key_cache, 0, idx);
            at::index_select_out(vc, value_cache, 0, idx);
        }
        off += nb_i;
    }
    // K/V views for the op: the op requires CONTIGUOUS K in the transposed layout
    // [total_nb, block, nkv, D/4] (probe: non-contiguous permuted view -> 561002).
    // Transpose-copy from the native-layout carrier into a persistent contiguous
    // carrier (both persistent -> NO alloc; copy_ is a device op, graph-capturable).
    K_contig.narrow(0,0,total_nb).copy_(K_carrier.narrow(0,0,total_nb).permute({0,2,1,3}));
    V_contig.narrow(0,0,total_nb).copy_(V_carrier.narrow(0,0,total_nb).permute({0,2,1,3}));
    at::Tensor k_b = K_contig.narrow(0,0,total_nb).view(torch::kInt8);
    at::Tensor v_b = V_contig.narrow(0,0,total_nb).view(torch::kInt8);
    // scales: PER-CALL (op is SHAPE-SENSITIVE: the carrier's fixed max_mid_cap
    // -> different tiling -> different output; must match v1's per-call max_mid).
    // Replicate v1's gather_s (index_select+reshape+narrow+zeros+copy_t+unsqueeze)
    // + cat + neg*mul. Per-call alloc (small, ~nkv*max_mid bf16); K/V stay carriers.
    auto gs = [&](at::Tensor cache, at::Tensor idx, int64_t nb_i, int64_t mid_len_i)->at::Tensor{
        int64_t cap=nb_i*blockSize; int64_t m=std::min(mid_len_i, cap);
        at::Tensor g=at::index_select(cache,0,idx).reshape({cap,nkv}).narrow(0,0,std::max((int64_t)0,m));
        at::Tensor gp=at::zeros({nkv,max_mid},g.options().dtype(at::kBFloat16));
        if(m>0) gp.narrow(1,0,m).copy_(g.t());
        return gp.unsqueeze(0);  // [1, nkv, max_mid]
    };
    std::vector<at::Tensor> ksv,kzv,vsv,vzv;
    for(int64_t i=0;i<B;i++){
        at::Tensor idx=block_table.narrow(1,blk_lo_list[i],nb_list[i]).select(0,b_list[i]).to(at::kLong);
        ksv.push_back(gs(k_scale_cache,idx,nb_list[i],mid_len_list[i]));
        kzv.push_back(gs(k_zero_cache,idx,nb_list[i],mid_len_list[i]));
        vsv.push_back(gs(v_scale_cache,idx,nb_list[i],mid_len_list[i]));
        vzv.push_back(gs(v_zero_cache,idx,nb_list[i],mid_len_list[i]));
    }
    at::Tensor ks_b=(ksv.size()==1)?ksv[0].contiguous():at::cat(ksv,0).contiguous();
    at::Tensor kz_b=(kzv.size()==1)?kzv[0].contiguous():at::cat(kzv,0).contiguous();
    at::Tensor vs_b=(vsv.size()==1)?vsv[0].contiguous():at::cat(vsv,0).contiguous();
    at::Tensor vz_b=(vzv.size()==1)?vzv[0].contiguous():at::cat(vzv,0).contiguous();
    at::Tensor ko_b=((-kz_b)*ks_b).contiguous(), vo_b=((-vz_b)*vs_b).contiguous();
    if (std::getenv("R144_BATCH_DBG") && std::string(std::getenv("R144_BATCH_DBG"))=="1") {
        auto ps=[&](const at::Tensor& t){ std::string s="["; for(auto x:t.sizes()){ s+=std::to_string(x); s+=","; } s+="]"; return s; };
        std::fprintf(stderr,"[R144-V2BD] q=%s k=%s v=%s bt=%s ks=%s ko=%s vs=%s vo=%s out=%s lse=%s askv=[",
            ps(q_stack).c_str(), ps(k_b).c_str(), ps(v_b).c_str(), ps(bt_b).c_str(), ps(ks_b).c_str(),
            ps(ko_b).c_str(), ps(vs_b).c_str(), ps(vo_b).c_str(), ps(out_stack).c_str(), ps(lse_stack).c_str());
        for(auto x:mid_len_list) std::fprintf(stderr,"%lld,",(long long)x);
        std::fprintf(stderr,"] k.dt=%d bt.dt=%d ks.dt=%d k_c=%d k_c_cont=%d\n",(int)k_b.scalar_type(),(int)bt_b.scalar_type(),(int)ks_b.scalar_type(),(int)k_b.is_contiguous(),(int)K_contig.is_contiguous());
        std::fflush(stderr);
    }
    aclTensor *a_q=make_acl(q_stack),*a_k=make_acl(k_b),*a_v=make_acl(v_b),*a_bt=make_acl(bt_b);
    aclTensor *a_ks=make_acl(ks_b),*a_ko=make_acl(ko_b),*a_vs=make_acl(vs_b),*a_vo=make_acl(vo_b);
    aclTensor *a_o=make_acl(out_stack),*a_lse=make_acl(lse_stack);
    std::vector<int64_t> asq(B,1);
    aclIntArray *a_asq=aclCreateIntArray(asq.data(),B),*a_askv=aclCreateIntArray(mid_len_list.data(),B);
    std::vector<char> lb{'B','N','S','D','\0'};
    aclrtStream s=c10_npu::getCurrentNPUStream(q_stack.device().index()).stream(false);
    lse_stack.fill_(torch::Scalar(-9999.0));

    // R218-P7 commit boundary: host task-queue flush AFTER the sentinel fill
    // (and any caller staging enqueued before this call), BEFORE the direct
    // ACLNN submit. Submission-order guarantee only; NOT a device sync.
    s = r218_qbound_flush(q_stack.device().index());
    TORCH_CHECK(s != nullptr, "R218-P7: task-queue flush failed dev=", (int)q_stack.device().index());
    r218_qbound_seen(5, "oscar_middle_prep_batched_v2");
    out_stack.zero_();   // R144 P1 sha1 FIX: op reads out's initial value (shadow
                        // proved: out1=zeros vs carrier=prior -> differ). v1 path
                        // zeros out each step (torch.zeros); v2 carrier must too.
    uint64_t ws=0; aclOpExecutor* ex=nullptr;
    aclnnStatus r=aclnnOscarInt2FusedAttnInt2LseGetWorkspaceSize(
        a_q,a_k,a_v,nullptr,nullptr,a_asq,a_askv,
        nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,
        a_bt,nullptr,nullptr,
        a_ks,a_ko,a_vs,a_vo,
        numHeads,scale,0,0,lb.data(),numKvHeads,
        0,0,blockSize,1,1,1,
        a_o,a_lse,&ws,&ex);
    TORCH_CHECK(r==0, "R218-P7: GetWorkspaceSize failed r=",(int64_t)r);
    TORCH_CHECK(ex!=nullptr, "R218-P7: GetWorkspaceSize returned null executor");
    int64_t lr=-1;
    {
        void* ws_p=nullptr;
        if(ws>0){
            if(!g_ws.defined() || (uint64_t)g_ws.numel()<ws){
                g_ws=torch::empty({(int64_t)ws},torch::TensorOptions().dtype(torch::kUInt8).device(q_stack.device())); g_ws_hits=0;
            } else { g_ws_hits++; }
            ws_p=g_ws.data_ptr();
        }
        lr=(int64_t)aclnnOscarInt2FusedAttnInt2Lse(ws_p,ws,ex,s);
        TORCH_CHECK(lr==0, "R218-P7: aclnn launch failed lr=",lr);
    }
    aclDestroyTensor(a_q);aclDestroyTensor(a_k);aclDestroyTensor(a_v);aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks);aclDestroyTensor(a_ko);aclDestroyTensor(a_vs);aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o);aclDestroyTensor(a_lse);
    aclDestroyIntArray(a_asq);aclDestroyIntArray(a_askv);
    // R144 P1 sha1 shadow: AFTER the V2 op (V2 output ready). Same-process
    // byte-compare V1 (original gather) vs V2 carriers (inputs) + V1 op call
    // vs V2 output (out/LSE). Gated by R144_V2_SHADOW=1; runs 8 times.
    if (std::getenv("R144_V2_SHADOW") && std::string(std::getenv("R144_V2_SHADOW"))=="1"
        && g_shadow_calls < 8) {
        int64_t step = g_shadow_calls;
        g_shadow_calls++;
        shadow_v1v2(block_table, b_list, blk_lo_list, nb_list, mid_len_list,
            key_cache, value_cache, k_scale_cache, k_zero_cache,
            v_scale_cache, v_zero_cache, q_stack,
            K_contig, V_contig, bt_b, ks_b, ko_b, vs_b, vo_b,
            out_stack, lse_stack, nkv, blockSize, step, numHeads, scale);
    }
    py::tuple res=py::make_tuple((int64_t)r,(int64_t)ws,(int64_t)lr,g_ws_hits);
    return res;
}
// ===================== R145 v3: fixed-capacity TENSOR-ABI middle op =====================
// Contract (P0/P1/P2-proven): device int64 askv carrier [BcapT]; storage stride = CAP
// enforced by ONE sentinel row (last) with askv=CAP; active rows askv=mid_len; inactive
// active-slot rows askv=1. Host rectangular check skipped (device-resident askv).
// Kernel R28: stride=max(askv) live; logical=askv[b] live; tail masked. Graph-capturable
// op call (no host reads of askv; R133_GRAPH_V1=1 env required by tiling fallback).
static py::object oscar_middle_prep_v3_fill(
        torch::Tensor block_table,
        std::vector<int64_t> b_list, std::vector<int64_t> blk_lo_list,
        std::vector<int64_t> nb_list, std::vector<int64_t> mid_len_list,
        torch::Tensor key_cache, torch::Tensor value_cache,
        torch::Tensor k_scale_cache, torch::Tensor k_zero_cache,
        torch::Tensor v_scale_cache, torch::Tensor v_zero_cache,
        torch::Tensor K_carrier, torch::Tensor V_carrier,   // [BcapT*nbC, bs, nkv, d4] native
        torch::Tensor K_contig, torch::Tensor V_contig,     // [BcapT*nbC, nkv, bs, d4] op layout
        torch::Tensor kS, torch::Tensor kO, torch::Tensor vS, torch::Tensor vO, // [BcapT,nkv,CAP]
        torch::Tensor bt,                                    // [BcapT,nbC] int32
        torch::Tensor askv,                                  // [BcapT] int64
        int64_t Bp, int64_t BcapT, int64_t nbC, int64_t CAP,
        int64_t numKvHeads, int64_t blockSize) {
    const c10::DeviceGuard device_guard(askv.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(askv.device().index()));
    auto dev = askv.device();
    int64_t nkv = numKvHeads;
    int64_t B = (int64_t)b_list.size();
    TORCH_CHECK(B == Bp, "b_list size must equal Bp");
    TORCH_CHECK(askv.size(0) == BcapT, "askv carrier must be [BcapT]");
    // bt: CONSTANT per carrier (bt[r,j] = r*nbC + j — pure address layout, independent
    // of per-step data). Initialized once by the Python adapter at carrier creation;
    // nothing to do here per step. (R145 fill-overhead fix: removes 1 zero + B aranges.)
    // K/V gather: per-row blocks into row slots [i*nbC, i*nbC+nb_i).
    for (int64_t i = 0; i < B; i++) {
        int64_t nb_i = nb_list[i];
        if (nb_i > 0) {
            at::Tensor idx = block_table.narrow(1, blk_lo_list[i], nb_i).select(0, b_list[i]).to(at::kLong);
            at::Tensor kdst = K_carrier.narrow(0, i * nbC, nb_i);
            at::Tensor vdst = V_carrier.narrow(0, i * nbC, nb_i);
            at::index_select_out(kdst, key_cache, 0, idx);
            at::index_select_out(vdst, value_cache, 0, idx);
        }
    }
    // Transpose-copy ONLY the active prefix [0, Bp*nbC) (sentinel + inactive tails keep
    // stale/zero content — their outputs are discarded; bt rows point at slot 0 beyond nb).
    int64_t act = Bp * nbC;
    if (act > 0) {
        K_contig.narrow(0, 0, act).copy_(K_carrier.narrow(0, 0, act).permute({0, 2, 1, 3}));
        V_contig.narrow(0, 0, act).copy_(V_carrier.narrow(0, 0, act).permute({0, 2, 1, 3}));
    }
    // scales: per-row gather into carrier slices [i, :, 0:m]; kO = -(kz)*ks in place.
    for (int64_t i = 0; i < B; i++) {
        int64_t nb_i = nb_list[i];
        int64_t m = std::min(mid_len_list[i], nb_i * blockSize);
        if (m <= 0) continue;
        at::Tensor idx = block_table.narrow(1, blk_lo_list[i], nb_i).select(0, b_list[i]).to(at::kLong);
        auto row = [&](at::Tensor& c) { return c.select(0, i).narrow(1, 0, m); };
        at::Tensor ksg = at::index_select(k_scale_cache, 0, idx).reshape({nb_i * blockSize, nkv}).narrow(0, 0, m);
        row(kS).copy_(ksg.t());
        at::Tensor kzg = at::index_select(k_zero_cache, 0, idx).reshape({nb_i * blockSize, nkv}).narrow(0, 0, m);
        row(kO).copy_(kzg.t()); row(kO).mul_(row(kS)); row(kO).neg_();
        at::Tensor vsg = at::index_select(v_scale_cache, 0, idx).reshape({nb_i * blockSize, nkv}).narrow(0, 0, m);
        row(vS).copy_(vsg.t());
        at::Tensor vzg = at::index_select(v_zero_cache, 0, idx).reshape({nb_i * blockSize, nkv}).narrow(0, 0, m);
        row(vO).copy_(vzg.t()); row(vO).mul_(row(vS)); row(vO).neg_();
    }
    // askv: active rows = mid_len; inactive active-slots = 1; sentinel row [BcapT-1] = CAP
    // (set once at carrier init; never rewritten here).
    {
        at::Tensor lens = torch::tensor(mid_len_list,
            at::TensorOptions().dtype(torch::kInt64)).to(dev);
        askv.narrow(0, 0, Bp).copy_(lens);
        if (BcapT - 1 > Bp) askv.narrow(0, Bp, BcapT - 1 - Bp).fill_(1);
    }
    py::tuple res = py::make_tuple((int64_t)0, (int64_t)Bp, (int64_t)act);
    return res;
}
// R145 v3: the op call ONLY (graph-capturable; all args persistent carriers).
static py::object oscar_middle_op_v3(
        torch::Tensor q, torch::Tensor K_contig, torch::Tensor V_contig,
        torch::Tensor bt, torch::Tensor kS, torch::Tensor kO, torch::Tensor vS, torch::Tensor vO,
        torch::Tensor out, torch::Tensor lse,
        torch::Tensor asq, torch::Tensor askv,
        int64_t numHeads, double scale, int64_t numKvHeads, int64_t blockSize,
        bool do_zero) {
    const c10::DeviceGuard device_guard(q.device());
    TORCH_CHECK(askv.scalar_type() == at::kLong && asq.scalar_type() == at::kLong,
                "asq/askv must be int64 device tensors");
    if (do_zero) { out.zero_(); lse.fill_(torch::Scalar(-9999.0)); }
    // K/V carriers are uint8 (packed INT2 bytes); the op's ABI wants INT8 — view-cast
    // (same bytes, matching v2's k_b = K_contig.view(torch::kInt8) convention).
    at::Tensor k8 = K_contig.view(torch::kInt8);
    at::Tensor v8 = V_contig.view(torch::kInt8);
    aclTensor *a_q = make_acl(q), *a_k = make_acl(k8), *a_v = make_acl(v8), *a_bt = make_acl(bt);
    aclTensor *a_ks = make_acl(kS), *a_ko = make_acl(kO), *a_vs = make_acl(vS), *a_vo = make_acl(vO);
    aclTensor *a_o = make_acl(out), *a_lse = make_acl(lse);
    aclTensor *a_asq = make_acl(asq), *a_askv = make_acl(askv);
    std::vector<char> lb{'B', 'N', 'S', 'D', '\0'};
    aclrtStream s = c10_npu::getCurrentNPUStream(q.device().index()).stream(false);

    // R218-P7 commit boundary: host task-queue flush AFTER the sentinel fill
    // (and any caller staging enqueued before this call), BEFORE the direct
    // ACLNN submit. Submission-order guarantee only; NOT a device sync.
    s = r218_qbound_flush(q.device().index());
    TORCH_CHECK(s != nullptr, "R218-P7: task-queue flush failed dev=", (int)q.device().index());
    r218_qbound_seen(6, "oscar_middle_op_v3");
    uint64_t ws = 0; aclOpExecutor* ex = nullptr;
    aclnnStatus r = aclnnOscarInt2FusedAttnInt2LseTensorGetWorkspaceSize(
        a_q, a_k, a_v, /*pse*/nullptr, /*mask*/nullptr,
        a_asq, a_askv,
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        a_bt, nullptr, nullptr,
        a_ks, a_ko, a_vs, a_vo,
        numHeads, scale, 0, 0, lb.data(), numKvHeads,
        0, 0, blockSize, 1, 1, 1,
        a_o, a_lse, &ws, &ex);
    TORCH_CHECK(r == 0, "R218-P7: GetWorkspaceSize failed r=",(int64_t)r);
    TORCH_CHECK(ex != nullptr, "R218-P7: GetWorkspaceSize returned null executor");
    int64_t lr = -1;
    {
        void* ws_p = nullptr;
        if (ws > 0) {
            if (!g_ws.defined() || (uint64_t)g_ws.numel() < ws) {
                g_ws = torch::empty({(int64_t)ws}, torch::TensorOptions().dtype(torch::kUInt8).device(q.device()));
                g_ws_hits = 0;
            } else { g_ws_hits++; }
            ws_p = g_ws.data_ptr();
        }
        lr = (int64_t)aclnnOscarInt2FusedAttnInt2Lse(ws_p, ws, ex, s);
        TORCH_CHECK(lr == 0, "R218-P7: aclnn launch failed lr=", lr);
    }
    aclDestroyTensor(a_q); aclDestroyTensor(a_k); aclDestroyTensor(a_v); aclDestroyTensor(a_bt);
    aclDestroyTensor(a_ks); aclDestroyTensor(a_ko); aclDestroyTensor(a_vs); aclDestroyTensor(a_vo);
    aclDestroyTensor(a_o); aclDestroyTensor(a_lse);
    aclDestroyTensor(a_asq); aclDestroyTensor(a_askv);
    py::tuple res = py::make_tuple((int64_t)r, (int64_t)ws, (int64_t)lr, g_ws_hits);
    return res;
}
// R204_MIXCHAIN: single-crossing port of the eager mixed-step R146 fill layer
// (~15 ATen dispatches per layer-call in Python). SAME at:: ops in the SAME
// order on the SAME tensors => value-identical; one pybind crossing instead
// of ~15 Python dispatches (r203d census: ~35.6ms host per mixed layer-call).
// args: key/value caches + scale caches, the flat gather index IDX64 slice,
// the R146 carriers (K/V native, Kc/Vc op layout, ksgS..vzgS staging, kS..vO),
// widths BpF/nbC/bs/nkv/d4/nF.
static py::object oscar_fill_v3(
        torch::Tensor kc, torch::Tensor vc,
        torch::Tensor ksc, torch::Tensor kzc, torch::Tensor vsc, torch::Tensor vzc,
        torch::Tensor idxS,
        torch::Tensor K_fl, torch::Tensor V_fl,       // K/V viewed [-1, bs, nkv, d4]
        torch::Tensor Kc, torch::Tensor Vc,
        torch::Tensor ksgS, torch::Tensor kzgS, torch::Tensor vsgS, torch::Tensor vzgS,
        torch::Tensor kS, torch::Tensor kO, torch::Tensor vS, torch::Tensor vO,
        int64_t BpF, int64_t nbC, int64_t bs, int64_t nkv, int64_t d4) {
    const c10::DeviceGuard device_guard(idxS.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(idxS.device().index()));
    int64_t nF = BpF * nbC;
    at::Tensor idx = idxS.narrow(0, 0, nF);
    at::Tensor kfl = K_fl.narrow(0, 0, nF);
    at::Tensor vfl = V_fl.narrow(0, 0, nF);
    at::index_select_out(kfl, kc, 0, idx);
    at::index_select_out(vfl, vc, 0, idx);
    Kc.narrow(0, 0, nF).copy_(
        K_fl.narrow(0, 0, BpF * nbC).view({BpF, nbC, bs, nkv, d4})
            .permute({0, 1, 3, 2, 4}).reshape({nF, nkv, bs, d4}));
    Vc.narrow(0, 0, nF).copy_(
        V_fl.narrow(0, 0, BpF * nbC).view({BpF, nbC, bs, nkv, d4})
            .permute({0, 1, 3, 2, 4}).reshape({nF, nkv, bs, d4}));
    at::Tensor ksg = ksgS.narrow(0, 0, nF);
    at::Tensor kzg = kzgS.narrow(0, 0, nF);
    at::Tensor vsg = vsgS.narrow(0, 0, nF);
    at::Tensor vzg = vzgS.narrow(0, 0, nF);
    at::index_select_out(ksg, ksc, 0, idx);
    at::index_select_out(kzg, kzc, 0, idx);
    at::index_select_out(vsg, vsc, 0, idx);
    at::index_select_out(vzg, vzc, 0, idx);
    kS.narrow(0, 0, BpF).copy_(
        ksg.view({BpF, nbC * bs, nkv}).permute({0, 2, 1}));
    kO.narrow(0, 0, BpF).copy_(
        kzg.view({BpF, nbC * bs, nkv}).permute({0, 2, 1}));
    kO.narrow(0, 0, BpF).mul_(kS.narrow(0, 0, BpF));
    kO.narrow(0, 0, BpF).neg_();
    vS.narrow(0, 0, BpF).copy_(
        vsg.view({BpF, nbC * bs, nkv}).permute({0, 2, 1}));
    vO.narrow(0, 0, BpF).copy_(
        vzg.view({BpF, nbC * bs, nkv}).permute({0, 2, 1}));
    vO.narrow(0, 0, BpF).mul_(vS.narrow(0, 0, BpF));
    vO.narrow(0, 0, BpF).neg_();
    return py::make_tuple((int64_t)0);
}
// R204_MIXCHAIN: single-crossing port of the batched S+R merge tail
// (max/exp/weighted-sum/div + bf16 cast + V-unrot matmul + writeback —
// ~10 Python dispatches per layer-call). Same at:: ops on the same values.
// out_dst rows [0, B) are written; Rv may be empty (no-unrot models).
static py::object oscar_merge_tail(
        torch::Tensor l_pr, torch::Tensor l_h,
        torch::Tensor o_pr, torch::Tensor o_h,
        torch::Tensor Rv, torch::Tensor out_dst, int64_t B) {
    const c10::DeviceGuard device_guard(l_pr.device());
    const c10_npu::NPUStreamGuard stream_guard(
        c10_npu::getCurrentNPUStream(l_pr.device().index()));
    at::Tensor m = at::maximum(l_pr, l_h);
    at::Tensor w_pr = at::sub(l_pr, m).exp_();
    at::Tensor w_h = at::sub(l_h, m).exp_();
    at::Tensor num = at::add(w_pr.unsqueeze(-1) * o_pr, w_h.unsqueeze(-1) * o_h);
    at::Tensor den = at::add(w_pr, w_h).unsqueeze(-1);
    at::Tensor merged = num.div_(den);
    at::Tensor outb = merged.to(out_dst.scalar_type());
    if (Rv.numel() > 0) {
        outb = at::matmul(outb, Rv.transpose(-1, -2));
    }
    out_dst.narrow(0, 0, B).copy_(outb.narrow(0, 0, B));
    return py::make_tuple((int64_t)0);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("r112fast_version",[](){return std::string("r113.orch.v9+r145v3+r218p7bound");});
    m.def("launch_lse_fast",&launch_lse_fast,"R112: sync-free launch, cached ws");
    m.def("verify_lse",&verify_lse,"R112: explicit readback for correctness legs");
    m.def("warm_ws",&warm_ws,"R113: pre-grow cached ws outside capture/task-update regions");
    m.def("oscar_middle_prep",&oscar_middle_prep,"R113 Fallback-1: single-call middle prep+launch (bit-exact)");
    m.def("oscar_middle_prep_batched",&oscar_middle_prep_batched,"R144 G1: general runtime-B batched middle prep+launch (one aclnn call, B requests)");
    m.def("oscar_middle_prep_batched_v2",&oscar_middle_prep_batched_v2,"R144 P1 StageV2: batched middle into persistent carriers (alloc=0/cat=0/CPU-bt=0/H2D=0)");
    m.def("oscar_middle_replay",&oscar_middle_replay,"R144 G1: B=1 replay from pre-gathered operands (controlled A/B/C arms)");
    m.def("oscar_middle_replay_batched",&oscar_middle_replay_batched,"R144 G1: B-agnostic batched replay from pre-gathered stacked operands");
    m.def("r144_read_counter",&r144_read_counter,"R144 G1: read counter table from counter .o output (1/G proof at B>1)");
    m.def("oscar_quant_kv",&oscar_quant_kv,"R113 Fallback-1b: single-call K/V INT2 quantize (bit-exact values)");
    m.def("oscar_scatter_int2",&oscar_scatter_int2,"R113 Fallback-1c: single-call paged KV scatter");
    m.def("oscar_middle_prep_v3_fill",&oscar_middle_prep_v3_fill,"R145 v3: fixed-capacity carrier fills (eager; per-step metadata)");
    m.def("oscar_middle_op_v3",&oscar_middle_op_v3,"R145 v3: fixed-capacity tensor-ABI op call (graph-capturable)");
    m.def("oscar_fill_v3",&oscar_fill_v3,"R204: single-crossing eager mixed-step fill layer (value-identical port)");
    m.def("oscar_merge_tail",&oscar_merge_tail,"R204: single-crossing S+R merge tail (value-identical port)");
    m.def("qbound_stats",[](){ return py::make_tuple((int64_t)g_qb_calls.load(), (int64_t)g_qb_ns.load(), (int64_t)g_qb_max_ns.load()); }, "R218-P7 commit-boundary counters (calls, ns_total, ns_max)");
}
