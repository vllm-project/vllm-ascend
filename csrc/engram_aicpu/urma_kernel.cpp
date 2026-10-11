// SPDX-License-Identifier: Apache-2.0
// Experimental 950DT UDMA path, locked to the audited device liburma ABI.
#include "urma_params.h"
#include <urma_api.h>
#include <urma_provider.h>
#include <dirent.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <new>

namespace {
constexpr uint64_t PAGE_SIZE = 4096;
constexpr unsigned WINDOW = 2048;
constexpr unsigned READ_WINDOW = 1024;
constexpr uint64_t COMPLETION_TIMEOUT_NS = 500000000;
constexpr size_t SYSFS_DEVICE_BACKPOINTER = 400;
constexpr size_t READER_CODE_BYTES = 3680;
constexpr uint64_t READER_CODE_HASH = 10952995829187389118ULL;
struct Engine {
    urma_context_t* context = nullptr;
    urma_jfc_t* jfc = nullptr;
    urma_jfs_t* jfs = nullptr;
    urma_target_seg_t* local = nullptr;
    urma_target_seg_t* remote = nullptr;
    urma_target_jetty_t* target = nullptr;
    urma_device_t* device = nullptr;
    void* descriptor = nullptr;
    void* provider = nullptr;
    uint64_t base = 0, length = 0;
    bool advised = false;
    uint64_t die = 0;
    uint32_t last_completion = 0;
    urma_jfs_wr_t requests[WINDOW]{};
    urma_sge_t source_sges[WINDOW]{}, destination_sges[WINDOW]{};
};
uint64_t now_ns() {
    timespec t{};clock_gettime(CLOCK_MONOTONIC, &t);
    return uint64_t(t.tv_sec)*1000000000+t.tv_nsec;
}
bool release(Engine* e) {
    if (e->advised && urma_unadvise_jfr(e->jfs,e->target)) return false;
    e->advised = false;
    if (e->jfs && urma_delete_jfs(e->jfs)) return false;
    e->jfs = nullptr;
    if (e->target && urma_unimport_jfr(e->target)) return false;
    e->target = nullptr;
    if (e->remote && urma_unimport_seg(e->remote)) return false;
    e->remote = nullptr;
    if (e->local && urma_unregister_seg(e->local)) return false;
    e->local = nullptr;
    if (e->jfc && urma_delete_jfc(e->jfc)) return false;
    e->jfc = nullptr;
    if (e->context && urma_delete_context(e->context)) return false;
    e->context = nullptr;
    free(e->device);free(e->descriptor);
    if (e->provider) dlclose(e->provider);
    delete e;
    return true;
}
bool initialize(Engine* e, const EngramUrmaParams& p) {
    auto reader = reinterpret_cast<void*(*)(dirent*)>(dlsym(RTLD_DEFAULT,"urma_read_sysfs_device"));
    if (!reader) return false;
    uint64_t hash = 14695981039346656037ULL;
    auto bytes = reinterpret_cast<const unsigned char*>(reader);
    for (size_t i=0;i<READER_CODE_BYTES;++i) hash = (hash^bytes[i])*1099511628211ULL;
    if (hash != READER_CODE_HASH || p.endpoint_chip>1 || p.endpoint_die>1) return false;
    dirent entry{};
    snprintf(entry.d_name,sizeof(entry.d_name),"udmac%lud%lue3",p.endpoint_chip,p.endpoint_die);
    e->descriptor = reader(&entry);
    e->provider = dlopen("/usr/lib64/urma/liburma-udma.so",RTLD_NOW|RTLD_LOCAL);
    auto ops = e->provider ? reinterpret_cast<urma_provider_ops_t*>(dlsym(e->provider,"g_udma_provider_ops")) : nullptr;
    if (!e->descriptor || !ops) return false;
    e->device = static_cast<urma_device_t*>(calloc(1,sizeof(urma_device_t)));
    if (!e->device) return false;
    strcpy(e->device->name,entry.d_name);
    snprintf(e->device->path,sizeof(e->device->path),"/dev/uburma/%s",entry.d_name);
    e->device->type=URMA_TRANSPORT_UB;e->device->ops=ops;
    e->device->sysfs_dev=reinterpret_cast<urma_sysfs_dev*>(e->descriptor);
    memcpy(static_cast<char*>(e->descriptor)+SYSFS_DEVICE_BACKPOINTER,&e->device,sizeof(e->device));
    e->context=urma_create_context(e->device,0);
    if (!e->context) return false;
    urma_jfc_cfg_t cq{};cq.depth=32;
    e->jfc=urma_create_jfc(e->context,&cq);
    if (!e->jfc) return false;
    urma_jfs_cfg_t queue{};queue.depth=4096;queue.trans_mode=URMA_TM_RM;
    queue.max_sge=queue.max_rsge=1;queue.jfc=e->jfc;
    e->jfs=urma_create_jfs(e->context,&queue);
    if (!e->jfs) return false;
    urma_seg_t segment{};urma_rjfr_t receiver{};urma_token_t token{};
    memcpy(&segment,reinterpret_cast<void*>(p.metadata),sizeof(segment));
    memcpy(&receiver,reinterpret_cast<char*>(p.metadata)+64,sizeof(receiver));
    urma_import_seg_flag_t flag{};flag.bs.access=URMA_ACCESS_READ;
    e->remote=urma_import_seg(e->context,&segment,&token,0,flag);
    e->target=urma_import_jfr(e->context,&receiver,&token);
    if (!e->remote || !e->target) return false;
    int rc=urma_advise_jfr(e->jfs,e->target);
    e->advised=(rc==0 || rc==17);
    e->die=p.endpoint_die;
    return e->advised;
}
int64_t read_id(const EngramUrmaParams& p,int64_t row) {
    auto offset=(row/p.local_heads)*p.ids_stride+p.head_start+row%p.local_heads;
    return p.id_bytes==4 ? reinterpret_cast<const int32_t*>(p.ids)[offset] : reinterpret_cast<const int64_t*>(p.ids)[offset];
}
bool gather(Engine* e,const EngramUrmaParams& p) {
    uint64_t base=p.codes&~(PAGE_SIZE-1);
    uint64_t length=((p.codes+p.tokens*p.local_heads*p.width+PAGE_SIZE-1)&~(PAGE_SIZE-1))-base;
    if (e->local && (base!=e->base || length!=e->length)) {
        if (urma_unregister_seg(e->local)) return false;
        e->local=nullptr;
    }
    if (!e->local) {
        urma_seg_cfg_t cfg{};cfg.va=base;cfg.len=length;
        cfg.flag.bs.access=URMA_ACCESS_READ|URMA_ACCESS_WRITE;cfg.flag.bs.non_pin=1;
        e->local=urma_register_seg(e->context,&cfg);
        if (!e->local) return false;
        e->base=base;e->length=length;
    }
    static_assert(WINDOW == 2*READ_WINDOW,"Lookahead requires two existing banks");
    e->last_completion=0;
    urma_seg_t segment{};memcpy(&segment,reinterpret_cast<void*>(p.metadata),sizeof(segment));
    const int64_t rows=p.tokens*p.local_heads;
    bool pending=false;
    uint64_t pending_end=0, deadline=0;
    unsigned bank=0;
    auto complete_pending = [&]() -> bool {
        if (!pending) return true;
        urma_cr_t cr{};
        while (true) {
            int count=urma_poll_jfc(e->jfc,1,&cr);
            if (count<0) return false;
            if (count>0) {
                e->last_completion=cr.status;
                if (cr.status!=0 || cr.user_ctx!=pending_end) return false;
                pending=false;
                return true;
            }
            if (now_ns()>=deadline) return false;
        }
    };
    for (int64_t begin=0;begin<rows;begin+=READ_WINDOW) {
        int64_t end=begin+READ_WINDOW<rows?begin+READ_WINDOW:rows;
        auto requests=e->requests+bank*READ_WINDOW;
        auto source_sges=e->source_sges+bank*READ_WINDOW;
        auto destination_sges=e->destination_sges+bank*READ_WINDOW;
        unsigned n=0;
        for (int64_t row=begin;row<end;++row) {
            int64_t index=read_id(p,row);
            uint64_t dest=p.codes+row*p.width;
            if (index<p.vocab_start || index>=p.vocab_end) {
                memset(reinterpret_cast<void*>(dest),0,p.width);continue;
            }
            source_sges[n].addr=segment.ubva.va+uint64_t(index-p.vocab_start)*p.width;
            source_sges[n].len=p.width;source_sges[n].tseg=e->remote;
            destination_sges[n].addr=dest;destination_sges[n].len=p.width;destination_sges[n].tseg=e->local;
            auto& wr=requests[n];wr.opcode=URMA_OPC_READ;wr.flag.value=0;wr.tjetty=e->target;wr.user_ctx=end;
            wr.rw.src.sge=source_sges+n;wr.rw.src.num_sge=1;
            wr.rw.dst.sge=destination_sges+n;wr.rw.dst.num_sge=1;
            wr.next=requests+n+1;++n;
        }
        // Even an all-invalid batch drains the previous transfer. It never
        // emits a WR or completion, and writes only its own output row range.
        if (!complete_pending()) return false;
        if (!n) continue;
        auto& last=requests[n-1];last.next=nullptr;
        last.flag.bs.complete_enable=1;last.flag.bs.comp_order=1;last.flag.bs.fence=1;
        urma_jfs_wr_t* bad=nullptr;
        if (urma_post_jfs_wr(e->jfs,requests,&bad)) return false;
        pending=true;pending_end=end;deadline=now_ns()+COMPLETION_TIMEOUT_NS;
        bank=1-bank;
    }
    return complete_pending();
}
static_assert(sizeof(urma_jfs_wr_t)==80 && sizeof(urma_sge_t)==32 && sizeof(urma_seg_t)==48 && sizeof(urma_rjfr_t)==40,"Unsupported URMA ABI");
}
extern "C" __attribute__((visibility("default"))) uint32_t EngramUrmaGather(void* raw) {
    EngramUrmaParams p{};memcpy(&p,static_cast<char*>(raw)+20,sizeof(p));
    auto slot=reinterpret_cast<uint64_t*>(p.state);
    auto e=reinterpret_cast<Engine*>(*slot);
    if (p.close) {
        if (e && !release(e)) return 1;
        *slot=0;return 0;
    }
    if (!e) {
        e=new(std::nothrow) Engine();
        if (!e) return 1;
        *slot=reinterpret_cast<uint64_t>(e);
        if (!initialize(e,p)) return 1;
    }
    if (!e->advised) return 1;
    if (!gather(e,p)) {
        // Probe only the other local die for a local-access completion. The
        // old queue must drain before retrying; transport/timeouts still fail.
        if (e->last_completion!=URMA_CR_LOC_ACCESS_ERR) return 1;
        p.endpoint_die=1-e->die;
        if (!release(e)) return 1;
        *slot=0;
        e=new(std::nothrow) Engine();
        if (!e) return 1;
        *slot=reinterpret_cast<uint64_t>(e);
        if (!initialize(e,p) || !gather(e,p)) return 1;
    }
    return 0;
}
