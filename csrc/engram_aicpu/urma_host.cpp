// SPDX-License-Identifier: Apache-2.0
#include <urma_api.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
struct HostState {
  urma_context_t* context;
  urma_target_seg_t* segment;
  urma_jfc_t* jfc;
  urma_jfr_t* jfr;
};
extern "C" void EngramReadRelease(void* raw) {
  if (!raw) return;
  auto s = static_cast<HostState*>(raw);
  if (s->jfr) urma_delete_jfr(s->jfr);
  if (s->jfc) urma_delete_jfc(s->jfc);
  if (s->segment) urma_unregister_seg(s->segment);
  if (s->context) urma_delete_context(s->context);
  free(s);
}
extern "C" void* EngramReadRegister(void* address,size_t length,void* metadata,
                                   unsigned* status,unsigned chip) {
  char name[] = "udmac0d1e2";
  if (chip > 1) return nullptr;
  name[5] = char('0' + chip);
  status[0] = sizeof(urma_seg_t); status[1] = sizeof(urma_rjfr_t);
  auto dev = urma_get_device_by_name(name);
  status[2] = dev ? 0 : errno;
  if (!dev) return nullptr;
  auto s = static_cast<HostState*>(calloc(1,sizeof(HostState)));
  if (!s) return nullptr;
  s->context = urma_create_context(dev,0);
  status[3] = s->context ? 0 : errno;
  if (!s->context) { EngramReadRelease(s); return nullptr; }
  urma_seg_cfg_t seg{}; seg.va = reinterpret_cast<uint64_t>(address); seg.len = length;
  seg.flag.bs.access = URMA_ACCESS_READ | URMA_ACCESS_WRITE;
  s->segment = urma_register_seg(s->context,&seg);
  status[4] = s->segment ? 0 : errno;
  if (!s->segment) { EngramReadRelease(s); return nullptr; }
  urma_jfc_cfg_t jfc{}; jfc.depth = 32;
  s->jfc = urma_create_jfc(s->context,&jfc);
  status[5] = s->jfc ? 0 : errno;
  if (!s->jfc) { EngramReadRelease(s); return nullptr; }
  urma_jfr_cfg_t jfr{}; jfr.depth = 8; jfr.max_sge = 1;
  jfr.trans_mode = URMA_TM_RM; jfr.jfc = s->jfc;
  s->jfr = urma_create_jfr(s->context,&jfr);
  status[6] = s->jfr ? 0 : errno;
  if (!s->jfr) { EngramReadRelease(s); return nullptr; }
  urma_rjfr_t remote{}; remote.jfr_id = s->jfr->jfr_id;
  remote.trans_mode = jfr.trans_mode; remote.tp_type = URMA_CTP;
  memcpy(metadata,&s->segment->seg,sizeof(urma_seg_t));
  memcpy(static_cast<char*>(metadata)+64,&remote,sizeof(remote));
  return s;
}
