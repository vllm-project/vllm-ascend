/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "hccl_comm_wrapper.h"

#ifdef VLLM_ASCEND_HCCL_QOS_CONFIG

#include <cstdint>
#include <cstring>

#include <hccl/hccl.h>

extern "C" PyObject* python_hccl_comm_init_root_info_config(
    PyObject* self, PyObject* args) {
  (void)self;
  unsigned int rank_size;
  unsigned long long root_info_address;
  unsigned int rank;
  unsigned int sdma_qos;
  unsigned int service_level;
  unsigned int traffic_class;
  const char* group_name = nullptr;
  if (!PyArg_ParseTuple(args, "IKIIIIs", &rank_size, &root_info_address, &rank,
                        &sdma_qos, &service_level, &traffic_class,
                        &group_name)) {
    return nullptr;
  }
  if (rank_size == 0 || rank >= rank_size) {
    PyErr_SetString(PyExc_ValueError,
                    "rank_size must be positive and rank must be in range");
    return nullptr;
  }
  if (root_info_address == 0) {
    PyErr_SetString(PyExc_ValueError, "root_info_address must not be null");
    return nullptr;
  }
  if (group_name == nullptr || group_name[0] == '\0') {
    PyErr_SetString(PyExc_ValueError, "group_name must not be empty");
    return nullptr;
  }

  const size_t name_len = strnlen(group_name, UDI_MAX_LENGTH);
  if (name_len >= UDI_MAX_LENGTH) {
    PyErr_SetString(PyExc_ValueError, "group_name exceeds UDI_MAX_LENGTH");
    return nullptr;
  }

  const auto* root_info = reinterpret_cast<const HcclRootInfo*>(
      static_cast<uintptr_t>(root_info_address));
  HcclCommConfig config;
  HcclCommConfigInit(&config);
  config.hcclQos = sdma_qos;
  config.hcclRdmaServiceLevel = service_level;
  config.hcclRdmaTrafficClass = traffic_class;
  memcpy(config.hcclUdi, group_name, name_len);
  config.hcclUdi[name_len] = '\0';

  HcclComm comm = nullptr;
  const HcclResult result = HcclCommInitRootInfoConfig(
      rank_size, root_info, rank, &config, &comm);
  return Py_BuildValue(
      "(iK)", static_cast<int>(result),
      static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(comm)));
}

#endif
