// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
#include <Python.h>
#include <hccl/hccl_comm.h>
#include <cstdint>
#include <vector>

// HcclCommConfig is versioned by CANN. Initialize it in C++ using the
// installed headers rather than duplicating the ABI in a ctypes structure.
extern "C" PyObject* python_hccl_subcomm(PyObject*, PyObject* args) {
  unsigned long long function_address, parent_address, sub_id;
  unsigned int rank, buffer_mb;
  PyObject* rank_ids;
  if (!PyArg_ParseTuple(args, "KKOKII", &function_address, &parent_address,
                        &rank_ids, &sub_id, &rank, &buffer_mb)) return nullptr;
  PyObject* sequence = PySequence_Fast(rank_ids, "rank_ids must be a sequence");
  if (!sequence) return nullptr;
  const auto size = PySequence_Fast_GET_SIZE(sequence);
  if (!function_address || !parent_address || size < 1 || !buffer_mb) {
    Py_DECREF(sequence);
    PyErr_SetString(PyExc_ValueError, "Invalid HCCL sub-communicator arguments");
    return nullptr;
  }
  std::vector<uint32_t> ranks;
  for (Py_ssize_t i = 0; i < size; ++i) {
    unsigned long value = PyLong_AsUnsignedLong(PySequence_Fast_GET_ITEM(sequence, i));
    if (PyErr_Occurred() || value > UINT32_MAX) {
      Py_DECREF(sequence);
      if (!PyErr_Occurred()) PyErr_SetString(PyExc_ValueError, "Rank exceeds uint32");
      return nullptr;
    }
    ranks.push_back(static_cast<uint32_t>(value));
  }
  Py_DECREF(sequence);
  HcclCommConfig config;
  HcclCommConfigInit(&config);
  config.hcclBufferSize = buffer_mb;
  HcclComm parent = reinterpret_cast<HcclComm>(parent_address), child = nullptr;
  auto function = reinterpret_cast<decltype(&HcclCreateSubCommConfig)>(function_address);
  HcclResult result;
  // Every parent rank participates; excluded ranks use the documented sentinel.
  bool included = rank < ranks.size();
  Py_BEGIN_ALLOW_THREADS
  result = function(&parent, ranks.size(), included ? ranks.data() : nullptr,
                    included ? sub_id : UINT32_MAX, included ? rank : 0,
                    &config, &child);
  Py_END_ALLOW_THREADS
  if (result != HCCL_SUCCESS) {
    PyErr_Format(PyExc_RuntimeError, "HcclCreateSubCommConfig failed: %d", result);
    return nullptr;
  }
  return PyLong_FromVoidPtr(child);
}
