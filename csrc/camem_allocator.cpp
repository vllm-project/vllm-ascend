/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

#include <iostream>

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include "acl/acl.h"

// Global references to Python callables
// NOTE: this is borrowed reference, so we don't need to DECREF them.
// This brings the limitation that the allocator needs to be singleton.
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;


// ---------------------------------------------------------------------------
// Helper functions:

void ensure_context(unsigned long long device) {
  aclrtContext pctx;
  aclrtGetCurrentContext(&pctx);
  if (!pctx) {
    // Ensure device context.
    aclrtCreateContext(&pctx, device);
    aclrtSetCurrentContext(pctx);
  }
}

void create_and_map(unsigned long long device, ssize_t size, void* d_mem,
                    aclrtDrvMemHandle* p_memHandle) {
  ensure_context(device);
  // Define memory allocation properties
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE ;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = device;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.reserve = 0;

  // Allocate memory using aclrtMallocPhysical
  aclError error_code = aclrtMallocPhysical(p_memHandle, size, &prop, 0);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;  
    return;
  }
  error_code = aclrtMapMem(d_mem, size, 0, *p_memHandle, 0);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;  
    return;
  }
}

void unmap_and_release(unsigned long long device, ssize_t size,
                       void* d_mem,
                       aclrtDrvMemHandle* p_memHandle) {
  // std::cout << "unmap_and_release: device=" << device << ", size=" << size <<
  // ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
  ensure_context(device);
  aclError error_code = aclrtUnmapMem(d_mem);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;  
    return;
  }
  error_code = aclrtFreePhysical(*p_memHandle);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;  
    return;
  }
}

PyObject* create_tuple_from_c_integers(unsigned long long a,
                                       unsigned long long b,
                                       unsigned long long c,
                                       unsigned long long d) {
  // Create a new tuple of size 4
  PyObject* tuple = PyTuple_New(4);
  if (!tuple) {
    return NULL;  // Return NULL on failure
  }

  // Convert integers to Python objects and set them in the tuple
  PyTuple_SetItem(
      tuple, 0,
      PyLong_FromUnsignedLongLong(a));  // Steals reference to the PyLong
  PyTuple_SetItem(tuple, 1, PyLong_FromUnsignedLongLong(b));
  PyTuple_SetItem(tuple, 2, PyLong_FromUnsignedLongLong(c));
  PyTuple_SetItem(tuple, 3, PyLong_FromUnsignedLongLong(d));

  // Note: PyTuple_SetItem "steals" a reference to each object,
  // so we do not need to Py_DECREF the PyLong objects explicitly.

  return tuple;  // Return the created tuple
}

// ---------------------------------------------------------------------------
// Our exported C functions that call Python:

__attribute__ ((visibility("default"))) void* my_malloc(ssize_t size, int device, aclrtStream stream) {
  ensure_context(device);

  // first allocation, align the size, and reserve an address, and also allocate
  // a aclrtDrvMemHandle

  // Define memory allocation properties
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE ;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = device;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.reserve = 0;

  // Check if the allocation is supported
  size_t granularity;
  aclError error_code = aclrtMemGetAllocationGranularity(&prop,
                                   ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
                                   &granularity);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
            << __LINE__ << std::endl;  
    return nullptr;
  }
  size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;
  void *d_mem;
  error_code = aclrtReserveMemAddress(&d_mem, alignedSize, 0, nullptr, 0);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
                << __LINE__ << std::endl;  
    return nullptr;
  }
  // allocate the aclrtDrvMemHandle
  aclrtDrvMemHandle* p_memHandle =
      (aclrtDrvMemHandle*)malloc(sizeof(aclrtDrvMemHandle));

  if (!g_python_malloc_callback) {
    std::cerr << "ERROR: g_python_malloc_callback not set.\n";
    return nullptr;
  }

  // Acquire GIL (not in stable ABI officially, but often works)
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* arg_tuple = create_tuple_from_c_integers(
      (unsigned long long)device, (unsigned long long)alignedSize,
      (unsigned long long)d_mem, (unsigned long long)p_memHandle);

  // Call g_python_malloc_callback
  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
  Py_DECREF(arg_tuple);

  if (!py_result) {
    PyErr_Print();
    PyGILState_Release(gstate);
    return nullptr;
  }

  PyGILState_Release(gstate);

  // do the final mapping
  create_and_map(device, alignedSize, d_mem, p_memHandle);

  return (void*)d_mem;
}

__attribute__ ((visibility("default"))) void my_free(void* ptr, ssize_t size, int device, aclrtStream stream) {
  // get memory handle from the pointer
  if (!g_python_free_callback) {
    std::cerr << "ERROR: g_python_free_callback not set.\n";
    return;
  }

  // Acquire GIL (not in stable ABI officially, but often works)
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* py_ptr =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));

  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);

  if (!py_result || !PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem, recv_p_memHandle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(py_result, "KKKK", &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return;
  }

  PyGILState_Release(gstate);

  // recv_size == size
  // recv_device == device

  // Free memory

  void *d_mem = (void*)recv_d_mem;
    // allocate the aclrtDrvMemHandle
  aclrtDrvMemHandle* p_memHandle =
      (aclrtDrvMemHandle*)recv_p_memHandle;
  unmap_and_release(device, size, d_mem, p_memHandle);

  // free address and the handle
  aclError error_code = aclrtReleaseMemAddress(d_mem);
  if (error_code != 0) {
    std::cerr << "acl Error, code: " << error_code << " at " << __FILE__ << ":" \
        << __LINE__ << std::endl;  
    return;
  }
  free(p_memHandle);
}

// ---------------------------------------------------------------------------
// Python extension boilerplate:

// Python-exposed function: init_module(python_malloc, python_free)
static PyObject* py_init_module(PyObject* self, PyObject* args) {
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;

  if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
    return nullptr;
  }

  if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }

  // Save the Python callables
  // This module does not handle GC of these objects, so they must be kept alive
  // outside of this module.
  g_python_malloc_callback = malloc_callback;
  g_python_free_callback = free_callback;

  Py_RETURN_NONE;
}

static PyObject* python_unmap_and_release(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem, recv_p_memHandle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, "KKKK", &recv_device, &recv_size, &recv_d_mem,
                        &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  void *d_mem_ptr = (void*)recv_d_mem;
  aclrtDrvMemHandle* p_memHandle =
      (aclrtDrvMemHandle*)recv_p_memHandle;

  unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle);

  Py_RETURN_NONE;
}

static PyObject* python_create_and_map(PyObject* self, PyObject* args) {
  if (!args || !PyTuple_Check(args) || PyTuple_Size(args) != 4) {
    PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    return nullptr;
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem, recv_p_memHandle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(args, "KKKK", &recv_device, &recv_size, &recv_d_mem,
                        &recv_p_memHandle)) {
    // PyArg_ParseTuple sets an error if it fails
    return nullptr;
  }

  void *d_mem_ptr = (void*)recv_d_mem;
  aclrtDrvMemHandle* p_memHandle =
      (aclrtDrvMemHandle*)recv_p_memHandle;

  create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle);

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"init_module", (PyCFunction)py_init_module, METH_VARARGS,
     "Initialize module with python_malloc and python_free callables."},
    {"python_create_and_map", (PyCFunction)python_create_and_map, METH_VARARGS,
     "Create and map memory on the device."},
    {"python_unmap_and_release", (PyCFunction)python_unmap_and_release,
     METH_VARARGS, "Unmap and release memory on the device."},
    {NULL, NULL, 0, NULL}  // sentinel
};

static struct PyModuleDef camem_allocator_module = {
    PyModuleDef_HEAD_INIT, "camem_allocator",
    "CANN-mem-based allocator for NPUPluggableAllocator", -1, module_methods};

PyMODINIT_FUNC PyInit_vllm_ascend_C(void) {
  // Initialize the module
  PyObject* module = PyModule_Create(&camem_allocator_module);
  if (!module) {
    return NULL;
  }
  return module;
}
}  // extern "C"
