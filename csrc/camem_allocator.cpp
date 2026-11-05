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
#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <sys/types.h>
#include "acl/acl.h"

// The fourth Python handle field remains an opaque integer pointer. Track
// physical ownership separately from the retained virtual reservation so free
// is valid both before and after sleep, including partial driver failures.
struct CaMemAllocation {
  aclrtDrvMemHandle physical = nullptr;
  bool mapped = false;
  std::mutex mutex;
};

// Global references to Python callables
// NOTE: this is borrowed reference, so we don't need to DECREF them.
// This brings the limitation that the allocator needs to be singleton.
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;
static PyObject* g_python_free_complete_callback = nullptr;
// A void deleter cannot report failure to its caller safely. Preserve the
// ownership record and make the next Python lifecycle/inference boundary fail
// closed instead of unwinding through a tensor destructor or replaying a graph.
static std::atomic<bool> g_free_failed{false};

void check_free_error() {
  if (g_free_failed.load()) {
    throw std::runtime_error("CaMem free callback failed; restart the worker before inference");
  }
}


// ---------------------------------------------------------------------------
// Helper functions:

void ensure_context(unsigned long long device) {
  aclrtContext pctx = nullptr;
  aclrtGetCurrentContext(&pctx);
  if (!pctx) {
    // Ensure device context.
    aclrtCreateContext(&pctx, device);
    aclrtSetCurrentContext(pctx);
  }
}

void create_and_map(unsigned long long device, ssize_t size, void* d_mem,
                    CaMemAllocation* allocation) {
  if (!allocation) {
    throw std::runtime_error("create_and_map: allocation handle is null");
  }
  check_free_error();
  std::lock_guard<std::mutex> lock(allocation->mutex);
  if (allocation->mapped) {
    return;
  }
  ensure_context(device);
  // Define memory allocation properties
  aclrtPhysicalMemProp prop = {};
  prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
  prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
  prop.memAttr = ACL_HBM_MEM_HUGE;
  prop.location.id = device;
  prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
  prop.reserve = 0;

  // Allocate memory using aclrtMallocPhysical
  aclError error_code = ACL_SUCCESS;
  if (allocation->physical == nullptr) {
    error_code = aclrtMallocPhysical(&allocation->physical, size, &prop, 0);
  }
  if (error_code != 0) {
    if (error_code == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      throw std::runtime_error("aclrtMallocPhysical failed with acl error code: " + 
                              std::to_string(error_code) + "(OOM: Out of Memory, allocation failed) " + 
                              __FILE__ + ":" + std::to_string(__LINE__));
    } else {
      throw std::runtime_error("aclrtMallocPhysical failed with acl error code: " +
                              std::to_string(error_code) + " " + __FILE__ + ":" + std::to_string(__LINE__));
    }
  }

  // Map memory
  error_code = aclrtMapMem(d_mem, size, 0, allocation->physical, 0);
  if (error_code != 0) {
    // Keep ownership if FreePhysical also fails, so a later cleanup can retry.
    if (aclrtFreePhysical(allocation->physical) == ACL_SUCCESS) {
      allocation->physical = nullptr;
    }
    throw std::runtime_error("aclrtMapMem failed with acl error code: " +
                            std::to_string(error_code) + " " + __FILE__ + ":" + std::to_string(__LINE__));
  }
  allocation->mapped = true;
}

void unmap_and_release(unsigned long long device, ssize_t size,
                       void* d_mem,
                       CaMemAllocation* allocation) {
  if (!allocation) {
    throw std::runtime_error("unmap_and_release: allocation handle is null");
  }
  // std::cout << "unmap_and_release: device=" << device << ", size=" << size <<
  // ", d_mem=" << d_mem << ", p_memHandle=" << p_memHandle << std::endl;
  ensure_context(device);
  std::lock_guard<std::mutex> lock(allocation->mutex);
  aclError error_code = ACL_SUCCESS;
  if (allocation->mapped) {
    error_code = aclrtUnmapMem(d_mem);
  }
  if (error_code != 0) {
    throw std::runtime_error("aclrtUnmapMem failed with acl error code: " +
                            std::to_string(error_code) + " " + __FILE__ + ":" + std::to_string(__LINE__));
  }
  allocation->mapped = false;
  if (allocation->physical == nullptr) {
    return;
  }
  error_code = aclrtFreePhysical(allocation->physical);
  if (error_code != 0) {
    throw std::runtime_error("aclrtFreePhysical failed with acl error code: " +
                            std::to_string(error_code) + " " + __FILE__ + ":" + std::to_string(__LINE__));
  }
  allocation->physical = nullptr;
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
  check_free_error();
  if (!g_python_malloc_callback) {
    throw std::runtime_error("my_malloc ERROR: g_python_malloc_callback not set." +
                            std::string(" ") + __FILE__ + ":" + std::to_string(__LINE__));
  }
  ensure_context(device);

  // Align the first allocation and reserve its virtual address.

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
    throw std::runtime_error("aclrtMemGetAllocationGranularity failed with acl error code: " +
                            std::to_string(error_code) + " " + __FILE__ + ":" + std::to_string(__LINE__));
  }
  size_t alignedSize = ((size + granularity - 1) / granularity) * granularity;
  void *d_mem;
  error_code = aclrtReserveMemAddress(&d_mem, alignedSize, 0, nullptr, 0);
  if (error_code != 0) {
    if (error_code == ACL_ERROR_RT_MEMORY_ALLOCATION) {
      throw std::runtime_error("aclrtReserveMemAddress failed with acl error code: " + 
                              std::to_string(error_code) + "(OOM: Out of Memory, allocation failed) " + 
                              __FILE__ + ":" + std::to_string(__LINE__));
    } else {
      throw std::runtime_error("aclrtReserveMemAddress failed with acl error code: " +
                              std::to_string(error_code) + " " + __FILE__ + ":" + std::to_string(__LINE__));
    }
  }
  // Allocate the opaque mapping/physical-ownership record.
  CaMemAllocation* p_memHandle = new CaMemAllocation();

  // Finish physical allocation before publishing the handle to Python. A
  // failed initial map must not leave a stale entry in pointer_to_data.
  try {
    create_and_map(device, alignedSize, d_mem, p_memHandle);
  } catch (...) {
    // create_and_map normally releases a failed map's physical allocation.
    // If the driver also fails cleanup, retain the reservation rather than
    // releasing an address whose ownership is unknown.
    if (p_memHandle->physical == nullptr) {
      aclrtReleaseMemAddress(d_mem);
      delete p_memHandle;
    }
    throw;
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
    unmap_and_release(device, alignedSize, d_mem, p_memHandle);
    aclrtReleaseMemAddress(d_mem);
    delete p_memHandle;
    return nullptr;
  }

  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  return (void*)d_mem;
}

void my_free_impl(void* ptr, ssize_t size, int device, aclrtStream stream) {
  // get memory handle from the pointer
  if (!g_python_free_callback) {
    throw std::runtime_error("aclrtDrvMemHandle ERROR: g_python_malloc_callback not set." +
                            std::string(" ") + __FILE__ + ":" + std::to_string(__LINE__));
  }

  // Acquire GIL (not in stable ABI officially, but often works)
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* py_ptr =
      PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));

  PyObject* py_result =
      PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);
  Py_DECREF(py_ptr);

  if (!py_result || !PyTuple_Check(py_result) || PyTuple_Size(py_result) != 4) {
    if (!PyErr_Occurred()) {
      PyErr_SetString(PyExc_TypeError, "Expected a tuple of size 4");
    }
    Py_XDECREF(py_result);
    PyErr_Print();
    PyGILState_Release(gstate);
    throw std::runtime_error("CaMem free callback did not return a valid handle");
  }

  unsigned long long recv_device, recv_size;
  unsigned long long recv_d_mem, recv_p_memHandle;
  // Unpack the tuple into four C integers
  if (!PyArg_ParseTuple(py_result, "KKKK", &recv_device, &recv_size,
                        &recv_d_mem, &recv_p_memHandle)) {
    Py_DECREF(py_result);
    PyErr_Print();
    PyGILState_Release(gstate);
    throw std::runtime_error("CaMem free callback returned a malformed handle");
  }

  Py_DECREF(py_result);
  PyGILState_Release(gstate);

  // The callback holds the actual device and aligned reservation size.
  void *d_mem = (void*)recv_d_mem;
  CaMemAllocation* p_memHandle = (CaMemAllocation*)recv_p_memHandle;
  unmap_and_release(recv_device, recv_size, d_mem, p_memHandle);

  // free address and the handle
  aclError error_code = aclrtReleaseMemAddress(d_mem);
  if (error_code != 0) {
    throw std::runtime_error("aclrtReleaseMemAddress failed with acl error code: " +
                            std::to_string(error_code) + " " + __FILE__ + ":" + std::to_string(__LINE__));
  }
  if (g_python_free_complete_callback) {
    // Keep Python ownership until all driver operations succeed. If this
    // notification fails, retain the empty native record and poison the
    // worker; a stale Python handle must never be used for restoration.
    gstate = PyGILState_Ensure();
    py_ptr = PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));
    py_result = PyObject_CallFunctionObjArgs(g_python_free_complete_callback, py_ptr, NULL);
    Py_DECREF(py_ptr);
    if (!py_result) {
      PyErr_Print();
      PyGILState_Release(gstate);
      throw std::runtime_error("CaMem free completion callback failed");
    }
    Py_DECREF(py_result);
    PyGILState_Release(gstate);
  }
  delete p_memHandle;
}

__attribute__ ((visibility("default"))) void my_free(void* ptr, ssize_t size, int device, aclrtStream stream) noexcept {
  try {
    my_free_impl(ptr, size, device, stream);
  } catch (const std::exception& error) {
    g_free_failed.store(true);
    std::cerr << "CaMem my_free failed; ownership retained until worker restart: " << error.what() << std::endl;
  } catch (...) {
    g_free_failed.store(true);
    std::cerr << "CaMem my_free failed with unknown error; restart the worker" << std::endl;
  }
}

// ---------------------------------------------------------------------------
// Python extension boilerplate:

// Python-exposed function: init_module(python_malloc, python_free)
static PyObject* py_init_module(PyObject* self, PyObject* args) {
  PyObject* malloc_callback = nullptr;
  PyObject* free_callback = nullptr;
  PyObject* free_complete_callback = Py_None;

  if (!PyArg_ParseTuple(args, "OO|O", &malloc_callback, &free_callback, &free_complete_callback)) {
    return nullptr;
  }

  if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback) ||
      (free_complete_callback != Py_None && !PyCallable_Check(free_complete_callback))) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }

  // Save the Python callables
  // This module does not handle GC of these objects, so they must be kept alive
  // outside of this module.
  g_python_malloc_callback = malloc_callback;
  g_python_free_callback = free_callback;
  g_python_free_complete_callback = free_complete_callback == Py_None ? nullptr : free_complete_callback;

  Py_RETURN_NONE;
}

static PyObject* python_check_free_error(PyObject* self, PyObject* args) {
  try {
    check_free_error();
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }
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
  CaMemAllocation* p_memHandle = (CaMemAllocation*)recv_p_memHandle;

  try {
    unmap_and_release(recv_device, recv_size, d_mem_ptr, p_memHandle);
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }

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
  CaMemAllocation* p_memHandle = (CaMemAllocation*)recv_p_memHandle;

  try {
    create_and_map(recv_device, recv_size, d_mem_ptr, p_memHandle);
  } catch (const std::exception& error) {
    PyErr_SetString(PyExc_RuntimeError, error.what());
    return nullptr;
  }

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"python_check_free_error", (PyCFunction)python_check_free_error, METH_NOARGS,
     "Reject further use after a native free callback failure; restart is required."},
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
