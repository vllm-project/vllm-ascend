#include <vector>
#include <string>
#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <acl/acl.h>
#include <iostream>
#include <torch/extension.h>
#include <mem.h>
#include <cstring>
#include <dev.h>

#define CHECK_ACLRT(call) do {\
    aclError status = call;\
    if (status != ACL_SUCCESS) {\
        fprintf(stderr, "Error: %s failed with status %d\n", #call, status);\
    }\
} while (0)

#define CHECK_RTS(call) do {\
    aclError status = call;\
    if (status != 0) {\
        fprintf(stderr, "Error: %s failed with status %d\n", #call, status);\
    }\
} while (0)


int64_t initialize_acl() {
    CHECK_ACLRT(aclInit(""));
    printf("ACL intializing in WORM C++ code: [[success]]");
    return 0;
}

int64_t hello_world() {
    std::cout << "Hello from C++ extension!" << std::endl;
    return 42;
}


typedef struct {
    char reserved[100];  // Fixed 64-byte reserved buffer
} rtIpcMemHandle_t;


static inline size_t elem_size_from_str(const std::string& dtype_str) {
    if (dtype_str == "float16")   return 2;
    if (dtype_str == "bfloat16")  return 2;
    if (dtype_str == "float32")   return 4;
    if (dtype_str == "float64")   return 8;
    if (dtype_str == "int8")      return 1;
    if (dtype_str == "uint8")     return 1;
    if (dtype_str == "int16")     return 2;
    if (dtype_str == "int32")     return 4;
    if (dtype_str == "int64")     return 8;
    if (dtype_str == "bool")      return 1;
    if (dtype_str == "complex64") return 8;
    if (dtype_str == "complex128")return 16;
    throw std::runtime_error("Unsupported dtype string: " + dtype_str);
}

static inline c10::ScalarType scalar_from_str(const std::string& dtype_str) {
    if (dtype_str == "float16")    return torch::kFloat16;
    if (dtype_str == "bfloat16")   return torch::kBFloat16;
    if (dtype_str == "float32")    return torch::kFloat32;
    if (dtype_str == "float64")    return torch::kFloat64;
    if (dtype_str == "int8")       return torch::kInt8;
    if (dtype_str == "uint8")      return torch::kUInt8;
    if (dtype_str == "int16")      return torch::kInt16;
    if (dtype_str == "int32")      return torch::kInt32;
    if (dtype_str == "int64")      return torch::kInt64;
    if (dtype_str == "bool")       return torch::kBool;
    if (dtype_str == "complex64")  return torch::kComplexFloat;
    if (dtype_str == "complex128") return torch::kComplexDouble;

    throw std::runtime_error("Unsupported dtype string: " + dtype_str);
}

size_t calculate_size(const std::vector<int64_t>& shape, const std::string& dtype_str) {
    size_t numel = 1;
    for (auto s : shape) numel *= static_cast<size_t>(s);
    return numel * elem_size_from_str(dtype_str);
}

int64_t rt_export_tensor_ipc_handle_c(void* ptr, size_t size, rtIpcMemHandle_t* handle) {
    aclError status = rtIpcSetMemoryName(ptr, size, handle->reserved, sizeof(rtIpcMemHandle_t));
    if (status != ACL_SUCCESS) {
        fprintf(stderr, "Error: rtIpcSetMemoryName failed with status %d\n", status);
        return -1;
    }
    return 0;
}

std::tuple<int64_t, at::Tensor> rt_export_tensor_ipc_handle(int64_t ptr, std::vector<int64_t> shape, const std::string& dtype_str) {
    size_t size_bytes = calculate_size(shape, dtype_str);

    rtIpcMemHandle_t handle;
    int64_t ret = rt_export_tensor_ipc_handle_c(reinterpret_cast<void*>(ptr), size_bytes, &handle);
    // std::cout << "handle string: " << handle.reserved << " " << ret << std::endl;

    auto tensor = torch::from_blob(
        &handle, 
        {sizeof(rtIpcMemHandle_t)}, 
        torch::TensorOptions().dtype(torch::kUInt8)
    ).clone();
//     torch::Tensor tensor = torch::tensor(std::vector<char>(handle.reserved.begin(), handle.reserved.end()), torch::dtype(torch::kByte));    

    return {ret, tensor};
}


int64_t rt_set_ipc_mem_pid(at::Tensor handle_tensor, std::vector<int64_t> pid_list) {
    if (!handle_tensor.device().is_cpu())
        throw std::runtime_error("Handle tensor must be on CPU.");
    if (handle_tensor.dtype() != torch::kByte)
        throw std::runtime_error("Handle tensor must be of type uint8.");
    if (handle_tensor.numel() < static_cast<int64_t>(sizeof(rtIpcMemHandle_t)))
        throw std::runtime_error("Invalid IPC handle tensor size.");
    if (pid_list.empty())
        throw std::runtime_error("PID list cannot be empty.");

    // Copy tensor to handle
    rtIpcMemHandle_t handle;
    std::memcpy(&handle, handle_tensor.data_ptr(), sizeof(rtIpcMemHandle_t));
    // std::cout << "handle string: " << handle.reserved << std::endl;

    // Convert to int32_t
    std::vector<int32_t> converted_pid_list;
    for (int64_t pid : pid_list) {
        converted_pid_list.push_back(static_cast<int32_t>(pid));
    }
    // std::cout << "Setting PIDs: " << converted_pid_list << std::endl;

    // Call rtSetIpcMemPid
    aclError status = rtSetIpcMemPid(handle.reserved, converted_pid_list.data(), converted_pid_list.size());
    if (status != ACL_SUCCESS) {
        std::cerr << "[torch_interface.cpp] rtSetIpcMemPid failed with status " << status << " pid_list:" << pid_list << std::endl;
        return -1;
    }

    return 0;
}

int64_t rt_get_tgid() {
    int32_t tgid = 0;
    aclError ret = aclrtDeviceGetBareTgid(&tgid);
    if (ret != ACL_ERROR_NONE) {
        std::cerr << "Failed to get bare TGID, error code: " << ret << std::endl;
        return -1;
    }
    // return tgid;
    return {static_cast<int64_t>(tgid)};
}


int64_t rt_open_ipc_handle_c(void** ptr, rtIpcMemHandle_t* handle) {
    aclError status = rtIpcOpenMemory(ptr, handle->reserved);
    if (status != ACL_SUCCESS) {
        fprintf(stderr, "Error: rtIpcOpenMemory failed with status %d\n", status);
        std::cout<<"Error mesaage in rt_open_ipc_handle_c() "<< aclGetRecentErrMsg() <<std::endl;
        return -1;
    }
    return 0;
}

std::tuple<int64_t, int64_t> rt_open_ipc_handle(at::Tensor handle_tensor) {
    if (!handle_tensor.device().is_cpu()) 
        throw std::runtime_error("Handle tensor must be on CPU for safe transfer.");
    if (handle_tensor.dtype() != torch::kByte)
        throw std::runtime_error("Handle tensor must be of type kByte (uint8).");
    if (handle_tensor.numel() < static_cast<int64_t>(sizeof(rtIpcMemHandle_t))) 
        throw std::runtime_error("Invalid IPC handle tensor size.");

    // init_npu(0);
    rtIpcMemHandle_t handle;
    std::memcpy(&handle, handle_tensor.data_ptr(), sizeof(rtIpcMemHandle_t));
    // std::cout << "\nhandle string: " << handle.reserved << std::endl;
    
    void* ptr;
    int64_t ret = rt_open_ipc_handle_c(&ptr, &handle);

    return {ret, reinterpret_cast<int64_t>(ptr)};
}

std::tuple<int64_t, int64_t> allocate_ipc_safe_tensor(std::vector<int64_t> shape, std::string dtype_str) {
    size_t numel = 1;
    for (auto s : shape) numel *= static_cast<size_t>(s);
    size_t size_bytes = numel * elem_size_from_str(dtype_str);

    int return_status = 0;
    void* ptr = nullptr;

    aclError status = aclrtMalloc(&ptr, size_bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (status != ACL_SUCCESS) {
        fprintf(stderr, "Error: aclrtMalloc failed with status %d\n", status);
        std::cout << "Error mesaage in allocate_ipc_safe_tensor() " << aclGetRecentErrMsg() << std::endl;
        return_status = -1;
    }
    return {return_status, reinterpret_cast<int64_t>(ptr)};
}

std::int64_t free_ipc_safe_tensor(std::int64_t ptr_int) {
    void* ptr = reinterpret_cast<void*>(ptr_int);
    // If you can, ensure no work is using this ptr; do stream/device sync outside.
    aclError status = aclrtFree(ptr);
    if (status != ACL_SUCCESS) {
        fprintf(stderr, "Error: aclrtFree failed with status %d\n", status);
        std::cout << "Error message in free_ipc_safe_tensor(): "
                  << aclGetRecentErrMsg() << std::endl;
        return -1;
    }
    return 0;
}


TORCH_LIBRARY(tensor_ipc_utils, m) {
    m.def("hello_world() -> int");
    m.impl("hello_world", []() { return hello_world();});

    m.def("rt_export_tensor_ipc_handle(int ptr ,int[] shape, str dtype) -> (int, Tensor)");
    m.impl("rt_export_tensor_ipc_handle", rt_export_tensor_ipc_handle);

    m.def("rt_open_ipc_handle(Tensor handle) -> (int, int)");
    m.impl("rt_open_ipc_handle", rt_open_ipc_handle);

    m.def("rt_set_ipc_mem_pid(Tensor handle, int[] pid_list) -> int");
    m.impl("rt_set_ipc_mem_pid", rt_set_ipc_mem_pid);

    m.def("rt_get_tgid() -> int");
    m.impl("rt_get_tgid", rt_get_tgid);

    m.def("initialize_acl() -> int");
    m.impl("initialize_acl", initialize_acl);

    m.def("allocate_ipc_safe_tensor(int[] shape, str dtype) -> (int, int)");
    m.impl("allocate_ipc_safe_tensor", allocate_ipc_safe_tensor);

    m.def("free_ipc_safe_tensor(int ptr_int) -> int");
    m.impl("free_ipc_safe_tensor", free_ipc_safe_tensor);
}


