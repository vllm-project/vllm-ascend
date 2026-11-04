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

#include <algorithm>
#include <atomic>
#include <array>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "camem_utils.hpp"

extern "C" {

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "acl/acl.h"

static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

namespace {

using Byte = uint8_t;

constexpr uint64_t kMB = 1024ULL * 1024ULL;
constexpr int kDefaultWaitTimeoutMs = 10000;
constexpr int kDefaultPollIntervalUs = 1000;
constexpr int kMacroWaitTimeoutMs = 10000;
constexpr std::chrono::milliseconds kHeartbeatInterval(500);

int64_t monotonicNowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

int pollIntervalUs() {
    static const int interval_us = []() {
        const char* raw = std::getenv("MDAEMON_POLL_INTERVAL_US");
        if (!raw || raw[0] == '\0') {
            return kDefaultPollIntervalUs;
        }
        char* end = nullptr;
        errno = 0;
        long value = std::strtol(raw, &end, 10);
        if (errno != 0 || end == raw || (end && *end != '\0') || value <= 0) {
            return kDefaultPollIntervalUs;
        }
        return static_cast<int>(value);
    }();
    return interval_us;
}

const std::array<uint64_t, 4> kSupportedGranularities = {
    16ULL * kMB, 8ULL * kMB, 4ULL * kMB, 2ULL * kMB
};

/* allocation mapping structure
d_mem_base -> Allocated by aclrtMalloc, with reserved_size
    | requested_size (<= reserved_size)
    |---> slot 0: virt_addr, granularity, device_id, state, handle
    |---> slot 1: virt_addr, granularity, device_id, state, handle
    ...
*/

struct ImportedHandle {
    uint64_t shareable_handle{std::numeric_limits<uint64_t>::max()};
    uint64_t granularity{0};
    int32_t device_id{-1};
    aclrtDrvMemHandle raw{};
    bool valid{false};
};

enum class SlotState : uint8_t {
    UNMAPPED = 0,
    NULL_MAPPED = 1,
    REAL_MAPPED = 2,
};

struct MappingSlot {
    void* virt_addr{nullptr};
    uint64_t granularity{0};
    int32_t device_id{-1};
    SlotState state{SlotState::UNMAPPED};
    ImportedHandle handle{};
};

struct AllocationRecord {
    void* base{nullptr};
    void* raw_base{nullptr};
    size_t requested_size{0};
    size_t reserved_size{0};
    int32_t device_id{-1};
    bool is_kvcache{false};
    uint64_t kvcache_granularity{0};
    size_t mapped_weight_bytes{0};
    std::vector<MappingSlot> slots;
};

struct RuntimeState {
    std::mutex mutex;
    bool initialized{false};
    std::atomic<bool> heartbeat_running{false};
    std::thread heartbeat_thread;
    int macro_fd{-1};
    int message_fd{-1};
    sem_t* message_sem{nullptr};
    ModelMacroSpace* macro_space{nullptr};
    ModelMessageSpace* message_space{nullptr};
    std::string model_shm_name;
    std::string message_sem_name;
    int32_t device_id{-1};
    int32_t model_npuid{kInvalidModelHandle};
    int32_t model_osid{kInvalidModelHandle};
    std::unordered_map<void*, AllocationRecord> allocations;
    std::unordered_map<uint64_t, ImportedHandle> null_pages;
};

RuntimeState g_state;

std::string buildMessageSemName(const std::string& shm_name) {
    return shm_name + "-sem";
}

void lockMessageSemaphore() {
    if (!g_state.message_sem) {
        throw std::runtime_error("message semaphore is not initialized");
    }
    while (::sem_wait(g_state.message_sem) == -1) {
        if (errno == EINTR) {
            continue;
        }
        throw std::runtime_error("sem_wait failed for message semaphore");
    }
}

void unlockMessageSemaphore() {
    if (!g_state.message_sem) {
        return;
    }
    if (::sem_post(g_state.message_sem) == -1) {
        throw std::runtime_error("sem_post failed for message semaphore");
    }
}

struct MessageSemaphoreGuard {
    MessageSemaphoreGuard() {
        lockMessageSemaphore();
    }
    ~MessageSemaphoreGuard() {
        if (g_state.message_sem) {
            (void)::sem_post(g_state.message_sem);
        }
    }
};

[[noreturn]] void throwAcl(aclError code, const char* op, const char* file, int line) {
    throw std::runtime_error(std::string(op) + " failed with acl error code: " +
                             std::to_string(code) + " at " + file + ":" +
                             std::to_string(line));
}

void checkAcl(aclError code, const char* op, const char* file, int line) {
    if (code != 0) {
        throwAcl(code, op, file, line);
    }
}

#define CHECK_ACL(op, expr) checkAcl((expr), (op), __FILE__, __LINE__)

bool daemonDebugLogsEnabled() {
    const char* flag = std::getenv("MDAEMON_DEBUG_LOG");
    return flag && std::string(flag) == "1";
}

void ensureContext(int32_t device) {
    CHECK_ACL("aclrtSetDevice", aclrtSetDevice(device));
    aclrtContext context;
    CHECK_ACL("aclrtGetCurrentContext", aclrtGetCurrentContext(&context));
    if (!context) {
        CHECK_ACL("aclrtCreateContext", aclrtCreateContext(&context, device));
        CHECK_ACL("aclrtSetCurrentContext", aclrtSetCurrentContext(context));
    }
}

uint64_t alignUp(uint64_t value, uint64_t align) {
    if (align == 0) {
        throw std::invalid_argument("align must be positive");
    }
    return ((value + align - 1) / align) * align;
}

uint64_t alignDown(uint64_t value, uint64_t align) {
    if (align == 0) {
        throw std::invalid_argument("align must be positive");
    }
    return (value / align) * align;
}

uintptr_t alignUpPtr(uintptr_t value, uint64_t align) {
    return static_cast<uintptr_t>(alignUp(static_cast<uint64_t>(value), align));
}

int readIntEnv(const char* name, int default_value) {
    const char* raw = std::getenv(name);
    if (!raw || raw[0] == '\0') {
        return default_value;
    }
    try {
        return std::stoi(raw);
    } catch (...) {
        return default_value;
    }
}

std::vector<int32_t> parseVisibleDeviceList() {
    const char* raw = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
    if (!raw || raw[0] == '\0') {
        raw = std::getenv("ASCEND_VISIBLE_DEVICES");
    }
    if (!raw || raw[0] == '\0') {
        raw = std::getenv("CUDA_VISIBLE_DEVICES");
    }
    if (!raw || raw[0] == '\0') {
        return {};
    }

    std::vector<int32_t> ids;
    const std::string visible(raw);
    size_t start = 0;
    while (start < visible.size()) {
        size_t end = visible.find(',', start);
        if (end == std::string::npos) {
            end = visible.size();
        }
        std::string token = visible.substr(start, end - start);
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) {
                        return std::isspace(c) != 0;
                    }),
                    token.end());
        if (!token.empty()) {
            ids.push_back(static_cast<int32_t>(std::stoi(token)));
        }
        start = end + 1;
    }
    return ids;
}

int32_t localToCanonicalDeviceId(int32_t local_device_id) {
    if (local_device_id < 0) {
        return local_device_id;
    }
    const auto ids = parseVisibleDeviceList();
    if (ids.empty() || static_cast<size_t>(local_device_id) >= ids.size()) {
        return local_device_id;
    }
    return ids[local_device_id];
}

int32_t canonicalToLocalDeviceId(int32_t canonical_device_id) {
    if (canonical_device_id < 0) {
        return canonical_device_id;
    }
    const auto ids = parseVisibleDeviceList();
    if (ids.empty()) {
        return canonical_device_id;
    }
    for (size_t i = 0; i < ids.size(); ++i) {
        if (ids[i] == canonical_device_id) {
            return static_cast<int32_t>(i);
        }
    }
    return canonical_device_id;
}

uint64_t getKvcacheGranularity() {
    const int mb = readIntEnv("MDAEMON_KV_GRAN_MB", 4);
    const uint64_t bytes = static_cast<uint64_t>(mb) * kMB;
    for (uint64_t g : kSupportedGranularities) {
        if (g == bytes) {
            return bytes;
        }
    }
    return 4ULL * kMB;
}

void callPythonMallocCallback(int device, uint64_t aligned_size, void* d_mem) {
    if (!g_python_malloc_callback) {
        return;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* arg_tuple = PyTuple_New(4);
    if (!arg_tuple) {
        PyGILState_Release(gstate);
        return;
    }
    PyTuple_SetItem(arg_tuple, 0, PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(device)));
    PyTuple_SetItem(arg_tuple, 1, PyLong_FromUnsignedLongLong(static_cast<unsigned long long>(aligned_size)));
    PyTuple_SetItem(arg_tuple, 2, PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(d_mem)));
    PyTuple_SetItem(arg_tuple, 3, PyLong_FromUnsignedLongLong(0ULL));
    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_malloc_callback, arg_tuple, NULL);
    Py_DECREF(arg_tuple);
    if (!py_result) {
        PyErr_Print();
    } else {
        Py_DECREF(py_result);
    }
    PyGILState_Release(gstate);
}

void callPythonFreeCallback(void* ptr) {
    if (!g_python_free_callback) {
        return;
    }
    PyGILState_STATE gstate = PyGILState_Ensure();
    PyObject* py_ptr = PyLong_FromUnsignedLongLong(reinterpret_cast<unsigned long long>(ptr));
    PyObject* py_result = PyObject_CallFunctionObjArgs(g_python_free_callback, py_ptr, NULL);
    Py_DECREF(py_ptr);
    if (!py_result) {
        PyErr_Print();
    } else {
        Py_DECREF(py_result);
    }
    PyGILState_Release(gstate);
}

void clearRequestList(ModelMessageSpace& space) {
    for (auto& req : space.handle_request_list) {
        if (req.request_num == 0 && req.granularity == 0 && req.device_id == -1) {
            // already cleared, skip to save some time
            break;
        }
        req.request_num = 0;
        req.granularity = 0;
        req.device_id = -1;
    }
}

void clearHandleInfoList(ModelMessageSpace& space) {
    for (auto& info : space.handle_info_list) {
        if (info.shareable_handle == std::numeric_limits<uint64_t>::max() &&
            info.granularity == 0 &&
            info.device_id == -1) {
            // already cleared, skip to save some time
            break;
        }
        info.shareable_handle = std::numeric_limits<uint64_t>::max();
        info.granularity = 0;
        info.device_id = -1;
    }
    space.offset_st = 0;
    space.offset_ed = 0;
}

void syncMessageToShm() {
    if (!g_state.message_space) {
        throw std::runtime_error("message space is not initialized");
    }
    std::atomic_thread_fence(std::memory_order_release);
}

void syncMessageFromShm() {
    if (!g_state.message_space) {
        throw std::runtime_error("message space is not initialized");
    }
    std::atomic_thread_fence(std::memory_order_acquire);
}

void touchHeartbeat() {
    if (!g_state.message_space || !g_state.message_sem) {
        return;
    }
    MessageSemaphoreGuard guard;
    syncMessageFromShm();
    g_state.message_space->heartbeat_ns = monotonicNowNs();
    syncMessageToShm();
}

void heartbeatLoop() {
    while (g_state.heartbeat_running.load(std::memory_order_relaxed)) {
        try {
            touchHeartbeat();
        } catch (...) {
            // Keep heartbeat best-effort and non-fatal.
        }
        std::this_thread::sleep_for(kHeartbeatInterval);
    }
}

void startHeartbeatThread() {
    if (g_state.heartbeat_running.exchange(true, std::memory_order_acq_rel)) {
        return;
    }
    g_state.heartbeat_thread = std::thread(heartbeatLoop);
}

void stopHeartbeatThread() {
    if (!g_state.heartbeat_running.exchange(false, std::memory_order_acq_rel)) {
        return;
    }
    if (g_state.heartbeat_thread.joinable()) {
        g_state.heartbeat_thread.join();
    }
}

bool waitForMessageState(MessageState target, int timeout_ms) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        bool reached = false;
        {
            MessageSemaphoreGuard guard;
            syncMessageFromShm();
            reached = (g_state.message_space->state == target);
        }
        if (reached) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(pollIntervalUs()));
    }
    return false;
}

ImportedHandle importShareableHandle(const handle_granularity& info) {
    ImportedHandle imported;
    imported.shareable_handle = info.shareable_handle;
    imported.granularity = info.granularity;
    imported.device_id = info.device_id;
    const int32_t local_device_id = canonicalToLocalDeviceId(info.device_id);
    aclrtDrvMemHandle raw{};
    CHECK_ACL("aclrtMemImportFromShareableHandle",
              aclrtMemImportFromShareableHandle(info.shareable_handle, local_device_id, &raw));
    imported.raw = raw;
    imported.valid = true;
    return imported;
}

void releaseImportedHandle(ImportedHandle& imported) {
    if (!imported.valid) {
        return;
    }
    CHECK_ACL("aclrtFreePhysical", aclrtFreePhysical(imported.raw));
    imported = ImportedHandle{};
}

void mapSlot(MappingSlot& slot, const ImportedHandle& imported, SlotState state) {
    if (!imported.valid) {
        throw std::runtime_error("cannot map invalid imported handle");
    }
    const uintptr_t vaddr = reinterpret_cast<uintptr_t>(slot.virt_addr);
    if (slot.granularity == 0 || (vaddr % slot.granularity) != 0) {
        throw std::runtime_error("virtual address is not aligned to mapping granularity");
    }
    CHECK_ACL("aclrtMapMem", aclrtMapMem(slot.virt_addr, slot.granularity, 0, imported.raw, 0));
    slot.handle = imported;
    slot.state = state;
}

void unmapSlot(MappingSlot& slot) {
    if (slot.state == SlotState::UNMAPPED) {
        return;
    }
    const SlotState old_state = slot.state;
    CHECK_ACL("aclrtUnmapMem", aclrtUnmapMem(slot.virt_addr));
    if (old_state == SlotState::REAL_MAPPED) {
        releaseImportedHandle(slot.handle);
    }
    slot.state = SlotState::UNMAPPED;
    slot.handle = ImportedHandle{};
}

std::vector<request_granularity> buildLargeFirstRequests(uint64_t bytes, int32_t device_id) {
    std::vector<request_granularity> requests;
    uint64_t remaining = bytes;
    for (uint64_t granularity : kSupportedGranularities) {
        const uint64_t num = remaining / granularity;
        if (num > 0) {
            requests.push_back({num, granularity, device_id});
            remaining -= num * granularity;
        }
    }
    if (remaining > 0) {
        const uint64_t smallest = kSupportedGranularities.back();
        requests.push_back({1, smallest, device_id});
    }
    return requests;
}

std::vector<request_granularity> buildSingleGranularityRequests(uint64_t bytes,
                                                                uint64_t granularity,
                                                                int32_t device_id) {
    const uint64_t aligned = alignUp(bytes, granularity);
    const uint64_t num = aligned / granularity;
    if (num == 0) {
        return {};
    }
    return {{num, granularity, device_id}};
}

std::vector<handle_granularity> requestHandlesFromDaemon(const std::vector<request_granularity>& requests,
                                                         int timeout_ms) {
    if (requests.empty()) {
        return {};
    }
    if (!g_state.message_space) {
        throw std::runtime_error("init_space must be called before requesting handles");
    }
    if (requests.size() > g_state.message_space->handle_request_list.size()) {
        throw std::runtime_error("too many handle requests for message space");
    }

    ModelMessageSpace& space = *g_state.message_space;
    {
        MessageSemaphoreGuard guard;
        clearRequestList(space);
        clearHandleInfoList(space);

        for (size_t i = 0; i < requests.size(); ++i) {
            space.handle_request_list[i] = requests[i];
        }

        space.state = MessageState::REQUEST_GET_HANDLES;
        syncMessageToShm();
    }

    if (!waitForMessageState(MessageState::HANDLES_READY, timeout_ms)) {
        throw std::runtime_error("timed out waiting for HANDLES_READY");
    }

    std::vector<handle_granularity> handles;
    {
        MessageSemaphoreGuard guard;
        syncMessageFromShm();
        const int32_t start = std::max<int32_t>(0, space.offset_st);
        const int32_t end = std::min<int32_t>(space.offset_ed,
                                              static_cast<int32_t>(space.handle_info_list.size()));
        for (int32_t i = start; i < end; ++i) {
            const handle_granularity& info = space.handle_info_list[i];
            if (info.granularity == 0 || info.device_id < 0 ||
                info.shareable_handle == std::numeric_limits<uint64_t>::max()) {
                continue;
            }
            handles.push_back(info);
        }

        clearRequestList(space);
        clearHandleInfoList(space);
        space.state = MessageState::NONE;
        syncMessageToShm();
    }

    return handles;
}

void returnHandlesToDaemon(const std::vector<handle_granularity>& handles,
                           int timeout_ms = kDefaultWaitTimeoutMs) {
    if (handles.empty()) {
        return;
    }
    if (!g_state.message_space) {
        throw std::runtime_error("init_space must be called before returning handles");
    }
    if (handles.size() > g_state.message_space->handle_info_list.size()) {
        throw std::runtime_error("too many handles to return for message space");
    }

    ModelMessageSpace& space = *g_state.message_space;
    {
        MessageSemaphoreGuard guard;
        clearRequestList(space);
        clearHandleInfoList(space);

        for (size_t i = 0; i < handles.size(); ++i) {
            space.handle_info_list[i] = handles[i];
        }
        space.offset_st = 0;
        space.offset_ed = static_cast<int32_t>(handles.size());
        space.state = MessageState::REQUEST_RETURN_HANDLES;
        syncMessageToShm();
    }

    if (!waitForMessageState(MessageState::HANDLES_RETURNED, timeout_ms)) {
        throw std::runtime_error("timed out waiting for HANDLES_RETURNED");
    }

    {
        MessageSemaphoreGuard guard;
        clearRequestList(space);
        clearHandleInfoList(space);
        space.state = MessageState::NONE;
        syncMessageToShm();
    }
}

struct ReservedAddress {
    void* raw_base{nullptr};
    void* aligned_base{nullptr};
    size_t raw_reserved_size{0};
    size_t usable_size{0};
};

ReservedAddress reserveVirtualAddress(size_t requested_usable_size, uint64_t align_granularity) {
    const uint64_t headroom = align_granularity - 1;
    const uint64_t raw_reserved =
        alignUp(static_cast<uint64_t>(requested_usable_size) + headroom, align_granularity);
    void* raw_base = nullptr;
    CHECK_ACL("aclrtReserveMemAddress",
              aclrtReserveMemAddress(&raw_base, static_cast<size_t>(raw_reserved), 0, nullptr, 0));

    const uintptr_t raw_addr = reinterpret_cast<uintptr_t>(raw_base);
    const uintptr_t aligned_addr = alignUpPtr(raw_addr, align_granularity);
    const size_t offset = static_cast<size_t>(aligned_addr - raw_addr);
    const size_t usable = static_cast<size_t>(raw_reserved) - offset;
    if (usable < requested_usable_size) {
        CHECK_ACL("aclrtReleaseMemAddress", aclrtReleaseMemAddress(raw_base));
        throw std::runtime_error("reserved virtual address usable range is smaller than requested");
    }

    return ReservedAddress{
        raw_base,
        reinterpret_cast<void*>(aligned_addr),
        static_cast<size_t>(raw_reserved),
        usable,
    };
}

AllocationRecord& findAllocationOrThrow(void* ptr) {
    auto it = g_state.allocations.find(ptr);
    if (it == g_state.allocations.end()) {
        throw std::runtime_error("allocation not found for pointer");
    }
    return it->second;
}

size_t currentKvcacheMappedRealSlots(const AllocationRecord& record) {
    size_t count = 0;
    while (count < record.slots.size() && record.slots[count].state == SlotState::REAL_MAPPED) {
        ++count;
    }
    return count;
}

void registerToDaemonAndOpenMessageSpace(int32_t local_device_id,
                                         int32_t canonical_device_id) {
    ensureContext(local_device_id);

    int32_t npuid = kInvalidModelHandle;
    CHECK_ACL("aclrtDeviceGetBareTgid", aclrtDeviceGetBareTgid(&npuid));
    const int32_t osid = static_cast<int32_t>(::getpid());

    sem_t* sem = ::sem_open(MODEL_SEMAPHORE_NAME, O_CREAT, 0666, 1);
    if (sem == SEM_FAILED) {
        throw std::runtime_error("sem_open failed for model semaphore");
    }
    if (::sem_wait(sem) == -1) {
        ::sem_close(sem);
        throw std::runtime_error("sem_wait failed for model semaphore");
    }

    auto sem_guard = [&]() {
        ::sem_post(sem);
        ::sem_close(sem);
    };

    int macro_fd = ::shm_open(MODEL_REG_SHM, O_RDWR | O_CREAT, 0666);
    if (macro_fd == -1) {
        sem_guard();
        throw std::runtime_error("shm_open failed for MODEL_REG_SHM");
    }
    if (::ftruncate(macro_fd, sizeof(ModelMacroSpace)) == -1) {
        ::close(macro_fd);
        sem_guard();
        throw std::runtime_error("ftruncate failed for MODEL_REG_SHM");
    }

    void* macro_region = ::mmap(nullptr,
                                sizeof(ModelMacroSpace),
                                PROT_READ | PROT_WRITE,
                                MAP_SHARED,
                                macro_fd,
                                0);
    if (macro_region == MAP_FAILED) {
        ::close(macro_fd);
        sem_guard();
        throw std::runtime_error("mmap failed for MODEL_REG_SHM");
    }

    ModelMacroSpace* macro_space = static_cast<ModelMacroSpace*>(macro_region);
    if (isDefaultShmName(macro_space->getShmName())) {
        macro_space->current_model_npuid = npuid;
        macro_space->current_model_osid = osid;
        macro_space->setShmName(defaultShmName());
        ::msync(macro_space, sizeof(ModelMacroSpace), MS_SYNC);
    }
    else {
        ::munmap(macro_space, sizeof(ModelMacroSpace));
        ::close(macro_fd);
        sem_guard();
        throw std::runtime_error("ModelMacroSpace is dirty, please check");
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(kMacroWaitTimeoutMs);
    while (std::chrono::steady_clock::now() < deadline && isDefaultShmName(macro_space->getShmName())) {
        std::this_thread::sleep_for(std::chrono::microseconds(pollIntervalUs()));
        ::msync(macro_space, sizeof(ModelMacroSpace), MS_SYNC);
    }
    if (isDefaultShmName(macro_space->getShmName())) {
        ::munmap(macro_space, sizeof(ModelMacroSpace));
        ::close(macro_fd);
        sem_guard();
        throw std::runtime_error("timed out waiting daemon registration response");
    }

    const std::string model_shm_name(macro_space->getShmName());
    macro_space->reset();
    ::msync(macro_space, sizeof(ModelMacroSpace), MS_SYNC);

    ::munmap(macro_space, sizeof(ModelMacroSpace));
    ::close(macro_fd);
    sem_guard();

    const int message_fd = ::shm_open(model_shm_name.c_str(), O_RDWR, 0666);
    if (message_fd == -1) {
        throw std::runtime_error("shm_open failed for model message shm");
    }

    void* message_region = ::mmap(nullptr,
                                  sizeof(ModelMessageSpace),
                                  PROT_READ | PROT_WRITE,
                                  MAP_SHARED,
                                  message_fd,
                                  0);
    if (message_region == MAP_FAILED) {
        ::close(message_fd);
        throw std::runtime_error("mmap failed for model message shm");
    }

    g_state.device_id = canonical_device_id;
    g_state.model_npuid = npuid;
    g_state.model_osid = osid;
    g_state.model_shm_name = model_shm_name;
    g_state.message_sem_name = buildMessageSemName(model_shm_name);
    g_state.message_sem = ::sem_open(g_state.message_sem_name.c_str(), O_CREAT, 0666, 1);
    if (g_state.message_sem == SEM_FAILED) {
        g_state.message_sem = nullptr;
        ::munmap(message_region, sizeof(ModelMessageSpace));
        ::close(message_fd);
        throw std::runtime_error("sem_open failed for model message semaphore");
    }
    g_state.message_fd = message_fd;
    g_state.message_space = static_cast<ModelMessageSpace*>(message_region);

    // check if messasge state becomes REGISTERED by daemon within timeout, otherwise consider it as registration failure
    if (!waitForMessageState(MessageState::REGISTERED, kMacroWaitTimeoutMs)) {
        ::sem_close(g_state.message_sem);
        ::munmap(message_region, sizeof(ModelMessageSpace));
        ::close(message_fd);
        throw std::runtime_error("timed out waiting for message space registration");
    }
}

void initializeNullPages() {
    if (daemonDebugLogsEnabled()) {
        std::cout << "\033[34m[CamemAllocator] Initializing null pages for all supported granularities\033[0m\n";
        std::cout << "model shm name: \033[32m" << g_state.model_shm_name << "\033[0m\n";
        std::cout << "device id: " << g_state.device_id << ", support granularities: ";
        for (uint64_t granularity : kSupportedGranularities) {
            std::cout << granularity << " ";
        }
        std::cout << "\n";
    }
    std::vector<request_granularity> requests;
    requests.reserve(kSupportedGranularities.size());
    for (uint64_t granularity : kSupportedGranularities) {
        requests.push_back({1, granularity, g_state.device_id});
    }

    const std::vector<handle_granularity> granted =
        requestHandlesFromDaemon(requests, kDefaultWaitTimeoutMs);
    if (granted.size() != kSupportedGranularities.size()) {
        throw std::runtime_error("failed to initialize null pages for all granularities");
    }

    g_state.null_pages.clear();
    for (const auto& info : granted) {
        g_state.null_pages[info.granularity] = importShareableHandle(info);
    }

    for (uint64_t granularity : kSupportedGranularities) {
        if (g_state.null_pages.find(granularity) == g_state.null_pages.end()) {
            throw std::runtime_error("missing null page for granularity");
        }
    }
}

void maybeFinalizeState() {
    stopHeartbeatThread();
    if (g_state.message_sem) {
        ::sem_close(g_state.message_sem);
        g_state.message_sem = nullptr;
    }
    if (g_state.message_space) {
        g_state.message_space->state = MessageState::DONE;
        ::munmap(g_state.message_space, sizeof(ModelMessageSpace));
        g_state.message_space = nullptr;
    }
    if (g_state.message_fd != -1) {
        ::close(g_state.message_fd);
        g_state.message_fd = -1;
    }
    g_state.model_shm_name.clear();
    g_state.message_sem_name.clear();
    g_state.initialized = false;
}

void requireInitialized() {
    if (!g_state.initialized || !g_state.message_space) {
        throw std::runtime_error("allocator message space is not initialized; call python_init_space first");
    }
}

void mapWeightRange(AllocationRecord& record, size_t final_size, int timeout_ms) {
    if (record.is_kvcache) {
        throw std::runtime_error("mapWeightRange called for kvcache allocation");
    }
    const uint64_t min_gran = kSupportedGranularities.back();
    const size_t target_bytes = static_cast<size_t>(
        std::min<uint64_t>(alignUp(static_cast<uint64_t>(final_size), min_gran),
                           static_cast<uint64_t>(record.reserved_size)));
    if (target_bytes <= record.mapped_weight_bytes) {
        return;
    }

    const size_t delta_bytes = target_bytes - record.mapped_weight_bytes;
    const std::vector<request_granularity> requests =
        buildLargeFirstRequests(delta_bytes, g_state.device_id);
    std::vector<handle_granularity> granted = requestHandlesFromDaemon(requests, timeout_ms);

    std::vector<handle_granularity> accepted_for_return;
    Byte* base = static_cast<Byte*>(record.base);
    for (const auto& info : granted) {
        if (record.mapped_weight_bytes + info.granularity > record.reserved_size) {
            accepted_for_return.push_back(info);
            continue;
        }

        ImportedHandle imported = importShareableHandle(info);
        MappingSlot slot;
        slot.virt_addr = base + record.mapped_weight_bytes;
        slot.granularity = info.granularity;
        slot.device_id = info.device_id;
        mapSlot(slot, imported, SlotState::REAL_MAPPED);

        record.slots.push_back(slot);
        record.mapped_weight_bytes += info.granularity;

        if (record.mapped_weight_bytes >= target_bytes) {
            break;
        }
    }

    if (!accepted_for_return.empty()) {
        returnHandlesToDaemon(accepted_for_return, timeout_ms);
    }

    if (record.mapped_weight_bytes < target_bytes) {
        throw std::runtime_error("daemon returned insufficient handles to satisfy weight mapping");
    }
}

void unmapWeightAll(AllocationRecord& record, int timeout_ms) {
    std::vector<handle_granularity> to_return;
    for (auto& slot : record.slots) {
        if (slot.state == SlotState::REAL_MAPPED) {
            ImportedHandle old = slot.handle;
            unmapSlot(slot);
            to_return.push_back({old.shareable_handle, old.granularity, old.device_id});
        } else if (slot.state == SlotState::NULL_MAPPED) {
            unmapSlot(slot);
        }
    }
    record.slots.clear();
    record.mapped_weight_bytes = 0;
    returnHandlesToDaemon(to_return, timeout_ms);
}

void mapKvcacheUntil(AllocationRecord& record, size_t final_size, int timeout_ms) {
    if (!record.is_kvcache) {
        throw std::runtime_error("mapKvcacheUntil called for weight allocation");
    }
    const uint64_t gran = record.kvcache_granularity;
    const uint64_t ceil_bytes = alignUp(static_cast<uint64_t>(final_size), gran);
    const size_t target_slots = static_cast<size_t>(
        std::min<uint64_t>(ceil_bytes / gran, static_cast<uint64_t>(record.slots.size())));

    const size_t current_real = currentKvcacheMappedRealSlots(record);
    if (target_slots <= current_real) {
        return;
    }

    const size_t need_slots = target_slots - current_real;
    const uint64_t need_bytes = static_cast<uint64_t>(need_slots) * gran;
    const std::vector<request_granularity> requests =
        buildSingleGranularityRequests(need_bytes, gran, g_state.device_id);
    const std::vector<handle_granularity> granted = requestHandlesFromDaemon(requests, timeout_ms);

    size_t mapped = 0;
    for (const auto& info : granted) {
        if (info.granularity != gran) {
            throw std::runtime_error("kvcache mapping received unexpected handle granularity" 
                + std::to_string(info.granularity) + " expected " + std::to_string(gran));
        }
        if (current_real + mapped >= record.slots.size()) {
            break;
        }
        MappingSlot& slot = record.slots[current_real + mapped];
        if (slot.state != SlotState::NULL_MAPPED) {
            throw std::runtime_error("kvcache slot is not null-mapped before real mapping");
        }
        unmapSlot(slot);

        ImportedHandle imported = importShareableHandle(info);
        mapSlot(slot, imported, SlotState::REAL_MAPPED);
        ++mapped;
        if (mapped == need_slots) {
            break;
        }
    }

    if (mapped != need_slots) {
        throw std::runtime_error("daemon returned insufficient handles for kvcache mapping");
    }
}

void unmapKvcacheUntil(AllocationRecord& record, size_t final_size, int timeout_ms) {
    if (!record.is_kvcache) {
        throw std::runtime_error("unmapKvcacheUntil called for weight allocation");
    }
    const uint64_t gran = record.kvcache_granularity;
    const uint64_t ceil_bytes = alignUp(static_cast<uint64_t>(final_size), gran);
    const size_t keep_slots = static_cast<size_t>(
        std::min<uint64_t>(ceil_bytes / gran, static_cast<uint64_t>(record.slots.size())));

    size_t current_real = currentKvcacheMappedRealSlots(record);
    if (keep_slots >= current_real) {
        return;
    }

    std::vector<handle_granularity> to_return;
    for (size_t idx = current_real; idx > keep_slots; --idx) {
        MappingSlot& slot = record.slots[idx - 1];
        if (slot.state != SlotState::REAL_MAPPED) {
            throw std::runtime_error("kvcache real-mapped slots must be a prefix");
        }
        ImportedHandle old_real = slot.handle;
        unmapSlot(slot);

        auto null_it = g_state.null_pages.find(gran);
        if (null_it == g_state.null_pages.end()) {
            throw std::runtime_error("null page handle missing for kvcache granularity");
        }
        mapSlot(slot, null_it->second, SlotState::NULL_MAPPED);

        to_return.push_back({old_real.shareable_handle, old_real.granularity, old_real.device_id});
    }

    returnHandlesToDaemon(to_return, timeout_ms);
}

void releaseAllocation(void* ptr, int timeout_ms) {
    AllocationRecord& record = findAllocationOrThrow(ptr);
    if (record.is_kvcache) {
        std::vector<handle_granularity> to_return;
        for (auto& slot : record.slots) {
            if (slot.state == SlotState::REAL_MAPPED) {
                ImportedHandle old = slot.handle;
                unmapSlot(slot);
                to_return.push_back({old.shareable_handle, old.granularity, old.device_id});
            } else if (slot.state == SlotState::NULL_MAPPED) {
                unmapSlot(slot);
            }
        }
        returnHandlesToDaemon(to_return, timeout_ms);
    } else {
        unmapWeightAll(record, timeout_ms);
    }

    void* release_base = record.raw_base ? record.raw_base : record.base;
    CHECK_ACL("aclrtReleaseMemAddress", aclrtReleaseMemAddress(release_base));
    g_state.allocations.erase(ptr);
}

void unmapNullPagesBeforeFinalize() {
    for (auto& allocation_kv : g_state.allocations) {
        AllocationRecord& record = allocation_kv.second;
        for (auto& slot : record.slots) {
            if (slot.state == SlotState::NULL_MAPPED) {
                unmapSlot(slot);
            }
        }
    }
}

} // namespace

__attribute__((visibility("default"))) void* my_malloc_weight(ssize_t size,
                                                               int device,
                                                               aclrtStream stream) {
    (void)stream;
    std::unique_lock<std::mutex> guard(g_state.mutex);
    requireInitialized();
    ensureContext(device);

    const uint64_t map_granularity = kSupportedGranularities.back();
    const uint64_t reserve_granularity = kSupportedGranularities.front();
    const uint64_t aligned_size = alignUp(static_cast<uint64_t>(size), map_granularity);
    const ReservedAddress reserved =
        reserveVirtualAddress(static_cast<size_t>(aligned_size), reserve_granularity);
    void* d_mem = reserved.aligned_base;
    
    if(daemonDebugLogsEnabled()) {
        fprintf(stderr,
                "my_malloc_weight: requested size=%zd, aligned size=%zu, reserve_gran=%zu, raw_reserved=%zu, raw_base=%p, aligned_base=%p\n",
                size,
                static_cast<size_t>(aligned_size),
                static_cast<size_t>(reserve_granularity),
                reserved.raw_reserved_size,
                reserved.raw_base,
                d_mem);
    }

    AllocationRecord record;
    record.base = d_mem;
    record.raw_base = reserved.raw_base;
    record.requested_size = static_cast<size_t>(size);
    record.reserved_size = static_cast<size_t>(aligned_size);
    record.device_id = device;
    record.is_kvcache = false;
    g_state.allocations[d_mem] = std::move(record);

    guard.unlock();
    callPythonMallocCallback(device, aligned_size, d_mem);
    return d_mem;
}

__attribute__((visibility("default"))) void my_free_weight(void* ptr,
                                                             ssize_t size,
                                                             int device,
                                                             aclrtStream stream) {
    (void)size;
    (void)device;
    (void)stream;
    std::unique_lock<std::mutex> guard(g_state.mutex);
    requireInitialized();
    releaseAllocation(ptr, kDefaultWaitTimeoutMs);
    guard.unlock();
    callPythonFreeCallback(ptr);
}

__attribute__((visibility("default"))) void* my_malloc_kvcache(ssize_t size,
                                                                int device,
                                                                aclrtStream stream) {
    (void)stream;
    std::unique_lock<std::mutex> guard(g_state.mutex);
    requireInitialized();
    ensureContext(device);

    const uint64_t granularity = getKvcacheGranularity();
    const uint64_t aligned_size = alignUp(static_cast<uint64_t>(size), granularity);
    const ReservedAddress reserved =
        reserveVirtualAddress(static_cast<size_t>(aligned_size), granularity);
    void* d_mem = reserved.aligned_base;

    auto null_it = g_state.null_pages.find(granularity);
    if (null_it == g_state.null_pages.end()) {
        throw std::runtime_error("null page for selected kvcache granularity is not initialized");
    }

    AllocationRecord record;
    record.base = d_mem;
    record.raw_base = reserved.raw_base;
    record.requested_size = static_cast<size_t>(size);
    record.reserved_size = static_cast<size_t>(aligned_size);
    record.device_id = device;
    record.is_kvcache = true;
    record.kvcache_granularity = granularity;

    const size_t slot_count = static_cast<size_t>(aligned_size / granularity);
    record.slots.reserve(slot_count);
    Byte* base = static_cast<Byte*>(d_mem);
    for (size_t i = 0; i < slot_count; ++i) {
        MappingSlot slot;
        slot.virt_addr = base + i * granularity;
        slot.granularity = granularity;
        slot.device_id = device;
        mapSlot(slot, null_it->second, SlotState::NULL_MAPPED);
        record.slots.push_back(slot);
    }

    if(daemonDebugLogsEnabled()) {
        fprintf(stderr,
            "my_malloc_kvcache: requested size=%zd, aligned size=%zu, granularity=%zu, raw_reserved=%zu, raw_base=%p, aligned_base=%p, slot count=%zu\n",
            size,
            static_cast<size_t>(aligned_size),
            granularity,
            reserved.raw_reserved_size,
            reserved.raw_base,
            d_mem,
            slot_count);
    }

    g_state.allocations[d_mem] = std::move(record);
    guard.unlock();
    callPythonMallocCallback(device, aligned_size, d_mem);
    return d_mem;
}

__attribute__((visibility("default"))) void my_free_kvcache(void* ptr,
                                                              ssize_t size,
                                                              int device,
                                                              aclrtStream stream) {
    (void)size;
    (void)device;
    (void)stream;
    std::unique_lock<std::mutex> guard(g_state.mutex);
    requireInitialized();
    releaseAllocation(ptr, kDefaultWaitTimeoutMs);
    guard.unlock();
    callPythonFreeCallback(ptr);
}

static PyObject* py_init_module(PyObject* self, PyObject* args) {
    (void)self;
    PyObject* malloc_callback = nullptr;
    PyObject* free_callback = nullptr;

    if (!PyArg_ParseTuple(args, "OO", &malloc_callback, &free_callback)) {
        return nullptr;
    }

    if (!PyCallable_Check(malloc_callback) || !PyCallable_Check(free_callback)) {
        PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
        return nullptr;
    }

    Py_XINCREF(malloc_callback);
    Py_XINCREF(free_callback);
    Py_XDECREF(g_python_malloc_callback);
    Py_XDECREF(g_python_free_callback);
    g_python_malloc_callback = malloc_callback;
    g_python_free_callback = free_callback;
    Py_RETURN_NONE;
}

static PyObject* python_init_space(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long device_id_ull;
    unsigned long long canonical_device_id_ull = 0;
    if (!PyArg_ParseTuple(args, "K|K", &device_id_ull, &canonical_device_id_ull)) {
        PyErr_SetString(PyExc_TypeError, "Expected (local_device_id[, canonical_device_id])");
        return nullptr;
    }

    try {
        std::lock_guard<std::mutex> guard(g_state.mutex);
        if (g_state.initialized) {
            Py_RETURN_TRUE;
        }
        const int32_t local_device_id = static_cast<int32_t>(device_id_ull);
        const int32_t canonical_device_id =
            (PyTuple_Size(args) >= 2)
                ? static_cast<int32_t>(canonical_device_id_ull)
                : localToCanonicalDeviceId(local_device_id);
        registerToDaemonAndOpenMessageSpace(local_device_id,
                            canonical_device_id);
        initializeNullPages();
        touchHeartbeat();
        startHeartbeatThread();
        g_state.initialized = true;
        Py_RETURN_TRUE;
    } catch (const std::exception& ex) {
        maybeFinalizeState();
        PySys_WriteStderr("python_init_space failed: %s\n", ex.what());
        Py_RETURN_FALSE;
    }
}

static PyObject* python_finalize_space(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    try {
        std::lock_guard<std::mutex> guard(g_state.mutex);
        if (!g_state.initialized) {
            Py_RETURN_NONE;
        }

        stopHeartbeatThread();

        unmapNullPagesBeforeFinalize();

        std::vector<handle_granularity> to_return;
        for (auto& kv : g_state.null_pages) {
            ImportedHandle& h = kv.second;
            to_return.push_back({h.shareable_handle, h.granularity, h.device_id});
            releaseImportedHandle(h);
        }
        returnHandlesToDaemon(to_return, kDefaultWaitTimeoutMs);
        g_state.null_pages.clear();

        maybeFinalizeState();
        Py_RETURN_NONE;
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return nullptr;
    }
}

static PyObject* python_map_weight(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long device_ull;
    unsigned long long d_mem_ull;
    unsigned long long final_size_ull;
    int timeout_ms = kDefaultWaitTimeoutMs;
    if (!PyArg_ParseTuple(args, "KKK|i", &device_ull, &d_mem_ull, &final_size_ull, &timeout_ms)) {
        PyErr_SetString(PyExc_TypeError, "Expected (device_id, d_mem, final_size[, timeout_ms])");
        return nullptr;
    }

    try {
        std::lock_guard<std::mutex> guard(g_state.mutex);
        requireInitialized();
        ensureContext(static_cast<int32_t>(device_ull));
        AllocationRecord& record = findAllocationOrThrow(reinterpret_cast<void*>(d_mem_ull));
        mapWeightRange(record, static_cast<size_t>(final_size_ull), timeout_ms);
        Py_RETURN_NONE;
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return nullptr;
    }
}

static PyObject* python_unmap_weight(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long device_ull;
    unsigned long long d_mem_ull;
    int timeout_ms = kDefaultWaitTimeoutMs;
    if (!PyArg_ParseTuple(args, "KK|i", &device_ull, &d_mem_ull, &timeout_ms)) {
        PyErr_SetString(PyExc_TypeError, "Expected (device_id, d_mem[, timeout_ms])");
        return nullptr;
    }

    try {
        std::lock_guard<std::mutex> guard(g_state.mutex);
        requireInitialized();
        ensureContext(static_cast<int32_t>(device_ull));
        AllocationRecord& record = findAllocationOrThrow(reinterpret_cast<void*>(d_mem_ull));
        unmapWeightAll(record, timeout_ms);
        Py_RETURN_NONE;
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return nullptr;
    }
}

static PyObject* python_map_kvcache_until(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long device_ull;
    unsigned long long d_mem_ull;
    unsigned long long final_size_ull;
    int timeout_ms = kDefaultWaitTimeoutMs;
    if (!PyArg_ParseTuple(args, "KKK|i", &device_ull, &d_mem_ull, &final_size_ull, &timeout_ms)) {
        PyErr_SetString(PyExc_TypeError, "Expected (device_id, d_mem, final_size[, timeout_ms])");
        return nullptr;
    }

    try {
        std::lock_guard<std::mutex> guard(g_state.mutex);
        requireInitialized();
        ensureContext(static_cast<int32_t>(device_ull));
        AllocationRecord& record = findAllocationOrThrow(reinterpret_cast<void*>(d_mem_ull));
        mapKvcacheUntil(record, static_cast<size_t>(final_size_ull), timeout_ms);
        Py_RETURN_NONE;
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return nullptr;
    }
}

static PyObject* python_unmap_kvcache_until(PyObject* self, PyObject* args) {
    (void)self;
    unsigned long long device_ull;
    unsigned long long d_mem_ull;
    unsigned long long final_size_ull;
    int timeout_ms = kDefaultWaitTimeoutMs;
    if (!PyArg_ParseTuple(args, "KKK|i", &device_ull, &d_mem_ull, &final_size_ull, &timeout_ms)) {
        PyErr_SetString(PyExc_TypeError, "Expected (device_id, d_mem, final_size[, timeout_ms])");
        return nullptr;
    }

    try {
        std::lock_guard<std::mutex> guard(g_state.mutex);
        requireInitialized();
        ensureContext(static_cast<int32_t>(device_ull));
        AllocationRecord& record = findAllocationOrThrow(reinterpret_cast<void*>(d_mem_ull));
        unmapKvcacheUntil(record, static_cast<size_t>(final_size_ull), timeout_ms);
        Py_RETURN_NONE;
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return nullptr;
    }
}

static PyObject* python_debug_allocations(PyObject* self, PyObject* args) {
    (void)self;
    (void)args;
    try {
        std::lock_guard<std::mutex> guard(g_state.mutex);
        PyObject* out = PyList_New(0);
        if (!out) {
            return nullptr;
        }
        for (const auto& kv : g_state.allocations) {
            const AllocationRecord& rec = kv.second;
            PyObject* item = Py_BuildValue("{s:K,s:K,s:K,s:i,s:i,s:K,s:K}",
                                           "d_mem",
                                           static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(rec.base)),
                                           "requested_size",
                                           static_cast<unsigned long long>(rec.requested_size),
                                           "reserved_size",
                                           static_cast<unsigned long long>(rec.reserved_size),
                                           "device_id",
                                           rec.device_id,
                                           "is_kvcache",
                                           rec.is_kvcache ? 1 : 0,
                                           "slots",
                                           static_cast<unsigned long long>(rec.slots.size()),
                                           "mapped_weight_bytes",
                                           static_cast<unsigned long long>(rec.mapped_weight_bytes));
            if (!item) {
                Py_DECREF(out);
                return nullptr;
            }
            PyList_Append(out, item);
            Py_DECREF(item);
        }
        return out;
    } catch (const std::exception& ex) {
        PyErr_SetString(PyExc_RuntimeError, ex.what());
        return nullptr;
    }
}

static PyMethodDef module_methods[] = {
    {"init_module", (PyCFunction)py_init_module, METH_VARARGS,
     "Initialize module with python_malloc and python_free callables."},
    {"python_init_space", (PyCFunction)python_init_space, METH_VARARGS,
     "Register to daemon and initialize per-model message space + null pages."},
    {"python_finalize_space", (PyCFunction)python_finalize_space, METH_NOARGS,
     "Return held null pages and close model message space."},
    {"python_map_weight", (PyCFunction)python_map_weight, METH_VARARGS,
     "Map weight range by requesting handles from daemon."},
    {"python_unmap_weight", (PyCFunction)python_unmap_weight, METH_VARARGS,
     "Unmap all weight handles and return them to daemon."},
    {"python_map_kvcache_until", (PyCFunction)python_map_kvcache_until, METH_VARARGS,
     "Map kvcache slots until target size (ceil policy)."},
    {"python_unmap_kvcache_until", (PyCFunction)python_unmap_kvcache_until, METH_VARARGS,
     "Unmap kvcache slots until target size (ceil policy)."},
    {"python_debug_allocations", (PyCFunction)python_debug_allocations, METH_NOARGS,
     "Return current allocation metadata for debugging."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef camem_allocator_module = {
    PyModuleDef_HEAD_INIT,
    "bifrost_ascend_C",
    "CANN-mem-based allocator for NPUPluggableAllocator",
    -1,
    module_methods
};

PyMODINIT_FUNC PyInit_bifrost_ascend_C(void) {
    PyObject* module = PyModule_Create(&camem_allocator_module);
    if (!module) {
        return NULL;
    }
    return module;
}

PyMODINIT_FUNC PyInit_vllm_ascend_C(void) {
    return PyInit_bifrost_ascend_C();
}

} // extern "C"
