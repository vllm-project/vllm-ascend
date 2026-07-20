#include "inc/handle_pool.hpp"
#include "inc/utils.h"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace mdaemon {

namespace {
std::atomic_bool g_acl_initialized{false};
std::mutex g_acl_init_mutex;

void ensureAclInitialized() {
    if (g_acl_initialized.load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard<std::mutex> guard(g_acl_init_mutex);
    if (!g_acl_initialized.load(std::memory_order_relaxed)) {
        CHECK_ERROR(aclInit(nullptr));
        g_acl_initialized.store(true, std::memory_order_release);
    }
}
} // namespace

Handle::Handle(Handle&& other) noexcept
    : handle_(other.handle_),
      device_id_(other.device_id_),
      granularity_(other.granularity_),
      owns_handle_(other.owns_handle_),
      shareable_handle_(other.shareable_handle_) {
    other.handle_ = aclrtDrvMemHandle{};
    other.device_id_ = -1;
    other.granularity_ = 0;
    other.owns_handle_ = false;
    other.shareable_handle_ = std::numeric_limits<uint64_t>::max();
}

Handle& Handle::operator=(Handle&& other) noexcept {
    if (this != &other) {
        cleanup();
        handle_ = other.handle_;
        device_id_ = other.device_id_;
        granularity_ = other.granularity_;
        owns_handle_ = other.owns_handle_;
        shareable_handle_ = other.shareable_handle_;

        other.handle_ = aclrtDrvMemHandle{};
        other.device_id_ = -1;
        other.granularity_ = 0;
        other.owns_handle_ = false;
        other.shareable_handle_ = std::numeric_limits<uint64_t>::max();
    }
    return *this;
}

Handle::~Handle() {
    cleanup();
}

bool Handle::allocate(int32_t device_id, uint64_t size) {
    if (owns_handle_) {
        throw std::logic_error("Handle already owns a physical resource");
    }
    if (size == 0) {
        throw std::invalid_argument("Handle size must be positive");
    }

    ensure_context(device_id);

    aclrtPhysicalMemProp prop = {};
    prop.handleType = ACL_MEM_HANDLE_TYPE_NONE;
    prop.allocationType = ACL_MEM_ALLOCATION_TYPE_PINNED;
    prop.memAttr = ACL_HBM_MEM_HUGE;
    prop.location.id = device_id;
    prop.location.type = ACL_MEM_LOCATION_TYPE_DEVICE;
    prop.reserve = 0;

    // Check if the allocation is supported
    size_t granularity;
    CHECK_ERROR(aclrtMemGetAllocationGranularity(&prop,
                                ACL_RT_MEM_ALLOC_GRANULARITY_MINIMUM,
                                &granularity));
    size = ((size + granularity - 1) / granularity) * granularity; // align size to granularity

    aclrtDrvMemHandle raw_handle;
    CHECK_ERROR(aclrtMallocPhysical(&raw_handle, static_cast<size_t>(size), &prop, 0));
    handle_ = raw_handle;
    device_id_ = device_id;
    granularity_ = size;
    owns_handle_ = true;
    return true;
}

void Handle::release() {
    cleanup();
}

void Handle::reset(int32_t device_id, aclrtDrvMemHandle handle, uint64_t granularity, uint64_t shareable_handle) noexcept {
    cleanup();
    handle_ = handle;
    device_id_ = device_id;
    granularity_ = granularity;
    owns_handle_ = true;
    shareable_handle_ = shareable_handle;
}

bool Handle::valid() const noexcept {
    return owns_handle_;
}

int32_t Handle::deviceId() const noexcept {
    return device_id_;
}

uint64_t Handle::granularity() const noexcept {
    return granularity_;
}

aclrtDrvMemHandle Handle::rawHandle() const noexcept {
    return handle_;
}

uint64_t Handle::shareableHandle() const noexcept {
    return shareable_handle_;
}

uint64_t Handle::exportShareableHandle() {
    if (!owns_handle_) {
        throw std::runtime_error("Cannot export an invalid handle");
    }
    uint64_t shareable = 0;
    CHECK_ERROR(aclrtMemExportToShareableHandle(handle_, ACL_MEM_HANDLE_TYPE_NONE, 0, &shareable));
    shareable_handle_ = shareable;
    return shareable_handle_;
}

void Handle::cleanup() noexcept {
    if (!owns_handle_) {
        return;
    }
    ensure_context(device_id_);
    CHECK_ERROR(aclrtFreePhysical(handle_));
    handle_ = aclrtDrvMemHandle{};
    device_id_ = -1;
    granularity_ = 0;
    owns_handle_ = false;
    shareable_handle_ = std::numeric_limits<uint64_t>::max();
}

HandlePool::HandlePool(uint64_t page_size)
    // page_size is 2MB by default
    : page_size_(page_size), running_(true) {
    if (page_size_ == 0) {
        throw std::invalid_argument("page_size must be positive");
    }
}

HandlePool::~HandlePool() {
    shutdown();
}

void HandlePool::ensureInitialized(int32_t device_id) {
    if (!running_) {
        throw std::runtime_error("HandlePool is no longer running");
    }
    ensureAclInitialized();
    ensure_context(device_id);
}

void HandlePool::allocateHandles(int32_t device_id,
                                 size_t count,
                                 const std::vector<int32_t>& npuid_list) {
    if (count == 0) {
        return;
    }
    ensureInitialized(device_id);
    auto& bucket = device_handle_map_[device_id];
    std::vector<int32_t> mutable_npuid_list = npuid_list;
    for (size_t idx = 0; idx < count; ++idx) {
        Handle handle;
        handle.allocate(device_id, page_size_);
        // by default, we set handle to share
        uint64_t shareable_handle = handle.exportShareableHandle();
        if (!mutable_npuid_list.empty()) {
            CHECK_ERROR(aclrtMemSetPidToShareableHandle(shareable_handle,
                                                       mutable_npuid_list.data(),
                                                       mutable_npuid_list.size()));
        }
        device_shareable_handle_map_[device_id].push_back(shareable_handle);
        bucket.emplace_back(std::move(handle));
        device_total_handle_map_[device_id] += 1;
    }
}

void HandlePool::initializeDevice(int32_t device_id,
                                  uint64_t total_bytes,
                                  const std::vector<int32_t>& npuid_list) {
    if (total_bytes == 0) {
        throw std::invalid_argument("total_bytes must be positive");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (device_handle_map_.count(device_id)) {
        return;
    }
    auto handle_count = static_cast<size_t>((total_bytes + page_size_ - 1) / page_size_);
    allocateHandles(device_id, std::max<size_t>(1, handle_count), npuid_list);
}

Handle HandlePool::acquire(int32_t device_id) {
    // this function will always return a Handle, if no available handle, return an invalid one
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_handle_map_.find(device_id);
    if (it == device_handle_map_.end() || it->second.empty()) {
        // no enough handles, return invalid Handle. Keep warning only in debug mode.
        if (daemonDebugLogsEnabled()) {
            daemonLogDebug("[HandlePool] no available handles for device " + std::to_string(device_id));
        }
        return Handle{};
    }
    Handle handle = std::move(it->second.front());
    it->second.pop_front();
    return handle;
}

void HandlePool::release(int32_t device_id, Handle handle) {
    if (!handle.valid()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    device_handle_map_[device_id].emplace_back(std::move(handle));
}

// void HandlePool::exportShareableHandles(int32_t device_id) {
//     std::cout << "[WARNING] exportShareableHandles is deprecated and will be removed in future versions.\n"
//                   "ShareableHandle is set by default when create the physical handle" << std::endl;
//     std::lock_guard<std::mutex> lock(mutex_);
//     auto it = device_handle_map_.find(device_id);
//     if (it == device_handle_map_.end()) {
//         return;
//     }
//     for (auto& handle : it->second) {
//         handle.exportShareableHandle();
//     }
//     return;
// }

bool HandlePool::extendHandles(int32_t device_id,
                               size_t additional_count,
                               const std::vector<int32_t>& npuid_list) {
    if (additional_count == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        allocateHandles(device_id, additional_count, npuid_list);
    } catch (const std::exception& ex) {
        daemonLogError("Failed to extend handles for device " + std::to_string(device_id)
                  + ": " + ex.what());
        return false;
    }
    return true;
}

bool HandlePool::removeHandles(int32_t device_id, size_t remove_count) {
    if (remove_count == 0) {
        return true;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_handle_map_.find(device_id);
    if (it == device_handle_map_.end()) {
        return false;
    }
    auto& bucket = it->second;
    if (remove_count > bucket.size()) {
        return false;
    }
    for (size_t idx = 0; idx < remove_count; ++idx) {
        uint64_t shareable_handle = bucket.back().shareableHandle();
        auto& shareable_handles = device_shareable_handle_map_[device_id];
        auto pos = std::find(shareable_handles.begin(), shareable_handles.end(), shareable_handle);
        if (pos != shareable_handles.end()) {
            shareable_handles.erase(pos);
        }
        bucket.back().release();
        bucket.pop_back();
    }
    auto total_it = device_total_handle_map_.find(device_id);
    if (total_it != device_total_handle_map_.end()) {
        total_it->second -= remove_count;
    }
    return true;
}

void HandlePool::memSetPidToShare(const std::vector<int32_t>& npuid_list, int32_t device_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_shareable_handle_map_.find(device_id);
    if (it == device_shareable_handle_map_.end()) {
        return;
    }
    std::vector<int32_t> mutable_npuid_list = npuid_list;
    for (auto& shareable_handle : it->second) {
        CHECK_ERROR(aclrtMemSetPidToShareableHandle(shareable_handle,
                                                   mutable_npuid_list.data(),
                                                   mutable_npuid_list.size()));
    }
}

size_t HandlePool::available(int32_t device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_handle_map_.find(device_id);
    if (it == device_handle_map_.end()) {
        return 0;
    }
    return it->second.size();
}

size_t HandlePool::total(int32_t device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_total_handle_map_.find(device_id);
    if (it == device_total_handle_map_.end()) {
        return 0;
    }
    return it->second;
}

size_t HandlePool::used(int32_t device_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const size_t available_count = device_handle_map_.count(device_id) ? device_handle_map_.at(device_id).size() : 0;
    const size_t total_count = device_total_handle_map_.count(device_id) ? device_total_handle_map_.at(device_id) : 0;
    return total_count >= available_count ? (total_count - available_count) : 0;
}

std::vector<int32_t> HandlePool::listDeviceIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<int32_t> device_ids;
    device_ids.reserve(device_handle_map_.size());
    for (const auto& pair : device_handle_map_) {
        device_ids.push_back(pair.first);
    }
    return device_ids;
}

void HandlePool::shutdown() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!running_) {
        return;
    }
    for (auto& pair : device_handle_map_) {
        for (auto& handle : pair.second) {
            handle.release();
        }
        pair.second.clear();
    }
    device_handle_map_.clear();
    device_shareable_handle_map_.clear();
    device_total_handle_map_.clear();
    running_ = false;
}

} // namespace mdaemon