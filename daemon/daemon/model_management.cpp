#include "inc/model_management.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace mdaemon {

namespace {

std::mt19937_64& modelIdEngine() {
    static std::mt19937_64 engine(std::random_device{}());
    return engine;
}

uint64_t drawModelId() noexcept {
    static std::uniform_int_distribution<uint64_t> dist(1, std::numeric_limits<uint64_t>::max() - 1);
    return dist(modelIdEngine());
}

void logShmError(const char* op, std::string_view shm_name) {
    std::cerr << "\033[31m[ModelConfig] " << op << " failed for shm '" << shm_name
              << "' errno=" << errno << " (" << std::strerror(errno) << ")\033[0m\n";
}

std::string buildMessageSemName(std::string_view shm_name) {
    return std::string(shm_name) + "-sem";
}

} // namespace

class ModelMacroRegistry {
public:
    ModelMacroRegistry() {
        fd_ = ::shm_open(MODEL_REG_SHM, O_RDWR | O_CREAT, 0666);
        if (fd_ == -1) {
            return;
        }
        struct stat file_state{};
        if (::fstat(fd_, &file_state) == -1) {
            cleanup();
            return;
        }
        const bool needs_init = file_state.st_size == 0;
        if (::ftruncate(fd_, sizeof(ModelMacroSpace)) == -1) {
            cleanup();
            return;
        }
        void* region = ::mmap(nullptr,
                              sizeof(ModelMacroSpace),
                              PROT_READ | PROT_WRITE,
                              MAP_SHARED,
                              fd_,
                              0);
        if (region == MAP_FAILED) {
            cleanup();
            return;
        }
        data_ = static_cast<ModelMacroSpace*>(region);
        if (needs_init && data_) {
            data_->reset();
        }
    }

    ~ModelMacroRegistry() {
        cleanup();
    }

    ModelMacroSpace* data() const noexcept {
        return data_;
    }

    bool hasPendingRegistration() const noexcept {
        if (!data_) {
            return false;
        }
        return data_->hasCandidate() && isDefaultShmName(data_->getShmName());
    }

    void writeShmName(std::string_view new_name) {
        if (!data_) {
            return;
        }
        data_->setShmName(new_name);
    }

private:
    void cleanup() noexcept {
        if (data_) {
            ::msync(data_, sizeof(ModelMacroSpace), MS_SYNC);
            ::munmap(data_, sizeof(ModelMacroSpace));
            data_ = nullptr;
        }
        if (fd_ != -1) {
            ::close(fd_);
            fd_ = -1;
        }
        ::shm_unlink(MODEL_REG_SHM);
    }

    int fd_{-1};
    ModelMacroSpace* data_{nullptr};
};

ModelConfig::~ModelConfig() {
    closeMessageSemaphore();
    closeMessageShmMapping();
    // Shared memory unlink is managed by ModelConfigManager::removeModelConfig
    // so that shm lifetime is tied to explicit model removal.
}

bool ModelConfig::setModelConfig() noexcept {
    std::lock_guard<std::mutex> guard(state_mutex_);
    if (model_id_ != std::numeric_limits<uint64_t>::max()) {
        return true;
    }
    model_id_ = generateModelId();
    state_ = ModelState::INITIALIZED;
    message_shm_name_ = buildShmName(model_id_);
    // message_space_.model_npuid = model_id_;
    // message_space_.model_osid = model_id_;
    return true;
}

uint64_t ModelConfig::getModelId() const noexcept {
    return model_id_;
}

ModelState ModelConfig::getState() const noexcept {
    std::lock_guard<std::mutex> guard(state_mutex_);
    return state_;
}

void ModelConfig::setState(ModelState new_state) noexcept {
    std::lock_guard<std::mutex> guard(state_mutex_);
    state_ = new_state;
}

ModelMessageSpace& ModelConfig::messageSpace() {
    if (!message_space_) {
        throw std::runtime_error("message space is not mapped");
    }
    return *message_space_;
}

const ModelMessageSpace& ModelConfig::messageSpace() const {
    if (!message_space_) {
        throw std::runtime_error("message space is not mapped");
    }
    return *message_space_;
}

bool ModelConfig::syncMessageSpaceFromShm() noexcept {
    if (!message_space_) {
        std::cerr << "\033[31m[ModelConfig] syncMessageSpaceFromShm skipped: shm not mapped for '"
                  << message_shm_name_ << "'\033[0m\n";
        return false;
    }
    // message_space_ is already the mapped shared memory region.
    return true;
}

bool ModelConfig::syncMessageSpaceToShm() noexcept {
    if (!message_space_) {
        std::cerr << "\033[31m[ModelConfig] syncMessageSpaceToShm skipped: shm not mapped for '"
                  << message_shm_name_ << "'\033[0m\n";
        return false;
    }
    // message_space_ is already the mapped shared memory region.
    return true;
}

bool ModelConfig::tryLockMessageSpace() noexcept {
    if (!message_sem_) {
        return false;
    }
    if (::sem_trywait(message_sem_) == 0) {
        return true;
    }
    if (errno == EAGAIN) {
        return false;
    }
    std::cerr << "\033[31m[ModelConfig] sem_trywait failed for '"
              << message_sem_name_ << "' errno=" << errno << " (" << std::strerror(errno)
              << ")\033[0m\n";
    return false;
}

void ModelConfig::unlockMessageSpace() noexcept {
    if (!message_sem_) {
        return;
    }
    if (::sem_post(message_sem_) == -1) {
        std::cerr << "\033[31m[ModelConfig] sem_post failed for '"
                  << message_sem_name_ << "' errno=" << errno << " (" << std::strerror(errno)
                  << ")\033[0m\n";
    }
}

bool ModelConfig::openAndMapMessageShm(bool create) noexcept {
    const std::string shm_name(message_shm_name_);
    if (shm_name.empty() || isDefaultShmName(shm_name)) {
        std::cerr << "\033[31m[ModelConfig] openAndMapMessageShm skipped: invalid shm name\033[0m\n";
        return false;
    }

    closeMessageSemaphore();
    closeMessageShmMapping();

    int flags = O_RDWR;
    if (create) {
        flags |= O_CREAT;
    }

    const int fd = ::shm_open(shm_name.c_str(), flags, 0666);
    if (fd == -1) {
        logShmError("shm_open", shm_name);
        return false;
    }

    if (create && ::ftruncate(fd, sizeof(ModelMessageSpace)) == -1) {
        logShmError("ftruncate", shm_name);
        ::close(fd);
        return false;
    }

    void* region = ::mmap(nullptr,
                          sizeof(ModelMessageSpace),
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED,
                          fd,
                          0);
    if (region == MAP_FAILED) {
        logShmError("mmap", shm_name);
        ::close(fd);
        return false;
    }

    message_fd_ = fd;
    message_space_ = static_cast<ModelMessageSpace*>(region);

    if (!openMessageSemaphore(create)) {
        closeMessageSemaphore();
        closeMessageShmMapping();
        return false;
    }
    return true;
}

void ModelConfig::closeMessageShmMapping() noexcept {
    if (message_space_) {
        ::munmap(message_space_, sizeof(ModelMessageSpace));
        message_space_ = nullptr;
    }
    if (message_fd_ != -1) {
        ::close(message_fd_);
        message_fd_ = -1;
    }
}

bool ModelConfig::openMessageSemaphore(bool create) noexcept {
    message_sem_name_ = buildMessageSemName(message_shm_name_);
    if (message_sem_name_.empty()) {
        return false;
    }

    int flags = 0;
    if (create) {
        flags |= O_CREAT;
    }

    message_sem_ = ::sem_open(message_sem_name_.c_str(), flags, 0666, 1);
    if (message_sem_ == SEM_FAILED) {
        message_sem_ = nullptr;
        std::cerr << "\033[31m[ModelConfig] sem_open failed for '"
                  << message_sem_name_ << "' errno=" << errno << " (" << std::strerror(errno)
                  << ")\033[0m\n";
        return false;
    }
    return true;
}

void ModelConfig::closeMessageSemaphore() noexcept {
    if (message_sem_) {
        ::sem_close(message_sem_);
        message_sem_ = nullptr;
    }
}

void ModelConfig::addAllocatedHandle(int32_t canonical_device_id, Handle&& handle) {
    const auto key = std::make_pair(canonical_device_id, handle.shareableHandle());
    std::lock_guard<std::mutex> guard(allocated_mutex_);
    if (allocated_handles_.find(key) != allocated_handles_.end()) {
        throw std::runtime_error("duplicate allocated handle key detected");
    }
    allocated_handles_.emplace(key, std::move(handle));
}

void ModelConfig::setPidToShareForAllocated(const std::vector<int32_t>& npuid_list) {
    if (npuid_list.empty()) {
        return;
    }
    std::vector<int32_t> mutable_npuid_list = npuid_list;
    std::lock_guard<std::mutex> guard(allocated_mutex_);
    for (const auto& kv : allocated_handles_) {
        const uint64_t shareable_handle = kv.first.second;
        CHECK_ERROR(aclrtMemSetPidToShareableHandle(shareable_handle,
                                                    mutable_npuid_list.data(),
                                                    mutable_npuid_list.size()));
    }
}

bool ModelConfig::takeAllocatedHandle(int32_t device_id, uint64_t shareable_handle, Handle& out_handle) {
    std::lock_guard<std::mutex> guard(allocated_mutex_);
    auto it = allocated_handles_.find({device_id, shareable_handle});
    if (it == allocated_handles_.end()) {
        return false;
    }
    out_handle = std::move(it->second);
    allocated_handles_.erase(it);
    return true;
}

std::vector<Handle> ModelConfig::drainAllocatedHandles() {
    std::lock_guard<std::mutex> guard(allocated_mutex_);
    std::vector<Handle> handles;
    handles.reserve(allocated_handles_.size());
    for (auto& kv : allocated_handles_) {
        handles.push_back(std::move(kv.second));
    }
    allocated_handles_.clear();
    return handles;
}

size_t ModelConfig::allocatedHandleCount() const noexcept {
    std::lock_guard<std::mutex> guard(allocated_mutex_);
    return allocated_handles_.size();
}

uint64_t ModelConfig::allocatedBytes() const noexcept {
    std::lock_guard<std::mutex> guard(allocated_mutex_);
    uint64_t total = 0;
    for (const auto& kv : allocated_handles_) {
        total += kv.second.granularity();
    }
    return total;
}

void ModelConfig::unlinkMessageShm() noexcept {
    const std::string shm_name(message_shm_name_);
    if (shm_name.empty() || isDefaultShmName(shm_name)) {
        return;
    }
    closeMessageSemaphore();
    closeMessageShmMapping();
    if (!message_sem_name_.empty()) {
        if (::sem_unlink(message_sem_name_.c_str()) == -1 && errno != ENOENT) {
            std::cerr << "\033[31m[ModelConfig] sem_unlink failed for '"
                      << message_sem_name_ << "' errno=" << errno << " (" << std::strerror(errno)
                      << ")\033[0m\n";
        }
    }
    if (::shm_unlink(shm_name.c_str()) == -1 && errno != ENOENT) {
        logShmError("shm_unlink", shm_name);
    }
}

uint64_t ModelConfig::generateModelId() noexcept {
    return drawModelId();
}

std::string ModelConfig::buildShmName(uint64_t model_id) {
    return std::string(SHM_PREFIX) + std::to_string(model_id);
}

ModelConfigManager::ModelConfigManager() noexcept
    : macro_registry_(new ModelMacroRegistry()) {}

ModelConfigManager::~ModelConfigManager() {
    std::vector<uint64_t> ids = listModelIds();
    for (uint64_t id : ids) {
        removeModelConfig(id);
    }
    // remove macro_registry_ shm file
    macro_registry_.reset();
}

ModelConfig* ModelConfigManager::addModelConfig() {
    std::lock_guard<std::mutex> guard(map_mutex_);
    constexpr int kMaxAttempts = 5;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        auto config = std::make_unique<ModelConfig>();
        if (!config->setModelConfig()) {
            return nullptr;
        }
        const uint64_t id = config->getModelId();
        auto [it, inserted] = model_config_map_.try_emplace(id, std::move(config));
        if (inserted) {
            ++total_models_;
            return it->second.get();
        }
    }
    return nullptr;
}

void ModelConfigManager::removeModelConfig(uint64_t model_id) {
    std::lock_guard<std::mutex> guard(map_mutex_);
    auto it = model_config_map_.find(model_id);
    if (it != model_config_map_.end()) {
        const int32_t npuid = it->second->messageSpace().model_npuid;
        it->second->unlinkMessageShm();
        model_config_map_.erase(it);
        if (total_models_ > 0) {
            --total_models_;
        }
        if (npuid != kInvalidModelHandle) {
            current_model_npuids_.erase(
                std::remove(current_model_npuids_.begin(), current_model_npuids_.end(), npuid),
                current_model_npuids_.end());
        }
    }
}

ModelConfig* ModelConfigManager::getModelConfig(uint64_t model_id) {
    std::lock_guard<std::mutex> guard(map_mutex_);
    auto it = model_config_map_.find(model_id);
    return it != model_config_map_.end() ? it->second.get() : nullptr;
}

bool ModelConfigManager::isModelDone(uint64_t model_id) {
    std::lock_guard<std::mutex> guard(map_mutex_);
    auto it = model_config_map_.find(model_id);
    if (it == model_config_map_.end()) {
        return false;
    }
    it->second->syncMessageSpaceFromShm();
    ModelMessageSpace& space = it->second->messageSpace();
    if (space.state != MessageState::DONE) {
        return false;
    }
    it->second->setState(ModelState::DONE);
    return true;
}

bool ModelConfigManager::updateModelState(uint64_t model_id, ModelState new_state) {
    std::lock_guard<std::mutex> guard(map_mutex_);
    auto it = model_config_map_.find(model_id);
    if (it == model_config_map_.end()) {
        return false;
    }
    it->second->setState(new_state);
    return true;
}

ModelState ModelConfigManager::getModelState(uint64_t model_id) const noexcept {
    std::lock_guard<std::mutex> guard(map_mutex_);
    auto it = model_config_map_.find(model_id);
    if (it == model_config_map_.end()) {
        return ModelState::INVALID;
    }
    return it->second->getState();
}

ModelMacroSpace* ModelConfigManager::macroSpace() const noexcept {
    return macro_registry_ ? macro_registry_->data() : nullptr;
}

std::vector<uint64_t> ModelConfigManager::listModelIds() const {
    std::lock_guard<std::mutex> guard(map_mutex_);
    std::vector<uint64_t> ids;
    ids.reserve(model_config_map_.size());
    for (const auto& kv : model_config_map_) {
        ids.push_back(kv.first);
    }
    return ids;
}


ModelConfig* ModelConfigManager::registerModelFromMacro() {
    if (!macro_registry_ || !macro_registry_->hasPendingRegistration()) {
        return nullptr;
    }
    ModelMacroSpace* macro_space = macro_registry_->data();
    if (!macro_space) {
        return nullptr;
    }
    ModelConfig* config = addModelConfig();
    if (!config) {
        return nullptr;
    }
    const uint64_t model_id = config->getModelId();
    bool setup_ok = config->openAndMapMessageShm(true);
    if (setup_ok) {
        ModelMessageSpace& space = config->messageSpace();
        space = ModelMessageSpace{};
        space.setShmName(config->message_shm_name_);
        space.model_npuid = macro_space->current_model_npuid;
        space.model_osid = macro_space->current_model_osid;
        space.state = MessageState::REQUEST_REGISTER;
        setup_ok = config->syncMessageSpaceToShm();
    }

    if (!setup_ok) {
        if (macro_registry_) {
            macro_registry_->writeShmName(defaultShmName());
        }
        removeModelConfig(model_id);
        return nullptr;
    }

    macro_registry_->writeShmName(config->message_shm_name_);

    // update current npu id list; set all to pidShare
    {
        std::lock_guard<std::mutex> guard(map_mutex_);
        ModelMessageSpace& space = config->messageSpace();
        if (space.model_npuid != kInvalidModelHandle) {
            auto pos = std::find(current_model_npuids_.begin(), current_model_npuids_.end(), space.model_npuid);
            if (pos == current_model_npuids_.end()) {
                current_model_npuids_.push_back(space.model_npuid);
            }
        }
    }
    
    return config;
}

} // namespace mdaemon
