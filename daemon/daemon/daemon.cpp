#include "inc/utils.h"
#include "inc/daemon.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <thread>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include <errno.h>
#include <unistd.h>
#include <semaphore.h>

namespace mdaemon {

namespace {
constexpr int kDefaultLoopIntervalMs = 50;
constexpr int64_t kDefaultLoopIntervalUs = static_cast<int64_t>(kDefaultLoopIntervalMs) * 1000;
constexpr int64_t kHeartbeatTimeoutNs = 5LL * 1000LL * 1000LL * 1000LL;

int64_t monotonicNowNs() {
	return std::chrono::duration_cast<std::chrono::nanoseconds>(
		std::chrono::steady_clock::now().time_since_epoch())
		.count();
}

bool heartbeatTimedOut(int64_t heartbeat_ns, int64_t now_ns) {
	if (heartbeat_ns <= 0) {
		return true;
	}
	if (heartbeat_ns > now_ns) {
		return false;
	}
	return (now_ns - heartbeat_ns) > kHeartbeatTimeoutNs;
}

int64_t daemonLoopIntervalUs() {
	const char* raw_us = std::getenv("MDAEMON_LOOP_INTERVAL_US");
	if (raw_us && raw_us[0] != '\0') {
		char* end = nullptr;
		errno = 0;
		long long value = std::strtoll(raw_us, &end, 10);
		if (errno != 0 || end == raw_us || (end && *end != '\0') || value <= 0) {
			daemonLogError(std::string("[Daemon] invalid MDAEMON_LOOP_INTERVAL_US='") + raw_us +
				"', using default " + std::to_string(kDefaultLoopIntervalUs) + "us");
			return kDefaultLoopIntervalUs;
		}
		return static_cast<int64_t>(value);
	}

	const char* raw = std::getenv("MDAEMON_LOOP_INTERVAL_MS");
	if (!raw || raw[0] == '\0') {
		return kDefaultLoopIntervalUs;
	}

	char* end = nullptr;
	errno = 0;
	long long value = std::strtoll(raw, &end, 10);
	if (errno != 0 || end == raw || (end && *end != '\0') || value <= 0) {
		daemonLogError(std::string("[Daemon] invalid MDAEMON_LOOP_INTERVAL_MS='") + raw +
			"', using default " + std::to_string(kDefaultLoopIntervalMs) + "ms");
		return kDefaultLoopIntervalUs;
	}
	if (value > static_cast<long long>(std::numeric_limits<int64_t>::max() / 1000LL)) {
		daemonLogError(std::string("[Daemon] MDAEMON_LOOP_INTERVAL_MS too large='") + raw +
			"', using default " + std::to_string(kDefaultLoopIntervalUs) + "us");
		return kDefaultLoopIntervalUs;
	}
	return static_cast<int64_t>(value) * 1000LL;
}

bool mapCanonicalToLocalDeviceId(int32_t canonical_device_id,
								 int32_t& local_device_id) {
	if (canonical_device_id < 0) {
		return false;
	}
	const char* raw = std::getenv("ASCEND_RT_VISIBLE_DEVICES");
	if (!raw || raw[0] == '\0') {
		raw = std::getenv("ASCEND_VISIBLE_DEVICES");
	}
	if (!raw || raw[0] == '\0') {
		raw = std::getenv("CUDA_VISIBLE_DEVICES");
	}
	if (!raw || raw[0] == '\0') {
		local_device_id = canonical_device_id;
		return true;
	}

	const std::string visible(raw);
	size_t start = 0;
	int32_t idx = 0;
	while (start < visible.size()) {
		size_t end = visible.find(',', start);
		if (end == std::string::npos) {
			end = visible.size();
		}
		const std::string token = visible.substr(start, end - start);
		if (!token.empty()) {
			try {
				if (static_cast<int32_t>(std::stoi(token)) == canonical_device_id) {
					local_device_id = idx;
					return true;
				}
				++idx;
			} catch (...) {
				return false;
			}
		}
		start = end + 1;
	}
	return false;
}
} // namespace

Daemon::~Daemon() {
	stop();

	// delete semaphore
	::sem_unlink(MODEL_SEMAPHORE_NAME);
}

void Daemon::start() {
	std::lock_guard<std::mutex> guard(daemon_mutex_);
	if (running_.load()) {
		return;
	}
	running_.store(true);
	worker_ = std::thread(&Daemon::runLoop, this);
}

void Daemon::stop() {
	{
		std::lock_guard<std::mutex> guard(daemon_mutex_);
		running_.store(false);
	}
	if (worker_.joinable()) {
		worker_.join();
	}
}

/* Handles pool manipulation */

HandlePool* Daemon::findHandlePool(uint64_t granularity) {
	std::lock_guard<std::mutex> guard(daemon_mutex_);
	auto it = handle_pools_.find(granularity);
	return it == handle_pools_.end() ? nullptr : it->second.get();
}

HandlePool* Daemon::ensureHandlePool(uint64_t granularity) {
	std::lock_guard<std::mutex> guard(daemon_mutex_);
	auto it = handle_pools_.find(granularity);
	if (it != handle_pools_.end()) {
		return it->second.get();
	}
	auto pool = std::make_unique<HandlePool>(granularity);
	auto [insert_it, _] = handle_pools_.try_emplace(granularity, std::move(pool));
	return insert_it->second.get();
}

void Daemon::initializeHandlePoolDevice(uint64_t granularity, int32_t device_id, uint64_t total_bytes) {
	if (HandlePool* pool = ensureHandlePool(granularity)) {
		const std::vector<int32_t> npuid_list = getCurrentModelNpuids();
		pool->initializeDevice(device_id, total_bytes, npuid_list);
	}
}

size_t Daemon::handlePoolAvailable(uint64_t granularity, int32_t device_id) const {
	std::lock_guard<std::mutex> guard(daemon_mutex_);
	if (auto it = handle_pools_.find(granularity); it != handle_pools_.end()) {
		return it->second->available(device_id);
	}
	return 0;
}

bool Daemon::extendHandles(uint64_t granularity, int32_t device_id, size_t count) {
	if (HandlePool* pool = findHandlePool(granularity)) {
		const std::vector<int32_t> npuid_list = getCurrentModelNpuids();
		return pool->extendHandles(device_id, count, npuid_list);
	}
	return false;
}

bool Daemon::removeHandles(uint64_t granularity, int32_t device_id, size_t count) {
	if (HandlePool* pool = findHandlePool(granularity)) {
		return pool->removeHandles(device_id, count);
	}
	return false;
}

std::vector<Daemon::HandlePoolDeviceSnapshot> Daemon::snapshotHandlePools() const {
	std::vector<std::pair<uint64_t, HandlePool*>> pools;
	{
		std::lock_guard<std::mutex> guard(daemon_mutex_);
		pools.reserve(handle_pools_.size());
		for (const auto& kv : handle_pools_) {
			pools.push_back({kv.first, kv.second.get()});
		}
	}

	std::vector<HandlePoolDeviceSnapshot> out;
	for (const auto& entry : pools) {
		const uint64_t granularity = entry.first;
		HandlePool* pool = entry.second;
		if (!pool) {
			continue;
		}
		for (int32_t device_id : pool->listDeviceIds()) {
			HandlePoolDeviceSnapshot snap;
			snap.granularity = granularity;
			snap.device_id = device_id;
			snap.available_handles = static_cast<uint64_t>(pool->available(device_id));
			snap.total_handles = static_cast<uint64_t>(pool->total(device_id));
			snap.used_handles = static_cast<uint64_t>(pool->used(device_id));
			snap.total_bytes = snap.total_handles * granularity;
			snap.available_bytes = snap.available_handles * granularity;
			snap.used_bytes = snap.used_handles * granularity;
			out.push_back(snap);
		}
	}
	return out;
}

std::vector<Daemon::ModelSnapshot> Daemon::snapshotModels(bool sync_message_space_from_shm) {
	std::vector<uint64_t> ids = model_manager_.listModelIds();
	std::vector<ModelSnapshot> out;
	out.reserve(ids.size());
	for (uint64_t id : ids) {
		ModelConfig* cfg = model_manager_.getModelConfig(id);
		if (!cfg) {
			continue;
		}
		if (sync_message_space_from_shm) {
			cfg->syncMessageSpaceFromShm();
		}
		const ModelMessageSpace& space = cfg->messageSpace();
		ModelSnapshot snap;
		snap.model_id = id;
		snap.state = cfg->getState();
		snap.message_state = space.state;
		snap.model_npuid = space.model_npuid;
		snap.model_osid = space.model_osid;
		snap.allocated_bytes = cfg->allocatedBytes();
		snap.allocated_handles = static_cast<uint64_t>(cfg->allocatedHandleCount());
		out.push_back(snap);
	}
	return out;
}

std::vector<std::string> Daemon::drainLogs() {
	return daemonDrainLogs();
}

Daemon::ModelSnapshot Daemon::getModelSnapshot(uint64_t model_id, bool sync_message_space_from_shm) {
	ModelSnapshot snap;
	snap.model_id = model_id;
	ModelConfig* cfg = model_manager_.getModelConfig(model_id);
	if (!cfg) {
		snap.state = ModelState::INVALID;
		snap.message_state = MessageState::INVALID;
		return snap;
	}
	if (sync_message_space_from_shm) {
		cfg->syncMessageSpaceFromShm();
	}
	const ModelMessageSpace& space = cfg->messageSpace();
	snap.state = cfg->getState();
	snap.message_state = space.state;
	snap.model_npuid = space.model_npuid;
	snap.model_osid = space.model_osid;
	snap.allocated_bytes = cfg->allocatedBytes();
	snap.allocated_handles = static_cast<uint64_t>(cfg->allocatedHandleCount());
	return snap;
}

/* Model manipulation */

ModelConfig* Daemon::getModelConfig(uint64_t model_id) {
	return model_manager_.getModelConfig(model_id);
}

bool Daemon::updateModelState(uint64_t model_id, ModelState new_state) {
	return model_manager_.updateModelState(model_id, new_state);
}

ModelState Daemon::getModelState(uint64_t model_id) const {
	return model_manager_.getModelState(model_id);
}

std::vector<uint64_t> Daemon::listModelIds() const {
	return model_manager_.listModelIds();
}

std::vector<int32_t> Daemon::getCurrentModelNpuids() const {
	return model_manager_.getCurrentModelNpuids();
}

ModelConfig* Daemon::registerModelFromMacro() {
	ModelConfig* cfg = model_manager_.registerModelFromMacro();
	if (cfg) {
		SetPidToShare(getCurrentModelNpuids());
		cfg->setState(ModelState::REGISTERED);
		cfg->messageSpace().state = MessageState::REGISTERED;
		cfg->messageSpace().heartbeat_ns = monotonicNowNs();
		cfg->syncMessageSpaceToShm();
	}
	return cfg;
}

/* Interactive operations between models and handles */

void Daemon::SetPidToShare(const std::vector<int32_t>& npuid_list) {
	// update all handle pools with the new pid list to share
	std::vector<HandlePool*> pools;
	{
		std::lock_guard<std::mutex> guard(daemon_mutex_);
		pools.reserve(handle_pools_.size());
		for (auto& pair : handle_pools_) {
			pools.push_back(pair.second.get());
		}
	}
	for (HandlePool* pool : pools) {
		for (int32_t device_id : pool->listDeviceIds()) {
			pool->memSetPidToShare(npuid_list, device_id);
		}
	}
}

bool Daemon::allocateHandlesForModel(ModelConfig& config) {
	/* --- Mechanism --- */
    /* req from handle_request_list -> get handle from pool -> put shareable_handle into handle_info_list
	   -> put handle into allocated_handles -> offset_ed=offset_st+num -> clear handle_request_list */
	// current logic is to allocate handles based on the handle_request_list in the model's message space;
	// larger granularities are preferred, and if unavailable it falls back to smaller granularities.
	ModelMessageSpace& space = config.messageSpace();
	const std::vector<request_granularity> requests = collectHandleRequests(space);

	std::vector<AllocatedHandle> allocated_handles;
	for (const auto& req : requests) {
		if (!allocateForRequest(req, allocated_handles)) {
			releaseAllocatedHandles(allocated_handles);
			space.offset_ed = space.offset_st;
			return false;
		}
	}
	space.offset_ed = space.offset_st + static_cast<int32_t>(allocated_handles.size());

	writeHandlesToMessageSpace(space, allocated_handles);
	// add in allocated_handles_ for later release when model is done
	for (auto& entry : allocated_handles) {
		config.addAllocatedHandle(entry.device_id, std::move(entry.handle));
	}

	// clear request list
	for (auto& req : space.handle_request_list) {
		if (req.request_num == 0 && req.granularity == 0 && req.device_id == -1) {
			break; // end of requests
		}
		req.request_num = 0;
		req.granularity = 0;
		req.device_id = -1;
	}
	return true;
}

std::vector<request_granularity> Daemon::collectHandleRequests(const ModelMessageSpace& space) const {
	std::vector<request_granularity> requests;
	requests.reserve(space.handle_request_list.size());
	for (const auto& req : space.handle_request_list) {
		if (req.request_num == 0 && req.granularity == 0 && req.device_id == -1) {
			break;
		}
		if (req.request_num > 0 && req.granularity > 0 && req.device_id >= 0) {
			requests.push_back(req);
		}
	}
	std::sort(requests.begin(), requests.end(), [](const request_granularity& a, const request_granularity& b) {
		return a.granularity > b.granularity;
	});
	return requests;
}

bool Daemon::allocateForRequest(const request_granularity& req,
										std::vector<AllocatedHandle>& allocated_handles) {
	const int32_t canonical_device_id = req.device_id;
	int32_t device_id = -1;
	const uint64_t granularity = req.granularity;
	if (!mapCanonicalToLocalDeviceId(canonical_device_id, device_id)) {
		daemonLogError("[Daemon] unknown canonical device id from model request: " +
			std::to_string(canonical_device_id) +
			", request_num=" + std::to_string(req.request_num) +
			", granularity=" + std::to_string(req.granularity));
		return false;
	}
	if (granularity == 0 || req.request_num == 0) {
		return false;
	}
	for (uint64_t idx = 0; idx < req.request_num; ++idx) {
		Handle handle;
		if (!tryAcquireHandleExact(granularity, device_id, handle)) {
			return false;
		}
		AllocatedHandle entry;
		entry.granularity = granularity;
		entry.device_id = canonical_device_id;
		entry.handle = std::move(handle);
		allocated_handles.push_back(std::move(entry));
	}
	return true;
}

bool Daemon::tryAcquireHandleExact(uint64_t granularity, int32_t device_id, Handle& out_handle) {
	HandlePool* pool = findHandlePool(granularity);
	if (!pool) {
		return false;
	}
	if (pool->available(device_id) == 0) {
		return false;
	}
	Handle handle = pool->acquire(device_id);
	if (!handle.valid()) {
		return false;
	}
	out_handle = std::move(handle);
	return true;
}

bool Daemon::tryAcquireHandle(uint64_t& granularity, int32_t device_id, Handle& out_handle) {
	while (granularity > 0) {
		HandlePool* pool = findHandlePool(granularity);
		if (!pool) {
			granularity /= 2;
			continue;
		}
		if (pool->available(device_id) == 0) {
			granularity /= 2;
			continue;
		}
		Handle handle = pool->acquire(device_id);
		if (!handle.valid()) {
			granularity /= 2;
			continue;
		}
		out_handle = std::move(handle);
		return true;
	}
	return false;
}

void Daemon::releaseAllocatedHandles(std::vector<AllocatedHandle>& allocated_handles) {
	for (auto& entry : allocated_handles) {
		HandlePool* pool = findHandlePool(entry.granularity);
		if (pool) {
			pool->release(entry.handle.deviceId(), std::move(entry.handle));
		}
	}
}

void Daemon::writeHandlesToMessageSpace(ModelMessageSpace& space,
										const std::vector<AllocatedHandle>& allocated_handles) {
	std::vector<const AllocatedHandle*> sorted;
	sorted.reserve(allocated_handles.size());
	for (const auto& entry : allocated_handles) {
		sorted.push_back(&entry);
	}
	std::sort(sorted.begin(), sorted.end(), [](const AllocatedHandle* a, const AllocatedHandle* b) {
		return a->granularity > b->granularity;
	});

	size_t out = static_cast<size_t>(std::max<int32_t>(0, space.offset_st));
	for (const auto* entry : sorted) {
		if (out >= space.handle_info_list.size()) {
			break;
		}
		space.handle_info_list[out].granularity = entry->granularity;
		space.handle_info_list[out].shareable_handle = entry->handle.shareableHandle();
		space.handle_info_list[out].device_id = entry->device_id;
		++out;
	}
	space.offset_ed = static_cast<int32_t>(out);
}

void Daemon::returnHandlesFromModel(ModelConfig& config) {
	/* Mechanism */
	/* get shareable_handle from handle_info_list -> find handle from allocated_handles_
	   -> release handle to pool -> erase from allocated_handles_ -> erase from handle_info_list 
	   -> offset_ed --*/
	// current logic is to return all handles from the model, and it relies on the model to correctly 
	// fill in the handle_info_list with the handles to be returned and their granularities and device_ids;
	ModelMessageSpace& space = config.messageSpace();
	// remove handles from offset_st to offset_ed in the handle_info_list
	const int32_t max_index = static_cast<int32_t>(space.handle_info_list.size());
	int32_t start = std::max<int32_t>(0, space.offset_st);
	int32_t end = std::max<int32_t>(start, space.offset_ed);
	if (end > max_index) {
		end = max_index;
	}
	for (int32_t idx = end - 1; idx > start - 1; --idx) {
		const handle_granularity& handle_info = space.handle_info_list[idx];
		const uint64_t shareable_handle = handle_info.shareable_handle;
		const int32_t device_id = handle_info.device_id;
		Handle handle;
		if (!config.takeAllocatedHandle(device_id, shareable_handle, handle)) {
			daemonLogError("[Daemon] returnHandlesFromModel miss: model_id=" +
				std::to_string(config.getModelId()) +
				", idx=" + std::to_string(idx) +
				", device_id=" + std::to_string(device_id) +
				", shareable_handle=" + std::to_string(shareable_handle));
			continue;
		}

		if (!handle.valid()) {
			daemonLogError("[Daemon] returnHandlesFromModel invalid handle: model_id=" +
				std::to_string(config.getModelId()) +
				", idx=" + std::to_string(idx) +
				", device_id=" + std::to_string(device_id) +
				", shareable_handle=" + std::to_string(shareable_handle));
			continue;
		}

		HandlePool* pool = findHandlePool(handle.granularity());
		if (!pool) {
			daemonLogError("[Daemon] returnHandlesFromModel missing pool: model_id=" +
				std::to_string(config.getModelId()) +
				", granularity=" + std::to_string(handle.granularity()) +
				", device_id=" + std::to_string(handle.deviceId()) +
				", shareable_handle=" + std::to_string(handle.shareableHandle()));
			continue;
		}

		pool->release(handle.deviceId(), std::move(handle));
		// erase handle info
		space.handle_info_list[idx].granularity = 0;
		space.handle_info_list[idx].shareable_handle = std::numeric_limits<uint64_t>::max();
		space.handle_info_list[idx].device_id = -1;
		space.offset_ed = idx;
	}
}

void Daemon::returnHandlesFromCfg(ModelConfig& config) {
	std::vector<Handle> handles = config.drainAllocatedHandles();
	for (auto& handle : handles) {
		if (!handle.valid()) {
			continue;
		}
		HandlePool* pool = findHandlePool(handle.granularity());
		if (pool) {
			pool->release(handle.deviceId(), std::move(handle));
		}
	}
}

void Daemon::runOnce() {
	const auto now = std::chrono::steady_clock::now();
	const bool should_check_liveness = now >= next_liveness_check_;
	if (should_check_liveness) {
		next_liveness_check_ = now + std::chrono::seconds(1);
	}

	const int64_t now_ns = monotonicNowNs();

	auto model_ids = model_manager_.listModelIds();
	std::vector<ModelConfig*> active_configs;
	active_configs.reserve(model_ids.size() + 1);

	// Sweep existing models for exit or completion
	for (uint64_t id : model_ids) {
		ModelConfig* cfg = model_manager_.getModelConfig(id);
		if (!cfg) {
			continue;
		}

		if (should_check_liveness) {
			if (!cfg->tryLockMessageSpace()) {
				active_configs.push_back(cfg);
				continue;
			}
			cfg->syncMessageSpaceFromShm();
			ModelMessageSpace& space = cfg->messageSpace();
			const ModelState state = cfg->getState();
			if (state == ModelState::DONE || state == ModelState::INVALID) {
				daemonLogDebug("[Daemon] cleaning up model " + std::to_string(id) +
					" with state " + std::to_string(static_cast<int>(state)));
				cfg->unlockMessageSpace();
				returnHandlesFromCfg(*cfg);
				model_manager_.removeModelConfig(id);
				if (daemonDebugLogsEnabled()) {
					printAll();
				}
				continue;
			}

			if (heartbeatTimedOut(space.heartbeat_ns, now_ns)) {
				const int64_t age_ns = (space.heartbeat_ns > 0 && now_ns > space.heartbeat_ns)
					? (now_ns - space.heartbeat_ns)
					: -1;
				daemonLogInfo("[Daemon] heartbeat timeout for model " + std::to_string(id) +
					" (npuid=" + std::to_string(space.model_npuid) +
					", osid=" + std::to_string(space.model_osid) +
					", age_ns=" + std::to_string(age_ns) + ")");
				cfg->setState(ModelState::DONE);
				cfg->unlockMessageSpace();
				returnHandlesFromCfg(*cfg);
				model_manager_.removeModelConfig(id);
				if (daemonDebugLogsEnabled()) {
					printAll();
				}
				continue;
			}
			cfg->unlockMessageSpace();
		}

		active_configs.push_back(cfg);
	}

	// Handle new registrations from the macro space
	if (ModelConfig* cfg = registerModelFromMacro()) {
		(void)cfg; // already handled inside registerModelFromMacro
		daemonLogInfo("[Daemon] registered new model with id " + std::to_string(cfg->getModelId()) 
			+ ", npuid " + std::to_string(cfg->messageSpace().model_npuid) + ", osid " 
			+ std::to_string(cfg->messageSpace().model_osid));
		active_configs.push_back(cfg);

		if (daemonDebugLogsEnabled()) {
			printAll();
		}
	}

	// traverse the model MessageSpace for each model to check if there are requests from model
	for (ModelConfig* cfg : active_configs) {
		if (!cfg) {
			continue;
		}
		if (!cfg->tryLockMessageSpace()) {
			continue;
		}
		const uint64_t id = cfg->getModelId();
		cfg->syncMessageSpaceFromShm();
		ModelMessageSpace& space = cfg->messageSpace();
		bool changed = false;
		if (space.state == MessageState::REQUEST_GET_HANDLES) {
			// model is requesting handles
			// daemonLogDebug("[Daemon] model " + std::to_string(id) + " REQUEST_GET_HANDLES");
			try {
				if (allocateHandlesForModel(*cfg)) {
					space.state = MessageState::HANDLES_READY;
					changed = true;
					daemonLogDebug("[Daemon] allocated handles for model " + std::to_string(id));
				}
			} catch (const std::exception& ex) {
				daemonLogError("[Daemon] exception during REQUEST_GET_HANDLES for model " +
					std::to_string(id) + ": " + ex.what());
			} catch (...) {
				daemonLogError("[Daemon] unknown exception during REQUEST_GET_HANDLES for model " +
					std::to_string(id));
			}
		} else if (space.state == MessageState::REQUEST_RETURN_HANDLES) {
			// model is returning handles
			// daemonLogDebug("[Daemon] model " + std::to_string(id) + " REQUEST_RETURN_HANDLES");
			try {
				returnHandlesFromModel(*cfg);
				space.state = MessageState::HANDLES_RETURNED;
				changed = true;
				daemonLogDebug("[Daemon] returned handles for model " + std::to_string(id));
			} catch (const std::exception& ex) {
				daemonLogError("[Daemon] exception during REQUEST_RETURN_HANDLES for model " +
					std::to_string(id) + ": " + ex.what());
			} catch (...) {
				daemonLogError("[Daemon] unknown exception during REQUEST_RETURN_HANDLES for model " +
					std::to_string(id));
			}
		}
		if (changed) {
			cfg->syncMessageSpaceToShm();
			if (daemonDebugLogsEnabled()) {
				printAll();
			}
		}
		cfg->unlockMessageSpace();
	}
}

void Daemon::runLoop() {
	const int64_t loop_interval_us = daemonLoopIntervalUs();
	while (running_.load()) {
		runOnce();
		std::this_thread::sleep_for(std::chrono::microseconds(loop_interval_us));
	}
}

} // namespace mdaemon