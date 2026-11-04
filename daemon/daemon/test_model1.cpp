#include "inc/model_management.hpp"
#include "inc/utils.h"

#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

// define colors for better visibility in test output
#define COLOR_RESET "\033[0m"
#define COLOR_RED "\033[31m"
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_BLUE "\033[34m"
#define COLOR_MAGENTA "\033[35m"

using namespace mdaemon;

static void write_macro(ModelMacroSpace* macro, int32_t npuid) {
    macro->current_model_npuid = npuid;
    macro->current_model_osid = static_cast<int32_t>(::getpid());
    macro->setShmName(defaultShmName());
    ::msync(macro, sizeof(ModelMacroSpace), MS_SYNC);
    std::cout << "[Model1] wrote npuid=" << npuid << " osid=" << macro->current_model_osid << "\n";
}

static void print_message_space(const ModelMessageSpace& space, const char* tag) {
    std::cout << COLOR_BLUE << "[Model1] " << tag << COLOR_RESET
              << " state=" << static_cast<int>(space.state)
              << " npuid=" << space.model_npuid << " osid=" << space.model_osid
              << " offset_st=" << space.offset_st << " offset_ed=" << space.offset_ed << "\n";
    const int32_t end = std::min<int32_t>(space.offset_ed,
                                          static_cast<int32_t>(space.handle_info_list.size()));
    for (int32_t i = space.offset_st; i < end; ++i) {
        const handle_granularity& info = space.handle_info_list[i];
        std::cout << "  handle[" << i << "] shareable=" << info.shareable_handle
                  << " granularity=" << (info.granularity / (1024 * 1024))
                  << "MB device=" << info.device_id << "\n";
    }
    std::cout << "------------------------------------------------\n";
}

static bool wait_for_state(ModelMessageSpace* space, MessageState target, int timeout_ms) {
    const int step_ms = 50;
    int waited = 0;
    while (space->state != target && waited < timeout_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(step_ms));
        waited += step_ms;
    }
    return space->state == target;
}

static void clear_requests(ModelMessageSpace* space) {
    for (auto& req : space->handle_request_list) {
        if (req.request_num == 0 && req.granularity == 0 && req.device_id == -1) {
            break; // end of requests
        }
        req.request_num = 0;
        req.granularity = 0;
        req.device_id = -1;
    }
}

static void clear_handles(ModelMessageSpace* space) {
    for (auto& info : space->handle_info_list) {
        if (info.shareable_handle == std::numeric_limits<uint64_t>::max() &&
            info.granularity == 0 &&
            info.device_id == -1) {
            break; // end of handles
        }
        info.shareable_handle = std::numeric_limits<uint64_t>::max();
        info.granularity = 0;
        info.device_id = -1;
    }
    space->offset_st = 0;
    space->offset_ed = 0;
}

int main() {
    int32_t device_id = 0; // for testing we just use device 0
    aclInit(nullptr);
    ensure_context(device_id);
    int32_t kNpuid;
    CHECK_ERROR(aclrtDeviceGetBareTgid(&kNpuid));

    sem_t* sem = ::sem_open(MODEL_SEMAPHORE_NAME, O_CREAT, 0666, 1);
    if (sem == SEM_FAILED) {
        std::cerr << COLOR_RED << "[Model1] sem_open failed\n" << COLOR_RESET;
        return 1;
    }
    if (::sem_wait(sem) == -1) {
        std::cerr << COLOR_RED << "[Model1] sem_wait failed\n" << COLOR_RESET;
        ::sem_close(sem);
        return 1;
    }

    int fd = ::shm_open(MODEL_REG_SHM, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        std::cerr << COLOR_RED << "[Model1] failed to open MODEL_REG_SHM\n" << COLOR_RESET;
        ::sem_post(sem);
        ::sem_close(sem);
        return 1;
    }
    if (::ftruncate(fd, sizeof(ModelMacroSpace)) == -1) {
        std::cerr << COLOR_RED << "[Model1] ftruncate failed\n" << COLOR_RESET;
        ::close(fd);
        ::sem_post(sem);
        ::sem_close(sem);
        return 1;
    }
    void* region = ::mmap(nullptr, sizeof(ModelMacroSpace), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (region == MAP_FAILED) {
        std::cerr << COLOR_RED << "[Model1] mmap failed\n" << COLOR_RESET;
        ::close(fd);
        ::sem_post(sem);
        ::sem_close(sem);
        return 1;
    }
    auto* macro = static_cast<ModelMacroSpace*>(region);
    if (isDefaultShmName(macro->getShmName())) {
        write_macro(macro, kNpuid);
    }

    // wait for daemon to fill shm_name
    while (isDefaultShmName(macro->getShmName())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::string shm_name(macro->getShmName());
    std::cout << "[Model1] got shm_name=" << shm_name << "\n";

    macro->reset();
    ::msync(macro, sizeof(ModelMacroSpace), MS_SYNC);
    ::munmap(region, sizeof(ModelMacroSpace));
    ::close(fd);
    ::sem_post(sem);
    ::sem_close(sem);


    // open per-model message space
    int msg_fd = ::shm_open(shm_name.c_str(), O_RDWR, 0666);
    if (msg_fd == -1) {
        std::cerr << COLOR_RED << "[Model1] failed to open model shm: " << shm_name << "\n" << COLOR_RESET;
        return 1;
    }
    void* msg_region = ::mmap(nullptr, sizeof(ModelMessageSpace), PROT_READ | PROT_WRITE, MAP_SHARED, msg_fd, 0);
    if (msg_region == MAP_FAILED) {
        std::cerr << COLOR_RED << "[Model1] mmap failed for model shm\n" << COLOR_RESET;
        ::close(msg_fd);
        return 1;
    }
    auto* space = static_cast<ModelMessageSpace*>(msg_region);
    print_message_space(*space, "after registration");
    while (space->state != MessageState::REGISTERED) {
        std::cerr << COLOR_RED << "[Model1] expected REGISTERED state, got " << static_cast<int>(space->state) << "\n" << COLOR_RESET;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // request small handle set: 3x16MB, 4x8MB, 2x4MB, 1x2MB
    clear_requests(space);
    clear_handles(space);
    space->handle_request_list[0] = {3, 16ULL * 1024 * 1024, device_id};
    space->handle_request_list[1] = {4, 8ULL * 1024 * 1024, device_id};
    space->handle_request_list[2] = {2, 4ULL * 1024 * 1024, device_id};
    space->handle_request_list[3] = {1, 2ULL * 1024 * 1024, device_id};
    space->state = MessageState::REQUEST_GET_HANDLES;
    print_message_space(*space, "request handles (small)");
    if (!wait_for_state(space, MessageState::HANDLES_READY, 10000)) {
        std::cerr << COLOR_RED << "[Model1] timed out waiting for HANDLES_READY\n" << COLOR_RESET;
    }
    print_message_space(*space, "handles ready (small)");

    std::vector<handle_granularity> allocated_handles;
    int32_t end = std::min<int32_t>(space->offset_ed,
                                          static_cast<int32_t>(space->handle_info_list.size()));
    for (int32_t i = space->offset_st; i < end; ++i) {
        allocated_handles.push_back(space->handle_info_list[i]);
    }
    space->offset_ed = space->offset_st; // reset offset_ed after reading
    std::cout << "[Model1] allocated handles:\n";
    for (const auto& info : allocated_handles) {
        std::cout << "  shareable=" << info.shareable_handle
                  << " granularity=" << (info.granularity / (1024 * 1024))
                  << "MB device=" << info.device_id << "\n";
    }

    // request large handles: 20x16MB
    clear_requests(space);
    clear_handles(space);
    space->handle_request_list[0] = {20, 16ULL * 1024 * 1024, device_id};
    space->state = MessageState::REQUEST_GET_HANDLES;
    print_message_space(*space, "request handles (large)");
    if (!wait_for_state(space, MessageState::HANDLES_READY, 10000)) {
        std::cerr << COLOR_RED << "[Model1] timed out waiting for HANDLES_READY (large)\n" << COLOR_RESET;
    }
    print_message_space(*space, "handles ready (large)");

    end = std::min<int32_t>(space->offset_ed, static_cast<int32_t>(space->handle_info_list.size()));
    for (int32_t i = space->offset_st; i < end; ++i) {
        allocated_handles.push_back(space->handle_info_list[i]);
    }
    space->offset_ed = space->offset_st; // reset offset_ed after reading
    std::cout << "[Model1] allocated handles (small + large):\n";
    for (const auto& info : allocated_handles) {
        std::cout << "  shareable=" << info.shareable_handle
                  << " granularity=" << (info.granularity / (1024 * 1024))
                  << "MB device=" << info.device_id << "\n";
    }

    // return first 15 handles
    clear_requests(space);
    clear_handles(space);
    for (size_t i = 0; i < 15 && i < allocated_handles.size(); ++i) {
        const auto& info = allocated_handles[i];
        space->handle_info_list[space->offset_ed] = info;
        ++space->offset_ed;
    }
    allocated_handles.erase(allocated_handles.begin(), allocated_handles.begin() + std::min<size_t>(15, allocated_handles.size()));
    space->state = MessageState::REQUEST_RETURN_HANDLES;
    print_message_space(*space, "return handles (first 15)");
    if (!wait_for_state(space, MessageState::HANDLES_RETURNED, 10000)) {
        std::cerr << COLOR_RED << "[Model1] timed out waiting for HANDLES_RETURNED\n" << COLOR_RESET;
    }
    print_message_space(*space, "handles returned (first 15)");

    // return the rest of handles
    clear_requests(space);
    clear_handles(space);
    for (const auto& info : allocated_handles) {
        space->handle_info_list[space->offset_ed] = info;
        ++space->offset_ed;
    }
    space->state = MessageState::REQUEST_RETURN_HANDLES;
    print_message_space(*space, "return handles (all the rest)");
    if (!wait_for_state(space, MessageState::HANDLES_RETURNED, 10000)) {
        std::cerr << COLOR_RED << "[Model1] timed out waiting for HANDLES_RETURNED (all the rest)\n" << COLOR_RESET;
    }
    print_message_space(*space, "handles returned (all the rest)");


    ::munmap(msg_region, sizeof(ModelMessageSpace));
    ::close(msg_fd);

    // random sleep 5-10s before exit
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(5, 10);
    int wait_s = dist(rng);
    std::cout << COLOR_GREEN << "[Model1] exiting after " << wait_s << "s\n" << COLOR_RESET;
    std::this_thread::sleep_for(std::chrono::seconds(wait_s));
    return 0;
}
