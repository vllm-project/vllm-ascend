#include "inc/model_management.hpp"
#include "inc/utils.h"

#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <random>
#include <semaphore.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>

using namespace mdaemon;

static void write_macro(ModelMacroSpace* macro, int32_t npuid) {
    macro->current_model_npuid = npuid;
    macro->current_model_osid = static_cast<int32_t>(::getpid());
    macro->setShmName(defaultShmName());
    ::msync(macro, sizeof(ModelMacroSpace), MS_SYNC);
    std::cout << "[Model3] wrote npuid=" << npuid << " osid=" << macro->current_model_osid << "\n";
}

int main() {
    constexpr int32_t kNpuid = 3003;
    sem_t* sem = ::sem_open(MODEL_SEMAPHORE_NAME, O_CREAT, 0666, 1);
    if (sem == SEM_FAILED) {
        std::cerr << "[Model3] sem_open failed\n";
        return 1;
    }
    if (::sem_wait(sem) == -1) {
        std::cerr << "[Model3] sem_wait failed\n";
        ::sem_close(sem);
        return 1;
    }

    int fd = ::shm_open(MODEL_REG_SHM, O_RDWR | O_CREAT, 0666);
    if (fd == -1) {
        std::cerr << "[Model3] failed to open MODEL_REG_SHM\n";
        ::sem_post(sem);
        ::sem_close(sem);
        return 1;
    }
    if (::ftruncate(fd, sizeof(ModelMacroSpace)) == -1) {
        std::cerr << "[Model3] ftruncate failed\n";
        ::close(fd);
        ::sem_post(sem);
        ::sem_close(sem);
        return 1;
    }
    void* region = ::mmap(nullptr, sizeof(ModelMacroSpace), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (region == MAP_FAILED) {
        std::cerr << "[Model3] mmap failed\n";
        ::close(fd);
        ::sem_post(sem);
        ::sem_close(sem);
        return 1;
    }
    auto* macro = static_cast<ModelMacroSpace*>(region);
    if (isDefaultShmName(macro->getShmName())) {
        write_macro(macro, kNpuid);
    }

    while (isDefaultShmName(macro->getShmName())) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::string shm_name(macro->getShmName());
    std::cout << "[Model3] got shm_name=" << shm_name << "\n";

    macro->reset();
    ::msync(macro, sizeof(ModelMacroSpace), MS_SYNC);
    ::munmap(region, sizeof(ModelMacroSpace));
    ::close(fd);
    ::sem_post(sem);
    ::sem_close(sem);

    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(5, 10);
    int wait_s = dist(rng);
    std::cout << "[Model3] exiting after " << wait_s << "s\n";
    std::this_thread::sleep_for(std::chrono::seconds(wait_s));
    return 0;
}
