#include "kv_mem_resolver.h"
#include "common.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <iostream>
#include <string>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#if !defined(_MSC_VER)
#include <stdlib.h>
#endif

namespace kvmem {

namespace {

struct PNodePool {
    void*  base     = nullptr;
    size_t map_len  = 0;
    bool   own_map  = false;
    std::vector<void*> blocks;
};

struct State {
    // P side
    bool p_inited = false;
    uint32_t p_num_nodes = 0;
    uint32_t p_blocks_per_node = 0;
    size_t   p_block_bytes_sz = 0;
    ScalarType p_dtype = ScalarType::BF16;
    std::string ip_base = "127.0.0.1";
    uint16_t base_port = 15000;
    std::vector<PNodePool> p_nodes;

    // D side
    bool d_inited = false;
    size_t d_total_blocks_cnt = 0;    // logic visible block number（applied start_offset）
    size_t d_block_bytes_sz   = 0;
    ScalarType d_dtype = ScalarType::BF16;
    void*  d_base = nullptr;     // mmap base address
    size_t d_map_len = 0;
    bool   d_own_map = false;

    std::vector<void*>    d_blocks;   
    std::vector<bool>     d_used;     
    std::vector<uint64_t> d_ids;     
    uint64_t next_d_id = 1000;

    // 0 = managed ID: lookup via d_ids / d_used
    // 1 = direct ID: compute as block_id = d_id_base + logical_index
    uint32_t d_id_mode = 1;
    uint64_t d_id_base = 0;

    std::mutex mu;
};

State& st() {
    static State s;
    return s;
}

size_t elem_size(ScalarType t) {
    switch (t) {
        case ScalarType::BF16: return 2;
        case ScalarType::F16:  return 2;
        case ScalarType::F32:  return 4;
        case ScalarType::F64:  return 8;
        case ScalarType::I8:   return 1;
        case ScalarType::I16:  return 2;
        case ScalarType::I32:  return 4;
        case ScalarType::I64:  return 8;
        default: return 1;
    }
}

static bool get_stable_size(int fd, size_t& out_len, int retries = 50, int sleep_ms = 10) {
    // Two consecutive identical readings are considered stable; increase if tighter stability is required.
    off_t prev = -1;
    for (int i = 0; i < retries; ++i) {
        struct stat stbuf{};
        if (fstat(fd, &stbuf) != 0) {
            std::cerr << "[kvmem] fstat failed: errno=" << strerror(errno) << std::endl;
            return false;
        }
        if (stbuf.st_size > 0 && stbuf.st_size == prev) {
            out_len = static_cast<size_t>(stbuf.st_size);
            return true;
        }
        prev = stbuf.st_size;
        usleep(sleep_ms * 1000);
    }
    // finally get the size
    struct stat stbuf{};
    if (fstat(fd, &stbuf) != 0) return false;
    out_len = static_cast<size_t>(stbuf.st_size);
    return out_len > 0;
}

void* mmap_file_ro_rw(const std::string& path, size_t& out_len, bool write = true) {
    int flags = write ? O_RDWR : O_RDONLY;
    int fd = ::open(path.c_str(), flags | O_CLOEXEC, 0600);
    if (fd < 0) {
        std::cerr << "[kvmem] open failed: " << path << " err=" << strerror(errno) << std::endl;
        return nullptr;
    }

    size_t stable_len = 0;
    if (!get_stable_size(fd, stable_len)) {
        std::cerr << "[kvmem] size not stable or invalid for: " << path << std::endl;
        ::close(fd);
        return nullptr;
    }

    int prot = PROT_READ | (write ? PROT_WRITE : 0);
    void* base = ::mmap(nullptr, stable_len, prot, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::cerr << "[kvmem] mmap failed: " << path << " err=" << strerror(errno)
                  << " size=" << stable_len << std::endl;
        ::close(fd);
        return nullptr;
    }
    ::close(fd);
    out_len = stable_len;

    std::cerr << "[kvmem] mmap ok pid=" << getpid()
              << " path=" << path
              << " size=" << out_len
              << " write=" << (write ? 1 : 0) << std::endl;
    return base;
}

void aligned_region_free(void* p) {
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

}


// Initialize P via mmap (single-node mode)
bool init_p_side_from_mmap(const std::string& path,
                           ScalarType dtype,
                           size_t offset_dtype,
                           const std::string& ip_base,
                           uint16_t base_port,
                           size_t start_offset) {
    auto& S = st();
    std::lock_guard<std::mutex> g(S.mu);
    if (S.p_inited) return true;

    size_t esz = elem_size(dtype);
    if (esz == 0 || offset_dtype == 0) {
        std::cerr << "[kvmem] init_p_side_from_mmap: invalid dtype/offset" << std::endl;
        return false;
    }

    size_t map_len = 0;
    void* base = mmap_file_ro_rw(path, map_len, /*write*/true);
    if (!base) {
        std::cerr << "Failed to map file '" << path << "': " << std::strerror(errno) << std::endl;
        return false;
    }

    const size_t block_bytes = offset_dtype * esz;
    // if (block_bytes == 0 || map_len % block_bytes != 0) {
    //     std::cerr << "[kvmem] P mmap size not multiple of block_bytes: file=" << map_len
    //               << " block_bytes=" << block_bytes << std::endl;
    //     ::munmap(base, map_len);
    //     return false;
    // }
    const size_t total_blocks_in_file = map_len / block_bytes;
    if (start_offset >= total_blocks_in_file) {
        std::cerr << "[kvmem] P start_offset out of range" << std::endl;
        ::munmap(base, map_len);
        return false;
    }
    const uint32_t blocks = static_cast<uint32_t>(total_blocks_in_file - start_offset);

    S.p_num_nodes = 1;
    S.p_blocks_per_node = blocks;
    S.p_block_bytes_sz = block_bytes;
    S.p_dtype = dtype;
    S.ip_base = ip_base;
    S.base_port = base_port;

    S.p_nodes.resize(1);
    auto& pool = S.p_nodes[0];
    pool.base = base;
    pool.map_len = map_len;
    pool.own_map = true;
    pool.blocks.resize(blocks, nullptr);
    for (uint32_t b = 0; b < blocks; ++b) {
        pool.blocks[b] = static_cast<char*>(base) + static_cast<size_t>(start_offset + b) * block_bytes;
    }
    S.p_inited = true;
    std::cout << "[kvmem] P initialized from mmap: blocks=" << blocks
              << " block_bytes=" << block_bytes << " path=" << path
              << " start_offset=" << start_offset << std::endl;
    return true;
}

// Initialize D via mmap
bool init_d_side_from_mmap(const std::string& path,
                           ScalarType dtype,
                           size_t offset_dtype,
                           size_t start_offset) {
    auto& S = st();
    std::lock_guard<std::mutex> g(S.mu);
    if (S.d_inited) return true;

    size_t esz = elem_size(dtype);
    if (esz == 0 || offset_dtype == 0) {
        std::cerr << "[kvmem] init_d_side_from_mmap: invalid dtype/offset" << std::endl;
        return false;
    }
    size_t map_len = 0;
    void* base = mmap_file_ro_rw(path, map_len, /*write*/true);
    if (!base) {
        std::cerr << "Failed to map file '" << path << "': " << std::strerror(errno) << std::endl;
        return false;
    }

    const size_t block_bytes = offset_dtype * esz;
    // if (block_bytes == 0 || map_len % block_bytes != 0) {
    //     std::cerr << "[kvmem] D mmap size not multiple of block_bytes: file=" << map_len
    //               << " block_bytes=" << block_bytes << std::endl;
    //     ::munmap(base, map_len);
    //     return false;
    // }
    const size_t total_blocks_in_file = map_len / block_bytes;
    if (start_offset >= total_blocks_in_file) {
        std::cerr << "[kvmem] D start_offset out of range" << std::endl;
        ::munmap(base, map_len);
        return false;
    }
    const size_t blocks = total_blocks_in_file - start_offset;

    S.d_total_blocks_cnt = blocks;
    S.d_block_bytes_sz   = block_bytes;
    S.d_dtype = dtype;
    S.d_base = base;
    S.d_map_len = map_len;
    S.d_own_map = true;

    S.d_blocks.resize(blocks, nullptr);
    S.d_used.resize(blocks, false);
    S.d_ids.resize(blocks, UINT64_MAX);
    for (size_t i = 0; i < blocks; ++i) {
        S.d_blocks[i] = static_cast<char*>(base) + (start_offset + i) * block_bytes;
    }

    // Default: managed-ID mode. Switch to direct-ID by calling d_set_id_mode(1, base_id).
    S.d_id_mode = 0;
    S.d_id_base = 0;

    S.next_d_id = 1000;
    S.d_inited = true;
    std::cout << "[kvmem] D initialized from mmap: blocks=" << blocks
              << " block_bytes=" << block_bytes << " path=" << path
              << " start_offset=" << start_offset << std::endl;
    return true;
}

void shutdown_all() {
    auto& S = st();
    std::lock_guard<std::mutex> g(S.mu);

    if (S.p_inited) {
        for (auto& pool : S.p_nodes) {
            if (pool.own_map && pool.base && pool.map_len) {
                ::munmap(pool.base, pool.map_len);
            } else {
                for (void* p : pool.blocks) if (p) aligned_region_free(p);
            }
            pool.base = nullptr; pool.map_len = 0; pool.own_map = false;
            pool.blocks.clear();
        }
        S.p_nodes.clear();
        S.p_inited = false;
    }

    if (S.d_inited) {
        if (S.d_own_map && S.d_base && S.d_map_len) {
            ::munmap(S.d_base, S.d_map_len);
        } else {
            for (void* p : S.d_blocks) if (p) aligned_region_free(p);
        }
        S.d_blocks.clear();
        S.d_used.clear();
        S.d_ids.clear();
        S.d_base = nullptr; S.d_map_len = 0; S.d_own_map = false;
        S.d_inited = false;
    }
}

// set id mode
bool d_set_id_mode(uint32_t mode, uint64_t id_base) {
    auto& S = st();
    std::lock_guard<std::mutex> g(S.mu);
    if (!S.d_inited) {
        std::cerr << "[kvmem] d_set_id_mode: D not initialized" << std::endl;
        return false;
    }
    if (mode > 1) {
        std::cerr << "[kvmem] d_set_id_mode: invalid mode=" << mode << std::endl;
        return false;
    }
    S.d_id_mode = mode;
    S.d_id_base = id_base;
    std::cout << "[kvmem] D id_mode=" << mode << " id_base=" << id_base << std::endl;
    return true;
}

bool p_get_block(uint32_t node_id, uint64_t p_block_id, BlockView& out) {
    auto& S = st();
    std::lock_guard<std::mutex> g(S.mu);
    if (!S.p_inited) {
        return false;
    }
    if (node_id >= S.p_num_nodes) return false;
    if (p_block_id >= S.p_blocks_per_node) return false;
    void* ptr = S.p_nodes[node_id].blocks[static_cast<size_t>(p_block_id)];
    out.ptr = ptr;
    out.bytes = S.p_block_bytes_sz;
    return ptr != nullptr;
}


bool d_get_block(uint64_t d_block_id, BlockView& out) {
    auto& S = st();
    std::lock_guard<std::mutex> g(S.mu);
    if (!S.d_inited) {
        return false;
    }

    if (S.d_id_mode == 1) {
        // direct ID：block_id = id_base + logical_index
        if (d_block_id < S.d_id_base) return false;
        uint64_t idx64 = d_block_id - S.d_id_base;
        if (idx64 >= S.d_total_blocks_cnt) return false;
        size_t idx = static_cast<size_t>(idx64);
        out.ptr = S.d_blocks[idx];
        out.bytes = S.d_block_bytes_sz;
        return out.ptr != nullptr;
    }

    // managed ID：search by mapping
    for (size_t i = 0; i < S.d_total_blocks_cnt; ++i) {
        if (S.d_used[i] && S.d_ids[i] == d_block_id) {
            out.ptr = S.d_blocks[i];
            out.bytes = S.d_block_bytes_sz;
            return true;
        }
    }
    return false;
}

// introspection
size_t p_num_nodes()        { return st().p_num_nodes; }
size_t p_blocks_per_node()  { return st().p_blocks_per_node; }
size_t p_block_bytes()      { return st().p_block_bytes_sz; }
size_t d_total_blocks()     { return st().d_total_blocks_cnt; }
size_t d_block_bytes()      { return st().d_block_bytes_sz; }

} // namespace kvmem