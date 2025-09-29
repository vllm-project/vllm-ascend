#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <utility>

struct bfloat16;

namespace kvmem {

struct BlockView {
    void*  ptr = nullptr;
    size_t bytes = 0;
};

enum class ScalarType : uint8_t {
    BF16 = 0,
    F16,
    F32,
    F64,
    I8,
    I16,
    I32,
    I64
};

size_t scalar_type_bytes(ScalarType t);

//  Initialize P via mmap (single-node mode)
bool init_p_side_from_mmap(const std::string& path,
                           ScalarType dtype,
                           size_t offset_dtype,
                           const std::string& ip_base,
                           uint16_t base_port,
                           size_t start_offset = 0);

//  Initialize D via mmap
bool init_d_side_from_mmap(const std::string& path,
                           ScalarType dtype,
                           size_t offset_dtype,
                           size_t start_offset = 0);

void shutdown_all();

// P-side
bool p_get_block(uint32_t node_id, uint64_t p_block_id, BlockView& out);

// D-side
bool d_get_block(uint64_t d_block_id, BlockView& out);

// mode=1: direct IDs – block_id = id_base + logical_index, no registration needed
bool d_set_id_mode(uint32_t mode, uint64_t id_base);

// introspection
size_t p_num_nodes();
size_t p_blocks_per_node();
size_t p_block_bytes();
size_t d_total_blocks();
size_t d_block_bytes();

} // namespace kvmem