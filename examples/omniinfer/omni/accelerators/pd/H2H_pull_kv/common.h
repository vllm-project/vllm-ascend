#pragma once

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <cstdint>
#include <set>
#include <cstring>
#include <sys/uio.h>

// BFloat16 type definition
struct bfloat16 {
    uint16_t data;

    bfloat16() : data(0) {}

    explicit bfloat16(float f) {
        union { float f; uint32_t i; } u;
        u.f = f;
        data = static_cast<uint16_t>(u.i >> 16);
    }

    explicit operator float() const {
        union { float f; uint32_t i; } u;
        u.i = static_cast<uint32_t>(data) << 16;
        return u.f;
    }
};

// KV block structure
struct KVBlock {
    uint64_t block_id;     // Global block ID on D side
    void* memory_ptr;
    size_t memory_size;

    KVBlock() : block_id(0), memory_ptr(nullptr), memory_size(0) {}
    KVBlock(uint64_t id, void* ptr, size_t size) : block_id(id), memory_ptr(ptr), memory_size(size) {}
    KVBlock(const KVBlock& other) : block_id(other.block_id), memory_ptr(other.memory_ptr), memory_size(other.memory_size) {}
    KVBlock& operator=(const KVBlock& other) {
        if (this != &other) { block_id = other.block_id; memory_ptr = other.memory_ptr; memory_size = other.memory_size; }
        return *this;
    }
    KVBlock(KVBlock&& other) noexcept : block_id(other.block_id), memory_ptr(other.memory_ptr), memory_size(other.memory_size) {
        other.block_id = 0; other.memory_ptr = nullptr; other.memory_size = 0;
    }
    KVBlock& operator=(KVBlock&& other) noexcept {
        if (this != &other) { block_id = other.block_id; memory_ptr = other.memory_ptr; memory_size = other.memory_size;
            other.block_id = 0; other.memory_ptr = nullptr; other.memory_size = 0; }
        return *this;
    }
};

// Mapping from P-side to D-side
struct PtoDMapping {
    uint32_t p_node_id;
    uint64_t p_block_id;    // Local block ID on P side
    uint64_t d_block_id;    // Global block ID on D side
    PtoDMapping() : p_node_id(0), p_block_id(0), d_block_id(0) {}
    PtoDMapping(uint32_t node_id, uint64_t p_id, uint64_t d_id) : p_node_id(node_id), p_block_id(p_id), d_block_id(d_id) {}
};

// Refactored KV transfer request
struct KVRequest {
    std::string request_id;
    std::vector<KVBlock> d_blocks;                    // D-side memory blocks
    std::vector<PtoDMapping> block_mappings;          // Mapping from P side to D side
    std::function<void(bool, const std::string&)> callback;

    KVRequest() = default;
    explicit KVRequest(const std::string& id) : request_id(id) {}

    std::vector<uint32_t> getAllPNodeIds() const {
        std::set<uint32_t> node_ids;
        for (const auto& mapping : block_mappings) node_ids.insert(mapping.p_node_id);
        return std::vector<uint32_t>(node_ids.begin(), node_ids.end());
    }
};

// Protocol constants
const uint32_t PROTOCOL_MAGIC = 0x4B56434B; // "KVCK"
const uint8_t PROTOCOL_VERSION = 1;

enum class MessageType : uint8_t { REQUEST = 1, RESPONSE = 2 };
enum class ErrorCode : uint32_t { SUCCESS = 0, INVALID_REQUEST = 1, BLOCK_NOT_FOUND = 2, INTERNAL_ERROR = 3 };

struct MessageHeader {
    uint32_t magic;
    uint8_t version;
    MessageType type;
    uint16_t reserved;
    uint32_t body_length;
    uint32_t node_id;
} __attribute__((packed));

struct TensorMetadata {
    uint32_t num_tensors;
    uint32_t dimensions;
    uint64_t total_size;
    uint32_t tensor_shape[4];
} __attribute__((packed));

// Network I/O functions
bool send_all(int socket, const void* buffer, size_t length);
bool recv_all(int socket, void* buffer, size_t length);
bool writev_all(int socket, const struct iovec* iov, int iovcnt, size_t total_len);
bool readv_all(int socket, const struct iovec* iov, int iovcnt, size_t total_len);

// Protocol serialization functions
std::vector<uint8_t> serialize_request(const std::string& request_id, uint32_t node_id, const std::vector<uint64_t>& block_ids);
bool parse_request(const std::vector<uint8_t>& data, std::string& request_id, uint32_t& node_id, std::vector<uint64_t>& block_ids);
std::vector<uint8_t> serialize_response(const std::string& request_id, uint32_t node_id, ErrorCode error_code,
                                       const std::vector<uint64_t>& block_ids, const TensorMetadata& tensor_meta);
bool parse_response(const std::vector<uint8_t>& data, std::string& request_id, uint32_t& node_id,
                   ErrorCode& error_code, std::vector<uint64_t>& block_ids, TensorMetadata& tensor_meta);