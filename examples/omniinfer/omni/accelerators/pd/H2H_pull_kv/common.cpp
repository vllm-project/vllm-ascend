#include "common.h"
#include <vector>
#include <cstring>
#include <sys/socket.h>
#include <sys/uio.h>
#include <iostream>
#include <algorithm>
#include <errno.h>
#include <chrono>
#include <thread>


bool send_all(int socket, const void* buffer, size_t length) {
    const char* ptr = static_cast<const char*>(buffer);
    size_t total_sent = 0;
    int retry_count = 0;
    const int max_retries = 3;
    while (total_sent < length) {
        ssize_t sent = send(socket, ptr + total_sent, length - total_sent, MSG_NOSIGNAL);
        if (sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                retry_count++;
                if (retry_count > max_retries) { std::cerr << "send_all failed after " << max_retries << " retries: " << strerror(errno) << std::endl; return false; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            } else { std::cerr << "send_all failed: " << strerror(errno) << std::endl; return false; }
        } else if (sent == 0) { std::cerr << "send_all: connection closed by peer" << std::endl; return false; }
        total_sent += sent; retry_count = 0;
    }
    return true;
}

bool recv_all(int socket, void* buffer, size_t length) {
    char* ptr = static_cast<char*>(buffer);
    size_t total_received = 0;
    int retry_count = 0;
    const int max_retries = 3;
    while (total_received < length) {
        ssize_t received = recv(socket, ptr + total_received, length - total_received, 0);
        if (received < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                retry_count++;
                if (retry_count > max_retries) { std::cerr << "recv_all failed after " << max_retries << " retries: " << strerror(errno) << std::endl; return false; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            } else { std::cerr << "recv_all failed: " << strerror(errno) << std::endl; return false; }
        } else if (received == 0) { return false; }
        total_received += received; retry_count = 0;
    }
    return true;
}

bool writev_all(int socket, const struct iovec* iov, int iovcnt, size_t total_len) {
    size_t total_sent = 0;
    std::vector<struct iovec> remaining_iov(iov, iov + iovcnt);
    int retry_count = 0;
    const int max_retries = 3;

    size_t calculated_len = 0;
    for (int i = 0; i < iovcnt; i++) calculated_len += iov[i].iov_len;
    if (calculated_len != total_len) {
        std::cerr << "writev_all: length mismatch - calculated: " << calculated_len << ", expected: " << total_len << std::endl;
        return false;
    }

    while (total_sent < total_len) {
        ssize_t sent = writev(socket, remaining_iov.data(), remaining_iov.size());
        if (sent < 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                retry_count++;
                if (retry_count > max_retries) { std::cerr << "writev_all failed after " << max_retries << " retries: " << strerror(errno) << std::endl; return false; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            } else { std::cerr << "writev_all failed: sent=" << sent << " errno=" << strerror(errno) << std::endl; return false; }
        } else if (sent == 0) { std::cerr << "writev_all: connection closed by peer" << std::endl; return false; }
        total_sent += sent; retry_count = 0;

        if (total_sent >= total_len) break;

        size_t bytes_to_skip = sent;
        auto new_end = std::remove_if(remaining_iov.begin(), remaining_iov.end(),
            [&bytes_to_skip](struct iovec& iov_elem) {
                if (bytes_to_skip >= iov_elem.iov_len) { bytes_to_skip -= iov_elem.iov_len; return true; }
                else { iov_elem.iov_base = static_cast<char*>(iov_elem.iov_base) + bytes_to_skip; iov_elem.iov_len -= bytes_to_skip; bytes_to_skip = 0; return false; }
            });
        remaining_iov.erase(new_end, remaining_iov.end());
    }
    return true;
}

std::vector<uint8_t> serialize_request(const std::string& request_id, uint32_t node_id, const std::vector<uint64_t>& block_ids) {
    std::vector<uint8_t> data;

    uint32_t request_id_length = request_id.size();
    uint32_t body_length = sizeof(uint32_t) + request_id_length +
                          sizeof(uint32_t) + block_ids.size() * sizeof(uint64_t);

    MessageHeader header;
    header.magic = PROTOCOL_MAGIC;
    header.version = PROTOCOL_VERSION;
    header.type = MessageType::REQUEST;
    header.body_length = body_length;
    header.node_id = node_id;

    data.insert(data.end(), reinterpret_cast<uint8_t*>(&header), reinterpret_cast<uint8_t*>(&header) + sizeof(MessageHeader));
    data.insert(data.end(), reinterpret_cast<uint8_t*>(&request_id_length), reinterpret_cast<uint8_t*>(&request_id_length) + sizeof(uint32_t));
    data.insert(data.end(), request_id.begin(), request_id.end());

    uint32_t num_blocks = block_ids.size();
    data.insert(data.end(), reinterpret_cast<uint8_t*>(&num_blocks), reinterpret_cast<uint8_t*>(&num_blocks) + sizeof(uint32_t));
    if (!block_ids.empty()) {
        data.insert(data.end(), reinterpret_cast<const uint8_t*>(block_ids.data()),
                   reinterpret_cast<const uint8_t*>(block_ids.data()) + block_ids.size() * sizeof(uint64_t));
    }
    return data;
}

bool parse_request(const std::vector<uint8_t>& data, std::string& request_id, uint32_t& node_id, std::vector<uint64_t>& block_ids) {
    if (data.size() < sizeof(uint32_t) * 2) { std::cerr << "parse_request: data too small: " << data.size() << std::endl; return false; }
    size_t offset = 0;
    const uint32_t* request_id_length = reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += sizeof(uint32_t);
    if (offset + *request_id_length > data.size()) { std::cerr << "parse_request: request_id too long: " << *request_id_length << std::endl; return false; }
    if (*request_id_length > 0) {
        const char* request_id_start = reinterpret_cast<const char*>(data.data() + offset);
        request_id.assign(request_id_start, *request_id_length);
        offset += *request_id_length;
    }
    if (offset + sizeof(uint32_t) > data.size()) { std::cerr << "parse_request: no space for block count" << std::endl; return false; }
    const uint32_t* num_blocks = reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += sizeof(uint32_t);
    if (offset + *num_blocks * sizeof(uint64_t) > data.size()) {
        std::cerr << "parse_request: not enough data for blocks: need " << (*num_blocks * sizeof(uint64_t)) << ", have " << (data.size() - offset) << std::endl;
        return false;
    }
    block_ids.clear();
    if (*num_blocks > 0) {
        const uint64_t* blocks_start = reinterpret_cast<const uint64_t*>(data.data() + offset);
        block_ids.assign(blocks_start, blocks_start + *num_blocks);
    }
    node_id = 0;
    return true;
}

std::vector<uint8_t> serialize_response(const std::string& request_id, uint32_t /* node_id */, ErrorCode error_code,
                                       const std::vector<uint64_t>& block_ids, const TensorMetadata& tensor_meta) {
    std::vector<uint8_t> data;
    uint32_t body_length = sizeof(ErrorCode) +
                          sizeof(uint32_t) + request_id.size() +
                          sizeof(uint32_t) + block_ids.size() * sizeof(uint64_t) +
                          sizeof(TensorMetadata);

    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&error_code),
               reinterpret_cast<const uint8_t*>(&error_code) + sizeof(ErrorCode));

    uint32_t request_id_length = request_id.size();
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&request_id_length),
               reinterpret_cast<const uint8_t*>(&request_id_length) + sizeof(uint32_t));
    data.insert(data.end(), request_id.begin(), request_id.end());

    uint32_t num_blocks = block_ids.size();
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&num_blocks),
               reinterpret_cast<const uint8_t*>(&num_blocks) + sizeof(uint32_t));
    if (!block_ids.empty()) {
        data.insert(data.end(), reinterpret_cast<const uint8_t*>(block_ids.data()),
                   reinterpret_cast<const uint8_t*>(block_ids.data()) + block_ids.size() * sizeof(uint64_t));
    }
    data.insert(data.end(), reinterpret_cast<const uint8_t*>(&tensor_meta),
               reinterpret_cast<const uint8_t*>(&tensor_meta) + sizeof(TensorMetadata));
    if (data.size() != body_length) {
        std::cerr << "Warning: Calculated body_length (" << body_length
                  << ") doesn't match actual data size (" << data.size() << ")" << std::endl;
    }
    return data;
}

bool parse_response(const std::vector<uint8_t>& data, std::string& request_id, uint32_t& node_id,
                   ErrorCode& error_code, std::vector<uint64_t>& block_ids, TensorMetadata& tensor_meta) {
    if (data.size() < sizeof(ErrorCode) + sizeof(uint32_t)) { std::cerr << "parse_response: data too small: " << data.size() << std::endl; return false; }
    size_t offset = 0;

    error_code = *reinterpret_cast<const ErrorCode*>(data.data() + offset);
    offset += sizeof(ErrorCode);

    const uint32_t* request_id_length = reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += sizeof(uint32_t);
    if (offset + *request_id_length > data.size()) { std::cerr << "parse_response: request_id too long" << std::endl; return false; }
    if (*request_id_length > 0) {
        const char* request_id_start = reinterpret_cast<const char*>(data.data() + offset);
        request_id.assign(request_id_start, *request_id_length);
        offset += *request_id_length;
    }

    if (offset + sizeof(uint32_t) > data.size()) { std::cerr << "parse_response: no space for block count" << std::endl; return false; }
    const uint32_t* num_blocks = reinterpret_cast<const uint32_t*>(data.data() + offset);
    offset += sizeof(uint32_t);

    if (offset + *num_blocks * sizeof(uint64_t) > data.size()) { std::cerr << "parse_response: not enough data for blocks" << std::endl; return false; }
    block_ids.clear();
    if (*num_blocks > 0) {
        const uint64_t* blocks_start = reinterpret_cast<const uint64_t*>(data.data() + offset);
        block_ids.assign(blocks_start, blocks_start + *num_blocks);
        offset += *num_blocks * sizeof(uint64_t);
    }

    if (offset + sizeof(TensorMetadata) > data.size()) { std::cerr << "parse_response: no space for tensor metadata" << std::endl; return false; }
    const TensorMetadata* meta = reinterpret_cast<const TensorMetadata*>(data.data() + offset);
    tensor_meta = *meta;

    node_id = 0;
    return true;
}