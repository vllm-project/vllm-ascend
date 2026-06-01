#include "cityhash.h"
#include <string.h>

// CityHash64 常量
static const uint64_t k0 = 0xc3a5c85c97cb3127ULL;
static const uint64_t k1 = 0xb492b66fbe98f273ULL;
static const uint64_t k2 = 0x9ae16a3b2f90404fULL;

static inline uint64_t rotate64(uint64_t val, int shift) {
    return shift == 0 ? val : ((val >> shift) | (val << (64 - shift)));
}

static inline uint64_t fetch64(const char* p) {
    uint64_t result;
    memcpy(&result, p, sizeof(result));
    return result;
}

static inline uint32_t fetch32(const char* p) {
    uint32_t result;
    memcpy(&result, p, sizeof(result));
    return result;
}

static inline uint64_t hash_len_16(uint64_t u, uint64_t v, uint64_t mul) {
    uint64_t a = (u ^ v) * mul;
    a ^= (a >> 47);
    uint64_t b = (v ^ a) * mul;
    b ^= (b >> 47);
    b *= mul;
    return b;
}

static uint64_t hash_len_0_to_16(const char* s, size_t len) {
    if (len >= 8) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = fetch64(s) + k2;
        uint64_t b = fetch64(s + len - 8);
        uint64_t c = rotate64(b, 37) * mul + a;
        uint64_t d = (rotate64(a, 25) + b) * mul;
        return hash_len_16(c, d, mul);
    }
    if (len >= 4) {
        uint64_t mul = k2 + len * 2;
        uint64_t a = fetch32(s);
        return hash_len_16(len + (a << 3), fetch32(s + len - 4), mul);
    }
    if (len > 0) {
        uint8_t a = (uint8_t)s[0];
        uint8_t b = (uint8_t)s[len >> 1];
        uint8_t c = (uint8_t)s[len - 1];
        uint32_t y = (uint32_t)a + ((uint32_t)b << 8);
        uint32_t z = (uint32_t)len + ((uint32_t)c << 2);
        return ((uint64_t)(y * k2) ^ (uint64_t)(z * k0)) * k2;
    }
    return k2;
}

static uint64_t hash_len_17_to_32(const char* s, size_t len) {
    uint64_t mul = k2 + len * 2;
    uint64_t a = fetch64(s) * k1;
    uint64_t b = fetch64(s + 8);
    uint64_t c = fetch64(s + len - 8) * mul;
    uint64_t d = fetch64(s + len - 16) * k2;
    return hash_len_16(rotate64(a + b, 43) + rotate64(c, 30) + d,
                       a + rotate64(b + k2, 18) + c, mul);
}

uint64_t cityhash64(const void* data, size_t len) {
    const char* s = (const char*)data;

    if (len <= 16) {
        return hash_len_0_to_16(s, len);
    }
    if (len <= 32) {
        return hash_len_17_to_32(s, len);
    }

    // 对于更长的数据，使用简化版本
    uint64_t x = fetch64(s + len - 40);
    uint64_t y = fetch64(s + len - 16) + fetch64(s + len - 56);
    uint64_t z = hash_len_16(fetch64(s + len - 48) + len,
                             fetch64(s + len - 24), k2);

    uint64_t v0 = len + rotate64(fetch64(s + len - 32), 30) * k0;
    uint64_t v1 = rotate64(y + z, 42) * k0 + fetch64(s + len - 8);
    uint64_t w0 = rotate64(x + v0, 33) * k0;
    uint64_t w1 = (y + fetch64(s)) * k0;

    return hash_len_16(v0 + w1, w0 + v1, k2);
}

uint64_t cityhash64_with_seed(const void* data, size_t len, uint64_t seed) {
    return hash_len_16(cityhash64(data, len) - k2, seed, k2);
}

// 针对 N-Gram 优化的哈希函数
uint64_t ngram_hash(const uint32_t* tokens, uint8_t length) {
    // 直接对 token 数组进行哈希
    return cityhash64(tokens, length * sizeof(uint32_t));
}
