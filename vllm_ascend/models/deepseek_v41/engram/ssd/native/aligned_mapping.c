/* aligned_mapping.c — 64KiB 对齐共享映射 helper(04 §4.3)。
 *
 * 目标: 所有 rank 把同一个 SharedMemory fd 映射到各自虚址空间中一个
 * 64KiB 对齐的地址, 载荷物理 offset 一律为 0。multiprocessing.shared_memory
 * 的 Python mmap 不保证对齐, 也不支持选定地址; 这里用三步 mmap:
 *   1) 匿名 PROT_NONE 预留 [size + align) 的虚址区间;
 *   2) 在预留区间内的对齐子区间以 MAP_SHARED|MAP_FIXED 映射 shm fd offset=0;
 *      MAP_FIXED 只覆盖本 helper 拥有的预留范围;
 *   3) munmap 预留区间头尾多余部分。
 * 失败时全部回滚, 不留半映射。
 */

#define _GNU_SOURCE
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

static void *_mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off) {
    return mmap(addr, len, prot, flags, fd, off);
}

void *am_map_shared_aligned(int shm_fd, size_t size, size_t align, char *err, int errlen) {
    if (align == 0 || (align & (align - 1)) != 0) {
        snprintf(err, errlen, "align must be a power of two, got %zu", align);
        return NULL;
    }
    /* 预留足够大, 保证其中必有一个 align 对齐边界 */
    size_t reserve = size + align;
    void *base = _mmap(NULL, reserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (base == MAP_FAILED) {
        snprintf(err, errlen, "reserve mmap failed: %s", strerror(errno));
        return NULL;
    }
    uintptr_t base_addr = (uintptr_t)base;
    uintptr_t aligned_addr = (base_addr + align - 1) / align * align;
    size_t head = aligned_addr - base_addr;
    size_t tail = reserve - head - size;
    void *mapped = _mmap((void *)aligned_addr, size, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED, shm_fd, 0);
    if (mapped == MAP_FAILED || mapped != (void *)aligned_addr) {
        int saved = errno;
        munmap(base, reserve);
        snprintf(err, errlen, "fixed mmap at aligned address failed: %s",
                 strerror(saved));
        return NULL;
    }
    if (head) munmap(base, head);
    if (tail) munmap((void *)(aligned_addr + size), tail);
    return (void *)aligned_addr;
}

int am_unmap(void *addr, size_t size) {
    if (!addr) return 0;
    return munmap(addr, size) == 0 ? 0 : errno;
}

int am_madvise_hugepage(void *addr, size_t size) {
#ifdef MADV_HUGEPAGE
    return madvise(addr, size, MADV_HUGEPAGE) == 0 ? 0 : errno;
#else
    return 0;
#endif
}
