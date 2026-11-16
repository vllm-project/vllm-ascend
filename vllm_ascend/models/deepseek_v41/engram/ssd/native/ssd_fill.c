/* ssd_fill.c — Engram SSD miss 填充引擎(可靠版, 04 §8 契约)。
 *
 * 相对 2026-11-07 原型版的修复(缺陷复现证据 phase-3/runs/20261109T2030Z-t1-old-engine-repro):
 *   1. completion 强校验 res == expected_nbytes 且 res2 == 0, 短读一律失败(不再 res>=288 放行);
 *   2. 部分提交/EAGAIN/EINTR: pending 游标回滚补齐提交; 失败批先排空在飞请求再返回;
 *   3. aio_data 携带 batch_generation + buffer slot, 完成按代校验, 旧代事件丢弃不上账;
 *   4. open 同步 ready(io_setup 在调用线程完成后才创建 worker), 任一失败统一 unwind;
 *      dual 第二盘失败回收第一盘; 线程内禁止 exit/abort;
 *   5. close 幂等: 全量 join、io_destroy、关 fd、释放缓冲, dual 同时关另一盘;
 *   6. fill 入口校验 id 有效区间/slot 容量/指针; 空批零统计;
 *   7. io_setup 申请量(io_nr)与 QD 分开参数化。
 *
 * 语义契约: ssdf_fill 返回 0 时, 所有 n 行的 256B codes + 32B scales 已写入平面;
 * 返回非 0 时失败行不写入, 且返回前在飞请求已排空(除非 poison)。
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdarg.h>
#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <errno.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>

/* ---------- ABI 常量与错误码 ---------- */

#define SSDF_BLK 4096
#define SSDF_BMAX 16384          /* 最大 extent(prefill 合并读, 4K 对齐) */
#define SSDF_BUFSZ SSDF_BMAX     /* 槽缓冲: 统一按最大 extent 分配 */
#define SSDF_STRADDLE 8192       /* 跨块行精确读宽(2 块), 不随 BUFSZ 变化 */
#define SSDF_ROW_BYTES 288
#define SSDF_SLOT_SHIFT 21       /* aio_data 低 21 位 = buffer slot(QD 至 2M) */
#define SSDF_MAX_INTERVALS 4     /* 有效逻辑行区间数 */
#define SSDF_DRAIN_TIMEOUT_MS 200
#define SSDF_STALL_POISON_MS 120000   /* 无进展 120s → poison */

#define SSDF_OK 0
#define SSDF_ERR_ARG 1          /* 参数非法(调用方) */
#define SSDF_ERR_RANGE 2        /* id/slot 越界 */
#define SSDF_ERR_IO 3           /* read 失败/短读/提交失败 */
#define SSDF_ERR_DRAIN 4        /* 无法排空 → poison */
#define SSDF_ERR_CLOSED 5
#define SSDF_ERR_SETUP 6
#define SSDF_ERR_POISON 7

typedef struct {
    int32_t code;
    char detail[256];
} ssdf_error;

/* prefill 合并读(04 §8.4): Extent=一次对齐读, Scatter=extent 内行→槽位散写 */
typedef struct {
    int64_t offset;          /* 4KiB 对齐 extent 起点(首位避免填充) */
    int32_t file_index;      /* dual: 0/1; 单盘必须 0 */
    int32_t length;          /* 4096 的倍数, <= SSDF_BMAX */
    int32_t scatter_begin;   /* scatter 数组起始下标 */
    int32_t scatter_count;
} ssdf_extent;

typedef struct {
    int32_t rel_offset;      /* extent 内行偏移(rel+288<=length) */
    int32_t target_slot;     /* DDR 槽位 */
} ssdf_scatter;

_Static_assert(sizeof(ssdf_extent) == 24, "extent layout");
_Static_assert(sizeof(ssdf_scatter) == 8, "scatter layout");

static void ssdf_set_err(ssdf_error *e, int32_t code, const char *fmt, ...) {
    if (!e) return;
    e->code = code;
    va_list ap; va_start(ap, fmt);
    vsnprintf(e->detail, sizeof(e->detail), fmt, ap);
    va_end(ap);
}

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* ---------- 测试注入钩子(生产路径默认直通; ssdf_test_hook) ---------- */

enum {
    SSDF_HOOK_CLEAR = 0,
    SSDF_HOOK_SUBMIT_SHORT = 1,   /* arg 次: io_submit 返回 count-1(部分提交) */
    SSDF_HOOK_SUBMIT_EAGAIN = 2,  /* arg 次: io_submit 返回 -EAGAIN */
    SSDF_HOOK_GETEVENTS_DELAY = 3,/* 每次 getevents 前睡 arg ms(0=清除) */
    SSDF_HOOK_FORGE_STALE = 4,    /* arg 个: 下一 getevents 先返回伪造旧代事件, 自清 */
    SSDF_HOOK_IO_SETUP_FAIL = 5,  /* 第 arg 个(0-based) io_setup 返回 -EAGAIN, 自清 */
};

static pthread_mutex_t g_hook_mu = PTHREAD_MUTEX_INITIALIZER;
static long g_submit_short_left = 0;
static long g_submit_eagain_left = 0;
static long g_getevents_delay_ms = 0;
static long g_forge_stale_count = 0;
static long g_io_setup_fail_idx = -1;
static long g_io_setup_seen = 0;

void ssdf_test_hook(int op, long arg) {
    pthread_mutex_lock(&g_hook_mu);
    switch (op) {
    case SSDF_HOOK_SUBMIT_SHORT: g_submit_short_left = arg; break;
    case SSDF_HOOK_SUBMIT_EAGAIN: g_submit_eagain_left = arg; break;
    case SSDF_HOOK_GETEVENTS_DELAY: g_getevents_delay_ms = arg; break;
    case SSDF_HOOK_FORGE_STALE: g_forge_stale_count = arg; break;
    case SSDF_HOOK_IO_SETUP_FAIL: g_io_setup_fail_idx = arg; g_io_setup_seen = 0; break;
    case SSDF_HOOK_CLEAR:
    default:
        g_submit_short_left = 0; g_submit_eagain_left = 0; g_getevents_delay_ms = 0;
        g_forge_stale_count = 0; g_io_setup_fail_idx = -1; break;
    }
    pthread_mutex_unlock(&g_hook_mu);
}

static long sys_io_setup(unsigned nr, aio_context_t *c) {
    pthread_mutex_lock(&g_hook_mu);
    long fail = (g_io_setup_fail_idx >= 0 && g_io_setup_seen == g_io_setup_fail_idx);
    if (g_io_setup_fail_idx >= 0) g_io_setup_seen++;
    pthread_mutex_unlock(&g_hook_mu);
    if (fail) { errno = EAGAIN; return -1; }
    return syscall(__NR_io_setup, nr, c);
}

static long sys_io_submit(aio_context_t c, long nr, struct iocb **b) {
    pthread_mutex_lock(&g_hook_mu);
    if (g_submit_eagain_left > 0) {
        g_submit_eagain_left--; pthread_mutex_unlock(&g_hook_mu);
        errno = EAGAIN; return -1;
    }
    long drop = (g_submit_short_left > 0 && nr > 1) ? (g_submit_short_left--, 1L) : 0;
    pthread_mutex_unlock(&g_hook_mu);
    /* 真实部分提交模拟: 只把前 nr-1 个交给内核(内核接受前缀语义) */
    if (drop) nr -= 1;
    return syscall(__NR_io_submit, c, nr, b);
}

static long sys_io_getevents(aio_context_t c, long mn, long mx,
                             struct io_event *e, struct timespec *t) {
    pthread_mutex_lock(&g_hook_mu);
    long delay = g_getevents_delay_ms;
    long forge = g_forge_stale_count;
    if (g_forge_stale_count > 0) g_forge_stale_count = 0;
    pthread_mutex_unlock(&g_hook_mu);
    if (delay > 0) {
        struct timespec d = { delay / 1000, (delay % 1000) * 1000000L };
        nanosleep(&d, NULL);
    }
    long n = syscall(__NR_io_getevents, c, mn, mx, e, t);
    if (n > 0 && forge > 0) {
        long extra = (forge < mx - n) ? forge : (mx - n);
        for (long i = n; i > 0 && i + extra > i; i--) e[i + extra - 1] = e[i - 1];
        for (long i = 0; i < extra; i++) {
            e[i].data = 0;   /* gen=0(首代之后必然不符), slot=0 */
            e[i].obj = 0; e[i].res = SSDF_BUFSZ; e[i].res2 = 0;
        }
        n += extra;
    }
    return n;
}

static long sys_io_destroy(aio_context_t c) { return syscall(__NR_io_destroy, c); }

/* ---------- 请求与 worker ---------- */

enum { REQ_IDLE = 0, REQ_SUBMITTED = 1, REQ_DONE = 2, REQ_FAILED = 3 };

typedef struct {
    int64_t k;            /* 批内请求下标 */
    uint64_t gen;
    uint32_t expected;    /* 期望字节数: 4096 或 8192 */
    uint32_t row_off;     /* 行在块内偏移 */
    int32_t target;       /* DDR 槽位 */
    int32_t state;
} ssdf_req;

typedef struct Handle Handle;

typedef struct {
    int fd, qd;
    aio_context_t ctx;
    int ctx_ready;
    struct iocb *iocbs; struct iocb **pp; struct io_event *evs;
    char *bufs; int *free_slots; ssdf_req *reqs;
    int nfree;
    /* 每批字段(fill 持 mu 写入后 broadcast) */
    const uint64_t *ids; const int32_t *slots;
    long begin, end;
    uint8_t *codes, *scales;
    /* read_extents 批字段 */
    const ssdf_extent *ext; const ssdf_scatter *sct;
    const int *ext_index;   /* 本 worker 处理的 extent 下标表 */
    int ext_n;
    int batch_kind;         /* 0=逐行 fill, 1=extent 合并读 */
    /* 每批统计 */
    long fails, ge_calls, ge_events, stale, partial_submits, eagain, eintr, cancelled;
    long hard_errno;
    int poisoned;
    double extract_ms, submit_ms;
} Worker;

struct Handle {
    pthread_mutex_t mu;            /* 保护 gen/jobs_left/shutdown 与批字段发布 */
    pthread_cond_t go, done;
    pthread_mutex_t fill_mu;       /* fill/close 串行化 */
    int nthreads, jobs_left, shutdown, closed, poisoned;
    uint64_t gen;                  /* 单调批代号, 从 1 开始 */
    Worker *w; pthread_t *th;
    int nthreads_started;
    int fd_owned;
    long fails, ge_calls, ge_events, stale, partial_submits, eagain, eintr;
    long rows_total;
    double extract_ms, submit_ms;
    Handle *other;                 /* dual: 另一层 */
    uint64_t base1;                /* dual: 第二文件全局行基址 */
    uint64_t rows1, rows2;
    uint64_t iv[2 * SSDF_MAX_INTERVALS];
    int n_iv;
};

typedef struct { Worker *w; Handle *h; } WArg;

static void worker_process(Worker *w, Handle *h, uint64_t gen);
static void worker_process_extents(Worker *w, Handle *h, uint64_t gen);
static void restore_free_from(Worker *w, struct iocb **pp, int beg, int m);
void *worker_loop_main(void *arg);

/* ---------- 打开/释放 ---------- */

static void handle_free(Handle *h) {
    if (!h) return;
    if (h->w) {
        for (int i = 0; i < h->nthreads; i++) {
            Worker *w = &h->w[i];
            if (w->ctx_ready) { sys_io_destroy(w->ctx); w->ctx_ready = 0; }
            free(w->iocbs); free(w->pp); free(w->evs); free(w->bufs);
            free(w->free_slots); free(w->reqs);
        }
        free(h->w); h->w = NULL;
    }
    free(h->th); h->th = NULL;
    if (h->fd_owned >= 0) { close(h->fd_owned); h->fd_owned = -1; }
    pthread_mutex_destroy(&h->mu); pthread_mutex_destroy(&h->fill_mu);
    pthread_cond_destroy(&h->go); pthread_cond_destroy(&h->done);
    free(h);
}

static void *ssdf_open_iv(const char *path, int threads, int qd, int io_nr, int cpubase,
                          const uint64_t *iv, int n_iv, ssdf_error *err) {
    if (!path || threads <= 0 || qd <= 0 || qd >= (1 << SSDF_SLOT_SHIFT) ||
        io_nr <= 0 || n_iv <= 0 || n_iv > SSDF_MAX_INTERVALS) {
        ssdf_set_err(err, SSDF_ERR_ARG, "bad open args: threads=%d qd=%d io_nr=%d n_iv=%d",
                     threads, qd, io_nr, n_iv);
        return NULL;
    }
    for (int i = 0; i < n_iv; i++)
        if (iv[2 * i + 1] > UINT64_MAX / SSDF_ROW_BYTES) {   /* 空区间(lo==hi)允许 */
            ssdf_set_err(err, SSDF_ERR_ARG, "bad interval %d", i);
            return NULL;
        }
    int fd = open(path, O_RDONLY | O_DIRECT);
    if (fd < 0) {
        ssdf_set_err(err, SSDF_ERR_SETUP, "open(%s) O_DIRECT failed: %s", path, strerror(errno));
        return NULL;
    }
    Handle *h = calloc(1, sizeof(Handle));
    if (!h) { close(fd); ssdf_set_err(err, SSDF_ERR_SETUP, "oom handle"); return NULL; }
    h->fd_owned = fd;
    h->nthreads = threads;
    pthread_mutex_init(&h->mu, NULL); pthread_mutex_init(&h->fill_mu, NULL);
    pthread_cond_init(&h->go, NULL); pthread_cond_init(&h->done, NULL);
    h->n_iv = n_iv;
    memcpy(h->iv, iv, sizeof(uint64_t) * 2 * (size_t)n_iv);
    h->w = calloc((size_t)threads, sizeof(Worker));
    h->th = calloc((size_t)threads, sizeof(pthread_t));
    if (!h->w || !h->th) { handle_free(h); ssdf_set_err(err, SSDF_ERR_SETUP, "oom workers"); return NULL; }

    /* 同步 ready: 资源与 io_setup 全部在调用线程完成, 任一失败立即 unwind(无线程残留) */
    for (int i = 0; i < threads; i++) {
        Worker *w = &h->w[i];
        w->fd = fd; w->qd = qd;
        w->iocbs = calloc((size_t)qd, sizeof(struct iocb));
        w->pp = calloc((size_t)qd, sizeof(struct iocb *));
        w->evs = calloc((size_t)qd, sizeof(struct io_event));
        w->free_slots = calloc((size_t)qd, sizeof(int));
        w->reqs = calloc((size_t)qd, sizeof(ssdf_req));
        if (posix_memalign((void **)&w->bufs, SSDF_BLK, (size_t)qd * SSDF_BUFSZ) != 0) w->bufs = NULL;
        if (!w->iocbs || !w->pp || !w->evs || !w->free_slots || !w->reqs || !w->bufs) {
            ssdf_set_err(err, SSDF_ERR_SETUP, "worker %d alloc failed", i);
            handle_free(h);
            return NULL;
        }
        memset(w->bufs, 0xEE, (size_t)qd * SSDF_BUFSZ);
        memset(&w->ctx, 0, sizeof(w->ctx));
        if (sys_io_setup((unsigned)io_nr, &w->ctx) != 0) {
            ssdf_set_err(err, SSDF_ERR_SETUP, "worker %d io_setup(%d) failed: %s",
                         i, io_nr, strerror(errno));
            handle_free(h);
            return NULL;
        }
        w->ctx_ready = 1;
    }
    for (int i = 0; i < threads; i++) {
        WArg *wa = malloc(sizeof(WArg));
        if (!wa) { ssdf_set_err(err, SSDF_ERR_SETUP, "oom warg"); goto fail_threads; }
        wa->w = &h->w[i]; wa->h = h;
        if (pthread_create(&h->th[i], NULL, worker_loop_main, wa) != 0) {
            free(wa);
            ssdf_set_err(err, SSDF_ERR_SETUP, "pthread_create %d failed: %s", i, strerror(errno));
            goto fail_threads;
        }
        h->nthreads_started++;
        if (cpubase >= 0) {
            cpu_set_t s; CPU_ZERO(&s); CPU_SET(cpubase + i, &s);
            pthread_setaffinity_np(h->th[i], sizeof(s), &s);
        }
    }
    return h;

fail_threads:
    pthread_mutex_lock(&h->mu);
    h->shutdown = 1; h->gen++;
    pthread_cond_broadcast(&h->go);
    pthread_mutex_unlock(&h->mu);
    for (int i = 0; i < h->nthreads_started; i++) pthread_join(h->th[i], NULL);
    handle_free(h);
    return NULL;
}

/* ---------- worker 主循环 ---------- */

void *worker_loop_main(void *arg) {
    WArg *wa = (WArg *)arg;
    Worker *w = wa->w; Handle *h = wa->h;
    free(wa);
    uint64_t mygen = 0;
    pthread_mutex_lock(&h->mu);
    for (;;) {
        while (!h->shutdown && h->gen == mygen) pthread_cond_wait(&h->go, &h->mu);
        if (h->shutdown) break;
        mygen = h->gen;
        pthread_mutex_unlock(&h->mu);
        if (w->batch_kind == 1)
            worker_process_extents(w, h, mygen);
        else
            worker_process(w, h, mygen);
        pthread_mutex_lock(&h->mu);
        h->fails += w->fails; w->fails = 0;
        h->ge_calls += w->ge_calls; w->ge_calls = 0;
        h->ge_events += w->ge_events; w->ge_events = 0;
        h->stale += w->stale; w->stale = 0;
        h->partial_submits += w->partial_submits; w->partial_submits = 0;
        h->eagain += w->eagain; w->eagain = 0;
        h->eintr += w->eintr; w->eintr = 0;
        h->extract_ms += w->extract_ms; w->extract_ms = 0;
        h->submit_ms += w->submit_ms; w->submit_ms = 0;
        if (w->poisoned) h->poisoned = 1;
        if (--h->jobs_left == 0) pthread_cond_signal(&h->done);
    }
    pthread_mutex_unlock(&h->mu);
    return NULL;   /* io_destroy 与缓冲释放由 close 完成 */
}

/* ---------- 提交/完成核心 ---------- */

static void build_iocb(Worker *w, int bslot, int64_t k, uint64_t gen) {
    uint64_t off = w->ids[k] * SSDF_ROW_BYTES;      /* id 已由 fill 校验 */
    uint64_t blk = off / SSDF_BLK * SSDF_BLK;
    uint32_t len = (off + SSDF_ROW_BYTES <= blk + SSDF_BLK) ? SSDF_BLK : SSDF_STRADDLE;
    ssdf_req *r = &w->reqs[bslot];
    r->k = k; r->gen = gen; r->expected = len; r->row_off = (uint32_t)(off % SSDF_BLK);
    r->target = w->slots[k]; r->state = REQ_IDLE;
    struct iocb *io = &w->iocbs[bslot];
    memset(io, 0, sizeof(*io));
    io->aio_data = (gen << SSDF_SLOT_SHIFT) | (uint64_t)bslot;
    io->aio_lio_opcode = IOCB_CMD_PREAD;
    io->aio_fildes = w->fd;
    io->aio_buf = (uint64_t)(w->bufs + (size_t)bslot * SSDF_BUFSZ);
    io->aio_nbytes = len;
    io->aio_offset = (int64_t)blk;
}

static int bslot_of(struct iocb *io) {
    return (int)(io->aio_data & ((1ULL << SSDF_SLOT_SHIFT) - 1));
}

/* 回滚 pp[beg..m): 未被内核接受, 槽位归还, 请求回 IDLE */
static void restore_free_from(Worker *w, struct iocb **pp, int beg, int m) {
    for (int i = beg; i < m; i++) {
        int bslot = bslot_of(pp[i]);
        w->reqs[bslot].state = REQ_IDLE;
        w->free_slots[w->nfree++] = bslot;
    }
}

static void worker_process(Worker *w, Handle *h, uint64_t gen) {
    (void)h;
    const long total = w->end - w->begin;
    if (total <= 0) return;
    w->fails = 0; w->ge_calls = 0; w->ge_events = 0; w->stale = 0;
    w->partial_submits = 0; w->eagain = 0; w->eintr = 0; w->cancelled = 0;
    w->hard_errno = 0; w->extract_ms = 0; w->submit_ms = 0;
    w->nfree = 0;
    for (int s = 0; s < w->qd; s++) { w->free_slots[w->nfree++] = s; w->reqs[s].state = REQ_IDLE; }

    long next = w->begin;          /* 下一个未尝试提交的请求 */
    long done_cnt = 0;             /* 已到达终态的请求数 */
    int batch_failed = 0;
    double last_progress = now_ms();
    struct timespec ts = { SSDF_DRAIN_TIMEOUT_MS / 1000,
                           (SSDF_DRAIN_TIMEOUT_MS % 1000) * 1000000L };

    while (done_cnt < total) {
        /* 1) 尽量提交 pending; 失败后不再提交新请求 */
        if (!batch_failed && w->nfree > 0 && next < w->end) {
            int m = 0;
            while (w->nfree > 0 && next < w->end) {
                int bslot = w->free_slots[--w->nfree];
                build_iocb(w, bslot, next, gen);
                w->pp[m++] = &w->iocbs[bslot];
                next++;
            }
            double t0 = now_ms();
            long sret = sys_io_submit(w->ctx, m, w->pp);
            w->submit_ms += now_ms() - t0;
            if (sret < 0) {
                if (errno == EINTR) { w->eintr++; next -= m; restore_free_from(w, w->pp, 0, m); continue; }
                if (errno == EAGAIN) { w->eagain++; next -= m; restore_free_from(w, w->pp, 0, m); goto reap; }
                w->hard_errno = errno; batch_failed = 1;
                next -= m; restore_free_from(w, w->pp, 0, m);
            } else {
                if (sret < m) w->partial_submits++;
                /* 前缀 [0,sret) 已在飞: 标记 SUBMITTED; 其余回滚待补 */
                for (long i = 0; i < sret; i++)
                    w->reqs[bslot_of(w->pp[i])].state = REQ_SUBMITTED;
                if (sret < m) { next -= (m - sret); restore_free_from(w, w->pp, (int)sret, m); }
                last_progress = now_ms();
            }
        }
        /* 2) 失败批: 取消全部未提交请求(无在飞, 无需归还) */
        if (batch_failed) {
            while (next < w->end) {
                w->cancelled++; w->fails++;
                next++; done_cnt++;
                last_progress = now_ms();
            }
        }
        /* 3) 全部终态则结束 */
        if (done_cnt >= total) break;
reap:
        /* 4) 排空在飞 completion */
        {
            long n = sys_io_getevents(w->ctx, 1, w->qd, w->evs, &ts);
            w->ge_calls++;
            if (n > 0) { w->ge_events += n; last_progress = now_ms(); }
            if (n < 0) {
                if (errno == EINTR) { w->eintr++; continue; }
                w->hard_errno = errno; w->poisoned = 1;
                break;   /* 无法继续排空 → poison, fill 报错并封禁句柄 */
            }
            for (long i = 0; i < n; i++) {
                struct io_event *ev = &w->evs[i];
                uint64_t egen = ev->data >> SSDF_SLOT_SHIFT;
                int bslot = (int)(ev->data & ((1ULL << SSDF_SLOT_SHIFT) - 1));
                if (egen != gen || bslot >= w->qd || w->reqs[bslot].state != REQ_SUBMITTED) {
                    w->stale++;          /* 旧代/异常事件: 丢弃, 不动 free list */
                    continue;
                }
                ssdf_req *r = &w->reqs[bslot];
                double te = now_ms();
                if (ev->res == (long)r->expected && ev->res2 == 0) {
                    const char *row = w->bufs + (size_t)bslot * SSDF_BUFSZ + r->row_off;
                    memcpy(w->codes + (size_t)r->target * 256, row, 256);
                    memcpy(w->scales + (size_t)r->target * 32, row + 256, 32);
                    r->state = REQ_DONE;
                } else {
                    r->state = REQ_FAILED;
                    w->fails++;          /* 短读/IO 错误: 整批失败 */
                    batch_failed = 1;
                }
                w->extract_ms += now_ms() - te;
                w->free_slots[w->nfree++] = bslot;
                done_cnt++;
            }
            if (now_ms() - last_progress > SSDF_STALL_POISON_MS) {
                w->poisoned = 1;
                break;
            }
        }
    }
}

/* ---------- prefill 合并读(04 §8.4) ----------
 * 每个 extent 一次对齐读; 完成后按 scatter 表散写行到槽位。
 * 与逐行路径共用: gen 校验、严格 res==expected、部分提交补齐、失败排空。
 */
static void build_extent_iocb(Worker *w, int bslot, int k, uint64_t gen) {
    const ssdf_extent *e = &w->ext[k];
    ssdf_req *r = &w->reqs[bslot];
    r->k = k; r->gen = gen; r->expected = (uint32_t)e->length;
    r->row_off = 0; r->target = -1; r->state = REQ_IDLE;
    struct iocb *io = &w->iocbs[bslot];
    memset(io, 0, sizeof(*io));
    io->aio_data = (gen << SSDF_SLOT_SHIFT) | (uint64_t)bslot;
    io->aio_lio_opcode = IOCB_CMD_PREAD;
    io->aio_fildes = w->fd;
    io->aio_buf = (uint64_t)(w->bufs + (size_t)bslot * SSDF_BUFSZ);
    io->aio_nbytes = (uint32_t)e->length;
    io->aio_offset = e->offset;
}

static void worker_process_extents(Worker *w, Handle *h, uint64_t gen) {
    (void)h;
    const int total = w->ext_n;
    if (total <= 0) return;
    w->fails = 0; w->ge_calls = 0; w->ge_events = 0; w->stale = 0;
    w->partial_submits = 0; w->eagain = 0; w->eintr = 0; w->cancelled = 0;
    w->hard_errno = 0; w->extract_ms = 0; w->submit_ms = 0;
    w->nfree = 0;
    for (int s2 = 0; s2 < w->qd; s2++) { w->free_slots[w->nfree++] = s2; w->reqs[s2].state = REQ_IDLE; }

    long next = 0, done_cnt = 0;
    int batch_failed = 0;
    double last_progress = now_ms();
    struct timespec ts = { SSDF_DRAIN_TIMEOUT_MS / 1000,
                           (SSDF_DRAIN_TIMEOUT_MS % 1000) * 1000000L };

    while (done_cnt < total) {
        if (!batch_failed && w->nfree > 0 && next < total) {
            int m = 0;
            while (w->nfree > 0 && next < total) {
                int bslot = w->free_slots[--w->nfree];
                build_extent_iocb(w, bslot, w->ext_index[next], gen);
                w->pp[m++] = &w->iocbs[bslot];
                next++;
            }
            double t0 = now_ms();
            long sret = sys_io_submit(w->ctx, m, w->pp);
            w->submit_ms += now_ms() - t0;
            if (sret < 0) {
                if (errno == EINTR) { w->eintr++; next -= m; restore_free_from(w, w->pp, 0, m); continue; }
                if (errno == EAGAIN) { w->eagain++; next -= m; restore_free_from(w, w->pp, 0, m); goto reap2; }
                w->hard_errno = errno; batch_failed = 1;
                next -= m; restore_free_from(w, w->pp, 0, m);
            } else {
                if (sret < m) w->partial_submits++;
                for (long i = 0; i < sret; i++)
                    w->reqs[bslot_of(w->pp[i])].state = REQ_SUBMITTED;
                if (sret < m) { next -= (m - sret); restore_free_from(w, w->pp, (int)sret, m); }
                last_progress = now_ms();
            }
        }
        if (batch_failed) {
            while (next < total) { w->cancelled++; w->fails++; next++; done_cnt++; last_progress = now_ms(); }
        }
        if (done_cnt >= total) break;
reap2:
        {
            long n = sys_io_getevents(w->ctx, 1, w->qd, w->evs, &ts);
            w->ge_calls++;
            if (n > 0) { w->ge_events += n; last_progress = now_ms(); }
            if (n < 0) {
                if (errno == EINTR) { w->eintr++; continue; }
                w->hard_errno = errno; w->poisoned = 1;
                break;
            }
            for (long i = 0; i < n; i++) {
                struct io_event *ev = &w->evs[i];
                uint64_t egen = ev->data >> SSDF_SLOT_SHIFT;
                int bslot = (int)(ev->data & ((1ULL << SSDF_SLOT_SHIFT) - 1));
                if (egen != gen || bslot >= w->qd || w->reqs[bslot].state != REQ_SUBMITTED) {
                    w->stale++;
                    continue;
                }
                ssdf_req *r = &w->reqs[bslot];
                double te = now_ms();
                if (ev->res == (long)r->expected && ev->res2 == 0) {
                    const ssdf_extent *e = &w->ext[r->k];
                    const char *base = w->bufs + (size_t)bslot * SSDF_BUFSZ;
                    for (int32_t j = 0; j < e->scatter_count; j++) {
                        const ssdf_scatter *sc = &w->sct[e->scatter_begin + j];
                        const char *row = base + sc->rel_offset;
                        memcpy(w->codes + (size_t)sc->target_slot * 256, row, 256);
                        memcpy(w->scales + (size_t)sc->target_slot * 32, row + 256, 32);
                    }
                    r->state = REQ_DONE;
                } else {
                    r->state = REQ_FAILED;
                    w->fails++;
                    batch_failed = 1;
                }
                w->extract_ms += now_ms() - te;
                w->free_slots[w->nfree++] = bslot;
                done_cnt++;
            }
            if (now_ms() - last_progress > SSDF_STALL_POISON_MS) {
                w->poisoned = 1;
                break;
            }
        }
    }
}

static void arm_extents(Handle *h, const ssdf_extent *ext, const ssdf_scatter *sct,
                        int n, const int *idx, uint8_t *codes, uint8_t *scales) {
    pthread_mutex_lock(&h->mu);
    h->fails = 0; h->ge_calls = 0; h->ge_events = 0; h->stale = 0;
    h->partial_submits = 0; h->eagain = 0; h->eintr = 0;
    h->rows_total = n; h->extract_ms = 0; h->submit_ms = 0;
    h->jobs_left = h->nthreads;
    /* extent 均分到 worker; 空片也参与计数 */
    long per = (n + h->nthreads - 1) / h->nthreads;
    for (int i = 0; i < h->nthreads; i++) {
        Worker *w = &h->w[i];
        w->ext = ext; w->sct = sct; w->codes = codes; w->scales = scales;
        w->begin = (long)i * per;
        w->end = ((long)i + 1) * per > n ? n : ((long)i + 1) * per;
        if (w->end < w->begin) w->end = w->begin;
        w->ext_index = idx + w->begin;
        w->ext_n = (int)(w->end - w->begin);
        w->batch_kind = 1;
    }
    h->gen++;
    pthread_cond_broadcast(&h->go);
    pthread_mutex_unlock(&h->mu);
}

/* ---------- fill ---------- */

static int id_in_intervals(Handle *h, uint64_t id) {
    for (int i = 0; i < h->n_iv; i++)
        if (id >= h->iv[2 * i] && id < h->iv[2 * i + 1]) return 1;
    return 0;
}

static void arm(Handle *h, const uint64_t *ids, const int32_t *slots, int64_t n,
                uint8_t *codes, uint8_t *scales) {
    pthread_mutex_lock(&h->mu);
    h->fails = 0; h->ge_calls = 0; h->ge_events = 0; h->stale = 0;
    h->partial_submits = 0; h->eagain = 0; h->eintr = 0;
    h->rows_total = n; h->extract_ms = 0; h->submit_ms = 0;
    h->jobs_left = h->nthreads;
    long per = (n + h->nthreads - 1) / h->nthreads;
    for (int i = 0; i < h->nthreads; i++) {
        Worker *w = &h->w[i];
        w->ids = ids; w->slots = slots; w->codes = codes; w->scales = scales;
        w->begin = (long)i * per;
        w->end = ((long)i + 1) * per > n ? n : ((long)i + 1) * per;
        if (w->end < w->begin) w->end = w->begin;
        w->batch_kind = 0;
    }
    h->gen++;
    pthread_cond_broadcast(&h->go);
    pthread_mutex_unlock(&h->mu);
}

static void wait_done(Handle *h) {
    pthread_mutex_lock(&h->mu);
    while (h->jobs_left > 0) pthread_cond_wait(&h->done, &h->mu);
    pthread_mutex_unlock(&h->mu);
}

int ssdf_read_extents(void *hv, const ssdf_extent *ext, int64_t n_ext,
                      const ssdf_scatter *sct, int64_t n_sct,
                      uint8_t *codes, uint8_t *scales, int64_t capacity,
                      double *wall_ms, ssdf_error *err) {
    Handle *h = (Handle *)hv;
    if (!h) { ssdf_set_err(err, SSDF_ERR_ARG, "null handle"); return SSDF_ERR_ARG; }
    if (n_ext < 0 || n_sct < 0) { ssdf_set_err(err, SSDF_ERR_ARG, "negative counts"); return SSDF_ERR_ARG; }
    if (n_ext == 0) {
        if (wall_ms) *wall_ms = 0;
        return SSDF_OK;
    }
    if (!ext || !sct || !codes || !scales || capacity <= 0) {
        ssdf_set_err(err, SSDF_ERR_ARG, "null pointers or capacity");
        return SSDF_ERR_ARG;
    }
    pthread_mutex_lock(&h->fill_mu);
    if (h->closed) { pthread_mutex_unlock(&h->fill_mu); ssdf_set_err(err, SSDF_ERR_CLOSED, "closed"); return SSDF_ERR_CLOSED; }
    if (h->poisoned) { pthread_mutex_unlock(&h->fill_mu); ssdf_set_err(err, SSDF_ERR_POISON, "poisoned"); return SSDF_ERR_POISON; }
    /* 校验: 对齐/长度/散写范围/文件索引/槽位 */
    for (int64_t i = 0; i < n_ext; i++) {
        const ssdf_extent *e = &ext[i];
        if (e->offset < 0 || (e->offset % SSDF_BLK) != 0 ||
            e->length <= 0 || (e->length % SSDF_BLK) != 0 || e->length > SSDF_BMAX) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_RANGE, "extent[%lld] not 4K-aligned/<=16K", (long long)i);
            return SSDF_ERR_RANGE;
        }
        if (e->scatter_begin < 0 || e->scatter_count < 0 ||
            (int64_t)e->scatter_begin + e->scatter_count > n_sct) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_RANGE, "extent[%lld] scatter range oob", (long long)i);
            return SSDF_ERR_RANGE;
        }
        if (!h->other && e->file_index != 0) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_RANGE, "extent[%lld] file_index!=0 on single-disk", (long long)i);
            return SSDF_ERR_RANGE;
        }
        if (h->other && e->file_index != 0 && e->file_index != 1) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_RANGE, "extent[%lld] bad file_index", (long long)i);
            return SSDF_ERR_RANGE;
        }
        for (int32_t j = 0; j < e->scatter_count; j++) {
            const ssdf_scatter *sc = &sct[e->scatter_begin + j];
            if (sc->rel_offset < 0 || sc->rel_offset + SSDF_ROW_BYTES > e->length ||
                sc->target_slot < 0 || (uint64_t)sc->target_slot >= (uint64_t)capacity) {
                pthread_mutex_unlock(&h->fill_mu);
                ssdf_set_err(err, SSDF_ERR_RANGE, "extent[%lld] scatter[%d] oob", (long long)i, j);
                return SSDF_ERR_RANGE;
            }
        }
    }
    double t0 = now_ms();
    if (h->other) {
        /* dual: 按 file_index 拆分(下标表), 两侧独立 arm/wait */
        int *idx = malloc(sizeof(int) * (size_t)n_ext);
        if (!idx) { pthread_mutex_unlock(&h->fill_mu); ssdf_set_err(err, SSDF_ERR_ARG, "oom idx"); return SSDF_ERR_ARG; }
        int64_t na = 0;
        for (int64_t i = 0; i < n_ext; i++) if (ext[i].file_index == 0) idx[na++] = (int)i;
        /* 两侧各自连续下标表(arm_extents 按 begin 切片) */
        int *idxb = malloc(sizeof(int) * (size_t)(n_ext - na > 0 ? n_ext - na : 1));
        if (!idxb) { free(idx); pthread_mutex_unlock(&h->fill_mu); ssdf_set_err(err, SSDF_ERR_ARG, "oom idxb"); return SSDF_ERR_ARG; }
        int64_t nb = 0;
        for (int64_t i = 0; i < n_ext; i++) if (ext[i].file_index == 1) idxb[nb++] = (int)i;
        if (na) arm_extents(h, ext, sct, (int)na, idx, codes, scales);
        if (nb) arm_extents(h->other, ext, sct, (int)nb, idxb, codes, scales);
        if (na) wait_done(h);
        if (nb) wait_done(h->other);
        free(idx); free(idxb);
        long f = h->fails + (nb ? h->other->fails : 0);
        int pois = h->poisoned || (nb ? h->other->poisoned : 0);
        if (wall_ms) *wall_ms = now_ms() - t0;
        if (pois) { pthread_mutex_unlock(&h->fill_mu); ssdf_set_err(err, SSDF_ERR_DRAIN, "engine poisoned"); return SSDF_ERR_DRAIN; }
        if (f != 0) { pthread_mutex_unlock(&h->fill_mu); ssdf_set_err(err, SSDF_ERR_IO, "read_extents failed: %lld extents", (long long)f); return SSDF_ERR_IO; }
        pthread_mutex_unlock(&h->fill_mu);
        return SSDF_OK;
    }
    /* 单盘: 全部 extent 顺序下标 */
    int *idx = malloc(sizeof(int) * (size_t)n_ext);
    if (!idx) { pthread_mutex_unlock(&h->fill_mu); ssdf_set_err(err, SSDF_ERR_ARG, "oom idx"); return SSDF_ERR_ARG; }
    for (int64_t i = 0; i < n_ext; i++) idx[i] = (int)i;
    arm_extents(h, ext, sct, (int)n_ext, idx, codes, scales);
    wait_done(h);
    long f = h->fails;
    int pois = h->poisoned;
    free(idx);
    if (wall_ms) *wall_ms = now_ms() - t0;
    pthread_mutex_unlock(&h->fill_mu);
    if (pois) { ssdf_set_err(err, SSDF_ERR_DRAIN, "engine poisoned"); return SSDF_ERR_DRAIN; }
    if (f != 0) { ssdf_set_err(err, SSDF_ERR_IO, "read_extents failed: %lld extents", (long long)f); return SSDF_ERR_IO; }
    return SSDF_OK;
}

int ssdf_fill(void *hv, const uint64_t *ids, const int32_t *slots, int64_t n,
              uint8_t *codes, uint8_t *scales, int64_t capacity,
              double *wall_ms, ssdf_error *err) {
    Handle *h = (Handle *)hv;
    if (!h) { ssdf_set_err(err, SSDF_ERR_ARG, "null handle"); return SSDF_ERR_ARG; }
    pthread_mutex_lock(&h->fill_mu);
    if (h->closed) {
        pthread_mutex_unlock(&h->fill_mu);
        ssdf_set_err(err, SSDF_ERR_CLOSED, "engine closed");
        return SSDF_ERR_CLOSED;
    }
    if (h->poisoned) {
        pthread_mutex_unlock(&h->fill_mu);
        ssdf_set_err(err, SSDF_ERR_POISON, "engine poisoned");
        return SSDF_ERR_POISON;
    }
    if (n == 0) {
        pthread_mutex_unlock(&h->fill_mu);
        if (wall_ms) *wall_ms = 0;
        return SSDF_OK;
    }
    if (!ids || !slots || !codes || !scales || capacity <= 0) {
        pthread_mutex_unlock(&h->fill_mu);
        ssdf_set_err(err, SSDF_ERR_ARG, "null pointers or capacity=%lld", (long long)capacity);
        return SSDF_ERR_ARG;
    }
    for (int64_t i = 0; i < n; i++) {
        if (!id_in_intervals(h, ids[i])) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_RANGE, "id[%lld]=%llu out of logical range",
                         (long long)i, (unsigned long long)ids[i]);
            return SSDF_ERR_RANGE;
        }
        if (slots[i] < 0 || (uint64_t)slots[i] >= (uint64_t)capacity) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_RANGE, "slot[%lld]=%d out of [0,%lld)",
                         (long long)i, slots[i], (long long)capacity);
            return SSDF_ERR_RANGE;
        }
    }
    double t0 = now_ms();
    if (h->other) {
        /* dual: 按 base1 分派; B 侧转局部 id */
        int64_t na = 0;
        for (int64_t i = 0; i < n; i++) if (ids[i] < h->base1) na++;
        int64_t nb = n - na;
        uint64_t *ai = NULL, *bi = NULL; int32_t *as_ = NULL, *bs = NULL;
        if (na > 0) {
            ai = malloc((size_t)na * 8); as_ = malloc((size_t)na * 4);
        }
        if (nb > 0) {
            bi = malloc((size_t)nb * 8); bs = malloc((size_t)nb * 4);
        }
        if ((na > 0 && (!ai || !as_)) || (nb > 0 && (!bi || !bs))) {
            free(ai); free(as_); free(bi); free(bs);
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_ARG, "oom dual split");
            return SSDF_ERR_ARG;
        }
        int64_t ia = 0, ib = 0;
        for (int64_t i = 0; i < n; i++) {
            if (ids[i] < h->base1) { ai[ia] = ids[i]; as_[ia] = slots[i]; ia++; }
            else { bi[ib] = ids[i] - h->base1; bs[ib] = slots[i]; ib++; }
        }
        if (na) arm(h, ai, as_, na, codes, scales);
        if (nb) arm(h->other, bi, bs, nb, codes, scales);
        if (na) wait_done(h);
        if (nb) wait_done(h->other);
        long f = h->fails + (nb ? h->other->fails : 0);
        int pois = h->poisoned || (nb ? h->other->poisoned : 0);
        free(ai); free(as_); free(bi); free(bs);
        if (wall_ms) *wall_ms = now_ms() - t0;
        if (pois) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_DRAIN, "engine poisoned");
            return SSDF_ERR_DRAIN;
        }
        if (f != 0) {
            pthread_mutex_unlock(&h->fill_mu);
            ssdf_set_err(err, SSDF_ERR_IO, "fill failed: %ld/%lld rows (short-read/io error)",
                         f, (long long)n);
            return SSDF_ERR_IO;
        }
        pthread_mutex_unlock(&h->fill_mu);
        return SSDF_OK;
    }
    arm(h, ids, slots, n, codes, scales);
    wait_done(h);
    long f = h->fails;
    int pois = h->poisoned;
    if (wall_ms) *wall_ms = now_ms() - t0;
    pthread_mutex_unlock(&h->fill_mu);
    if (pois) {
        ssdf_set_err(err, SSDF_ERR_DRAIN, "engine poisoned (hard_errno=%ld)", h->w[0].hard_errno);
        return SSDF_ERR_DRAIN;
    }
    if (f != 0) {
        ssdf_set_err(err, SSDF_ERR_IO, "fill failed: %ld/%lld rows (short-read/io error)",
                     f, (long long)n);
        return SSDF_ERR_IO;
    }
    return SSDF_OK;
}

/* ---------- open(单盘/dual) ---------- */

void *ssdf_open(const char *path, int threads, int qd, int io_nr, int cpubase,
                const uint64_t *iv, int n_iv, ssdf_error *err) {
    return ssdf_open_iv(path, threads, qd, io_nr, cpubase, iv, n_iv, err);
}

void *ssdf_open_dual(const char *pa, const char *pb, uint64_t base1, uint64_t rows1, uint64_t rows2,
                     int threads, int qd, int io_nr, int cpuA, int cpuB, ssdf_error *err) {
    uint64_t iv_a[4] = {0, rows1, base1, base1 + rows2};
    ssdf_error e1;
    Handle *a = ssdf_open_iv(pa, threads, qd, io_nr, cpuA, iv_a, 2, &e1);
    if (!a) {
        ssdf_set_err(err, SSDF_ERR_SETUP, "dual first open failed: %s", e1.detail);
        return NULL;
    }
    uint64_t iv_b[2] = {0, rows2};
    ssdf_error e2;
    Handle *b = ssdf_open_iv(pb, threads, qd, io_nr, cpuB, iv_b, 1, &e2);
    if (!b) {
        /* 第二盘失败: 完整回收第一盘(线程 join + 资源释放) */
        pthread_mutex_lock(&a->mu);
        a->shutdown = 1; a->gen++;
        pthread_cond_broadcast(&a->go);
        pthread_mutex_unlock(&a->mu);
        for (int i = 0; i < a->nthreads_started; i++) pthread_join(a->th[i], NULL);
        handle_free(a);
        ssdf_set_err(err, SSDF_ERR_SETUP, "dual second open failed, first unwound: %s", e2.detail);
        return NULL;
    }
    a->other = b; a->base1 = base1; a->rows1 = rows1; a->rows2 = rows2;
    return a;
}

/* ---------- close(幂等, 含 dual 另一盘) ---------- */

static void close_one(Handle *h) {
    pthread_mutex_lock(&h->fill_mu);
    if (h->closed) { pthread_mutex_unlock(&h->fill_mu); return; }
    h->closed = 1;
    pthread_mutex_unlock(&h->fill_mu);
    pthread_mutex_lock(&h->mu);
    h->shutdown = 1; h->gen++;
    pthread_cond_broadcast(&h->go);
    pthread_mutex_unlock(&h->mu);
    for (int i = 0; i < h->nthreads_started; i++) pthread_join(h->th[i], NULL);
    handle_free(h);   /* fill_mu 随 handle 一起销毁; 此后外部不得再触碰句柄 */
}

void ssdf_close(void *hv) {
    Handle *h = (Handle *)hv;
    if (!h) return;
    Handle *b = h->other;
    h->other = NULL;
    if (b) close_one(b);
    close_one(h);
}

/* ---------- stats ---------- */

void ssdf_stats(void *hv, int64_t *out, int64_t n_out) {
    Handle *h = (Handle *)hv;
    if (!h || !out || n_out <= 0) return;
    pthread_mutex_lock(&h->mu);
    int64_t v[10] = {h->rows_total, h->ge_calls, h->ge_events, h->fails, h->stale,
                     h->partial_submits, h->eagain, h->eintr, h->poisoned, 0};
    int64_t n = n_out < 10 ? n_out : 10;
    memcpy(out, v, sizeof(int64_t) * (size_t)n);
    pthread_mutex_unlock(&h->mu);
}

void ssdf_stats_ms(void *hv, double *out, int64_t n_out) {
    Handle *h = (Handle *)hv;
    if (!h || !out || n_out <= 0) return;
    pthread_mutex_lock(&h->mu);
    double v[2] = {h->extract_ms, h->submit_ms};
    int64_t n = n_out < 2 ? n_out : 2;
    memcpy(out, v, sizeof(double) * (size_t)n);
    pthread_mutex_unlock(&h->mu);
}

void ssdf_version(char *buf, int64_t buflen) {
    snprintf(buf, (size_t)buflen, "ssd_fill reliable v2 (gen-guarded, strict-res)");
}
