#ifndef ASYNC_WORKER_H
#define ASYNC_WORKER_H

#include <stdint.h>
#include <stddef.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

// 异步任务类型
typedef enum {
    ASYNC_TASK_WARMUP,
    ASYNC_TASK_UPDATE,
} AsyncTaskType;

// 异步任务
typedef struct {
    AsyncTaskType type;
    uint64_t req_hash;
    uint32_t* tokens;       // 拥有所有权，需要 free
    size_t token_count;
} AsyncTask;

// 任务队列（环形缓冲区）
#define ASYNC_QUEUE_CAPACITY 4096

typedef struct {
    AsyncTask tasks[ASYNC_QUEUE_CAPACITY];
    volatile size_t head;   // 消费者读
    volatile size_t tail;   // 生产者写
    size_t count;           // 当前队列中的任务数（仅生产者维护）
} AsyncQueue;

// 异步工作线程
typedef struct {
    pthread_t thread;
    pthread_mutex_t mutex;
    pthread_cond_t cond_work;    // 有新任务
    pthread_cond_t cond_done;    // 任务全部完成
    AsyncQueue queue;
    volatile int running;        // 线程是否运行
    volatile int pending;        // 待处理任务数（原子递增/递减）
    void* cache;                 // ZipfCache* (避免循环依赖用 void*)
} AsyncWorker;

// 创建异步工作线程
AsyncWorker* async_worker_create(void* cache);

// 提交任务（非阻塞，拷贝 tokens 数据）
void async_worker_submit_warmup(AsyncWorker* worker, uint64_t req_hash,
                                const uint32_t* tokens, size_t count);
void async_worker_submit_update(AsyncWorker* worker, uint64_t req_hash,
                                const uint32_t* tokens, size_t count);

// 等待所有待处理任务完成（fence）
void async_worker_fence(AsyncWorker* worker);

// 销毁
void async_worker_free(AsyncWorker* worker);

#ifdef __cplusplus
}
#endif

#endif // ASYNC_WORKER_H
