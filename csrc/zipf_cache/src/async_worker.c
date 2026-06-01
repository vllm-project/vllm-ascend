#include "async_worker.h"
#include "zipf_cache.h"
#include <stdlib.h>
#include <string.h>

// ============================================================
// 后台线程函数
// ============================================================

static void process_task(AsyncWorker* worker, const AsyncTask* task) {
    ZipfCache* cache = (ZipfCache*)worker->cache;
    switch (task->type) {
        case ASYNC_TASK_WARMUP:
            zipf_cache_warmup_from_prompt(cache, task->req_hash,
                                          task->tokens, task->token_count);
            break;
        case ASYNC_TASK_UPDATE:
            zipf_cache_update_with_history(cache, task->req_hash,
                                           task->tokens, task->token_count);
            break;
    }
    free(task->tokens);
}

static void* worker_thread_func(void* arg) {
    AsyncWorker* worker = (AsyncWorker*)arg;

    while (1) {
        pthread_mutex_lock(&worker->mutex);

        // 等待任务或退出信号
        while (worker->queue.head == worker->queue.tail && worker->running) {
            pthread_cond_wait(&worker->cond_work, &worker->mutex);
        }

        if (!worker->running && worker->queue.head == worker->queue.tail) {
            pthread_mutex_unlock(&worker->mutex);
            break;
        }

        // 取出一个任务
        size_t head = worker->queue.head;
        AsyncTask task = worker->queue.tasks[head % ASYNC_QUEUE_CAPACITY];
        worker->queue.head = head + 1;

        pthread_mutex_unlock(&worker->mutex);

        // 执行任务（不持锁）
        process_task(worker, &task);

        // 递减 pending 计数，通知 fence
        pthread_mutex_lock(&worker->mutex);
        worker->pending--;
        if (worker->pending == 0) {
            pthread_cond_broadcast(&worker->cond_done);
        }
        pthread_mutex_unlock(&worker->mutex);
    }

    return NULL;
}

// ============================================================
// 公共 API
// ============================================================

AsyncWorker* async_worker_create(void* cache) {
    AsyncWorker* worker = calloc(1, sizeof(AsyncWorker));
    if (!worker) return NULL;

    worker->cache = cache;
    worker->running = 1;
    worker->pending = 0;
    worker->queue.head = 0;
    worker->queue.tail = 0;
    worker->queue.count = 0;

    pthread_mutex_init(&worker->mutex, NULL);
    pthread_cond_init(&worker->cond_work, NULL);
    pthread_cond_init(&worker->cond_done, NULL);

    if (pthread_create(&worker->thread, NULL, worker_thread_func, worker) != 0) {
        pthread_mutex_destroy(&worker->mutex);
        pthread_cond_destroy(&worker->cond_work);
        pthread_cond_destroy(&worker->cond_done);
        free(worker);
        return NULL;
    }

    return worker;
}

static void submit_task(AsyncWorker* worker, AsyncTaskType type,
                        uint64_t req_hash, const uint32_t* tokens, size_t count) {
    if (!worker || !tokens || count == 0) return;

    // 拷贝 token 数据（生产者拥有原始数据的生命周期不确定）
    uint32_t* tokens_copy = malloc(count * sizeof(uint32_t));
    if (!tokens_copy) return;
    memcpy(tokens_copy, tokens, count * sizeof(uint32_t));

    pthread_mutex_lock(&worker->mutex);

    size_t tail = worker->queue.tail;
    // 如果队列满了，丢弃（不应该发生，4096 足够大）
    if (tail - worker->queue.head >= ASYNC_QUEUE_CAPACITY) {
        pthread_mutex_unlock(&worker->mutex);
        free(tokens_copy);
        return;
    }

    AsyncTask* task = &worker->queue.tasks[tail % ASYNC_QUEUE_CAPACITY];
    task->type = type;
    task->req_hash = req_hash;
    task->tokens = tokens_copy;
    task->token_count = count;

    worker->queue.tail = tail + 1;
    worker->pending++;

    pthread_cond_signal(&worker->cond_work);
    pthread_mutex_unlock(&worker->mutex);
}

void async_worker_submit_warmup(AsyncWorker* worker, uint64_t req_hash,
                                const uint32_t* tokens, size_t count) {
    submit_task(worker, ASYNC_TASK_WARMUP, req_hash, tokens, count);
}

void async_worker_submit_update(AsyncWorker* worker, uint64_t req_hash,
                                const uint32_t* tokens, size_t count) {
    submit_task(worker, ASYNC_TASK_UPDATE, req_hash, tokens, count);
}

void async_worker_fence(AsyncWorker* worker) {
    if (!worker) return;
    pthread_mutex_lock(&worker->mutex);
    while (worker->pending > 0) {
        pthread_cond_wait(&worker->cond_done, &worker->mutex);
    }
    pthread_mutex_unlock(&worker->mutex);
}

void async_worker_free(AsyncWorker* worker) {
    if (!worker) return;

    // 停止线程
    pthread_mutex_lock(&worker->mutex);
    worker->running = 0;
    pthread_cond_signal(&worker->cond_work);
    pthread_mutex_unlock(&worker->mutex);

    pthread_join(worker->thread, NULL);

    // 清理未处理的任务
    while (worker->queue.tail - worker->queue.head > 0) {
        size_t head = worker->queue.head;
        AsyncTask* task = &worker->queue.tasks[head % ASYNC_QUEUE_CAPACITY];
        free(task->tokens);
        worker->queue.head = head + 1;
    }

    pthread_mutex_destroy(&worker->mutex);
    pthread_cond_destroy(&worker->cond_work);
    pthread_cond_destroy(&worker->cond_done);
    free(worker);
}
