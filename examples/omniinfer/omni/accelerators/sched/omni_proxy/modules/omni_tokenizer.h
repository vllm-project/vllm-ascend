// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct
{
    bool failed;
    const char *input_data;
    size_t input_len;

    char *prompt;
    size_t prompt_buf_size;
    size_t prompt_len;

    int64_t *input_ids;
    size_t input_ids_buf_size;
    size_t input_ids_len;

    int64_t *block_hashes;
    size_t block_hashes_buf_size;
    size_t block_hashes_len;

    int multi_modal_size;
} omni_tokenizer_request;

int omni_tokenizer_init();
void omni_tokenizer_cleanup();
int omni_init_tokenizer(const char *model_path);
int omni_batch_chat_encode(omni_tokenizer_request **requests, size_t num_reqs);
void print_tokenize_result(omni_tokenizer_request *requests);
