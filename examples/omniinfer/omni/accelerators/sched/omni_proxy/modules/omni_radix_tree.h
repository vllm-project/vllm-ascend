// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

#pragma once

#include <ngx_config.h>
#include <ngx_core.h>
#include <ngx_http.h>

typedef struct omni_radix_node_s omni_radix_node_t;

struct omni_radix_node_s
{
    ngx_rbtree_node_t rb_node;
    uint64_t hash_key;
    omni_radix_node_t *first_child;
    omni_radix_node_t *next_sibling;
    omni_radix_node_t *parent; /* 新增：指向父节点，便于 unlink */
};

typedef struct
{
    ngx_slab_pool_t *shpool;
    omni_radix_node_t *root;
    ngx_rbtree_t lookup_tree;
    ngx_rbtree_node_t sentinel;
    ngx_shmtx_t mutex;
    ngx_shmtx_sh_t lock;
    ngx_uint_t writer_pid;
    ngx_uint_t version;
} omni_radix_tree_t;

omni_radix_tree_t *omni_radix_tree_init(ngx_slab_pool_t *shpool);

ngx_int_t omni_radix_tree_add_chain(omni_radix_tree_t *tree,
                                   uint64_t *hash_chain,
                                   ngx_uint_t chain_len);

ngx_uint_t omni_radix_tree_match(omni_radix_tree_t *tree,
                                uint64_t *hash_chain,
                                ngx_uint_t chain_len);

ngx_uint_t omni_radix_tree_match_optimistic(omni_radix_tree_t *tree,
                                            uint64_t *hash_chain,
                                            ngx_uint_t chain_len);

/* Remove all nodes whose node->hash_key == hash_to_remove (and their subtrees).
   Returns NGX_OK if at least one node removed; NGX_ERROR if none found.
*/
ngx_int_t omni_radix_tree_remove(omni_radix_tree_t *tree,
                                uint64_t hash_to_remove);

void omni_radix_tree_destroy(omni_radix_tree_t *tree);

void omni_radix_tree_test(omni_radix_tree_t *tree);