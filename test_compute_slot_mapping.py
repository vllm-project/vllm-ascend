import numpy as np

dcp_world_size = 2
pcp_world_size = 2
block_size = 2
max_num_blocks_per_req = 1
blocks_per_phys_block = 2
cp_kv_cache_interleave_size = 1
pcp_rank = 1
dcp_rank = 1
block_numbers = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] # 一个请求 该请求被分配到第一个块
# 三种size phy size = 16，block_size = 2， logical_size = 8

req_indices = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) # 10个token都属于第一个请求
positions = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) # token的位置分别是0-9

def compute_slot_mapping(req_indices: np.ndarray, positions: np.ndarray) -> None:
    # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
    # -> [0, 0, K, K, K + 1, K + 1, K + 2, 2 * K, 2 * K, 2 * K + 1]
    # where K is the max_num_blocks_per_req and the block size is 2.
    # NOTE(woosuk): We can't simply use `token_indices // block_size`
    # here because M (max_model_len) is not necessarily divisible by
    # block_size.

    if dcp_world_size * pcp_world_size > 1:
        # Note(hc): The DCP implement store kvcache with an interleave
        # style, the kvcache for the token whose token_idx is i is
        # always stored on the GPU whose dcp_rank equals i % pcp_world_size:

        # Use a "virtual block" which equals to world_size * block_size
        # for block_table_indices calculation.
        virtual_block_size = block_size * dcp_world_size * pcp_world_size
        print(f"virtual_block_size: {virtual_block_size}")

        # IMPORTANT: In hybrid mode, positions are in logical block space,
        # but we need to map them to the correct logical block table indices
        logical_block_idx = positions // virtual_block_size
        print(f"logical_block_idx: {logical_block_idx}")

        # Account for the expanded logical table
        # (always needed with unified tensor)
        # Each physical block is split into multiple logical blocks
        # The logical table has been expanded to accommodate this
        block_table_indices = (
            req_indices * max_num_blocks_per_req * blocks_per_phys_block + logical_block_idx
        )
        print(f"block_table_indices: {block_table_indices}")

        # Use virtual_block_size for mask calculation, which marks local
        # tokens.
        virtual_block_offsets = positions % virtual_block_size
        print(f"virtual_block_offsets: {virtual_block_offsets}")
        current_rank = dcp_world_size * pcp_rank + dcp_rank
        print(f"current_rank: {current_rank}")
        mask = (
            virtual_block_offsets // cp_kv_cache_interleave_size % (dcp_world_size * pcp_world_size)
            == current_rank
        )
        print(f"mask: {mask}")
        # Calculate local block_offsets
        block_offsets = (
            virtual_block_offsets
            // (dcp_world_size * pcp_world_size * cp_kv_cache_interleave_size)
            * cp_kv_cache_interleave_size
            + virtual_block_offsets % cp_kv_cache_interleave_size
        )
        print(f"block_offsets: {block_offsets}")
        # Calculate slot_mapping
        slot_mapping = block_numbers * block_size + block_offsets
        print(f"slot_mapping: {slot_mapping} {slot_mapping.shape}")
        
        # Write final slots, use -1 for not-local
        print(slot_mapping[: req_indices.shape[0]].shape) 
        print(np.where(mask, slot_mapping, -1).shape)
        print(mask.shape)
        slot_mapping[: req_indices.shape[0]] = np.where(mask, slot_mapping, -1)
        print(f"slot_mapping: {slot_mapping} {slot_mapping.shape}")

compute_slot_mapping(req_indices, positions)