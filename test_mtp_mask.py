#!/usr/bin/env python3
# 测试修改后的 MTP attention mask 生成逻辑 - v7 (正确版本)

class MockPCPManager:
    def __init__(self, pcp_world_size, pcp_rank, dcp_world_size=1, dcp_rank=0):
        self.pcp_world_size = pcp_world_size
        self.pcp_world_rank = pcp_rank
        self.dcp_world_size = dcp_world_size
        self.dcp_rank = dcp_rank


def generate_mtp_attention_mask_v7(manager, history_len, mtp_len):
    """
    正确的分配逻辑：
    1. 哪个 rank 短就补哪个
    2. 如果长度相等，按 token 索引奇偶分配（偶数给当前 rank，奇数给其他 rank）

    历史切分 (DualChunkSwap):
    - cp0: [a,c,e] (3个), positions [0,2,4]
    - cp1: [b,d] (2个), positions [1,3]

    MTP 添加过程:
    - 初始: cp0=3, cp1=2 -> cp1 短
    - f: cp1 更短 -> f 到 cp1 -> cp0=3, cp1=3
    - g: 相等 -> g 索引 1 是奇数 -> cp0 更长，优先补 cp1 -> 不对

    让我重新分析：
    - 初始: cp0=3, cp1=2
    - f: cp1 更短 -> f 到 cp1 -> cp0=3, cp1=3
    - g: cp0=3, cp1=3 -> 相等 -> 按 token 索引分配，g 是第 2 个 (idx=1)，奇数
      但此时 cp0 之前更长，应该给 cp0，所以逻辑应该是：
      相等时，看之前哪个更长，之前长的这次先补

    实际上，让我直接验证用户的期望：
    - cp0: [a,c,e,g,i] (5个) -> positions [0,2,4,6,8]
    - cp1: [b,d,f,h] (4个) -> positions [1,3,5,7]

    分配过程反推：
    - 初始: cp0=3, cp1=2
    - f: cp1 短 -> f 到 cp1 (pos 5)
    - g: 相等 -> cp0 更长，先补 cp0 -> g 到 cp0 (pos 6)
    - h: cp0=4, cp1=3 -> cp1 短 -> h 到 cp1 (pos 7)
    - i: 相等 -> cp0 更长，先补 cp0 -> i 到 cp0 (pos 8)

    结论：相等时，看之前哪个 rank 更长（历史更长），之前长的先补
    """
    cp_rank = manager.pcp_world_rank * manager.dcp_world_size + manager.dcp_rank

    print(f"history_len={history_len}, mtp_len={mtp_len}, cp_rank={cp_rank}")

    # 历史 token 分布
    if cp_rank % 2 == 0:
        local_history_positions = [p for p in range(history_len) if p % 2 == 0]
    else:
        local_history_positions = [p for p in range(history_len) if p % 2 == 1]

    print(f"本 rank 历史位置: {local_history_positions}")

    # 计算每个 rank 的历史长度
    if history_len % 2 == 0:
        even_rank_history = history_len // 2
        odd_rank_history = history_len // 2
    else:
        even_rank_history = history_len // 2 + 1
        odd_rank_history = history_len // 2

    if cp_rank % 2 == 0:
        current_rank_len = even_rank_history
        other_rank_len = odd_rank_history
    else:
        current_rank_len = odd_rank_history
        other_rank_len = even_rank_history

    print(f"初始: current={current_rank_len}, other={other_rank_len}")

    local_all_positions = list(local_history_positions)

    for mtp_idx in range(mtp_len):
        if current_rank_len < other_rank_len:
            # 当前 rank 更短
            local_all_positions.append(history_len + mtp_idx)
            current_rank_len += 1
            print(f"MTP {mtp_idx} -> 本 rank (短)")
        elif current_rank_len > other_rank_len:
            # 其他 rank 更短
            other_rank_len += 1
            print(f"MTP {mtp_idx} -> 其他 rank (短)")
        else:
            # 相等时，之前更长的 rank 优先
            if cp_rank % 2 == 0:
                # 当前 rank (even) 历史更长，优先
                local_all_positions.append(history_len + mtp_idx)
                current_rank_len += 1
                print(f"MTP {mtp_idx} -> 本 rank (相等，even优先)")
            else:
                # 当前 rank (odd) 历史更短，其他 rank 优先
                other_rank_len += 1
                print(f"MTP {mtp_idx} -> 其他 rank (相等，even优先)")

    print(f"本地位置: {local_all_positions}")

    # 生成 mask
    mtp_positions = list(range(history_len, history_len + mtp_len))
    local_len = len(local_all_positions)
    mask = [[False for _ in range(local_len)] for _ in range(mtp_len)]
    for m_idx, mtp_global_pos in enumerate(mtp_positions):
        for k_idx, local_global_pos in enumerate(local_all_positions):
            mask[m_idx][k_idx] = (mtp_global_pos >= local_global_pos)

    print(f"\nRank {cp_rank} mask ({mtp_len}x{local_len}):")
    for row in mask:
        print([1 if x else 0 for x in row])

    return mask


print("=" * 60)
print("cp0 (rank 0)")
print("=" * 60)
manager0 = MockPCPManager(pcp_world_size=2, pcp_rank=0)
mask0 = generate_mtp_attention_mask_v7(manager0, history_len=5, mtp_len=4)

print("\n" + "=" * 60)
print("cp1 (rank 1)")
print("=" * 60)
manager1 = MockPCPManager(pcp_world_size=2, pcp_rank=1)
mask1 = generate_mtp_attention_mask_v7(manager1, history_len=5, mtp_len=4)

print("\n期望:")
print("cp0: [0,2,4,6,8] -> mask 4x5")
print("cp1: [1,3,5,7] -> mask 4x4")