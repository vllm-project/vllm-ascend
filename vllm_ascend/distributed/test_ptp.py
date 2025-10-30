def init_ascend_model_parallel(parallel_config: ParallelConfig, ):
    if model_parallel_initialized():
        return
    assert torch.distributed.is_initialized()
    world_size = torch.distributed.get_world_size()
    backend = torch.distributed.get_backend(get_world_group().device_group)

    # The layout of all ranks: ExternalDP * EP
    # ExternalDP is the data parallel group that is not part of the model,
    # every dp rank can generate independently (in verl integration).
    all_ranks = torch.arange(world_size).reshape(
        -1, parallel_config.data_parallel_size *
        parallel_config.tensor_parallel_size)

    pd_tp_ratio = get_ascend_config().pd_tp_ratio
    pd_head_ratio = get_ascend_config().pd_head_ratio
    global _P_TP
    assert _P_TP is None, (
        "distributed prefill tensor parallel group is already initialized")
    prefill_tensor_model_parallel_size = pd_tp_ratio
    # divide alltoall groups
    if pd_head_ratio > 1 and get_current_vllm_config(
    ).kv_transfer_config.is_kv_producer:
        num_head_replica = get_ascend_config().num_head_replica
        remote_tp_size = parallel_config.tensor_parallel_size // pd_tp_ratio
        if num_head_replica <= 1:
            group_ranks = all_ranks.view(
                -1, prefill_tensor_model_parallel_size).unbind(0)
        else:
            group_ranks = all_ranks.clone().view(
                parallel_config.data_parallel_size, -1,
                num_head_replica)  # [DP_size, num_head, num_head_replica]
            group_ranks = group_ranks.permute(0, 2, 1)
            group_ranks = group_ranks.reshape(
                -1,
                group_ranks.size(-1))  # [DP_size * num_head_replica, num_head]
            alltoall_group_size = group_ranks.size(-1) // remote_tp_size
            group_ranks = group_ranks.unsqueeze(-1).view(
                parallel_config.data_parallel_size, num_head_replica, -1,
                alltoall_group_size
            )  # [DP_size, num_head_replica, num_alltoall_group, alltoall_group_size]
            group_ranks = group_ranks.reshape(-1,
                                              alltoall_group_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        local_rank = get_world_group().local_rank
        num = next(
            (i for i, ranks in enumerate(group_ranks) if local_rank in ranks),
            None)
        _P_TP = init_model_parallel_group(group_ranks,
                                          get_world_group().local_rank,
                                          backend,
                                          group_name=f"p_tp_{num}")
