import os

import pytest
from pytest_mock import MockerFixture
import torch
import torch_npu
import llm_datadist

from vllm_ascend.distributed.llmdatadist_connector import (
    get_device_ips,
    KVTransferEngine,
    LLMDataDistConnector,
)


def test_get_device_ips(mocker: MockerFixture):
    # curent world_size in func get_device_ips is 8
    # should change this var when origin fun change it's world_size
    world_size = 8
    ip_list = [
        '124.0.2.90', '124.0.2.91', '124.0.2.92', '124.0.2.93', '124.0.2.94',
        '124.0.2.95', '124.0.2.96', '124.0.2.97'
    ]
    generator_ip_list = (x for x in ip_list)

    def subprocess_run(cmd, stdout, stderr, universal_newlines=False):
        if "npu-smi" in cmd:
            mock_npu_info = mocker.MagicMock()
            mock_npu_info.returncode = 0
            return mock_npu_info
        else:
            mock_device_ip = mocker.MagicMock()
            return mock_device_ip

    def re_match(pattern, string):
        if "ipaddr" not in pattern:
            mock_npu_info = mocker.MagicMock()
            mock_npu_info.group.return_value = "0"
            return mock_npu_info
        else:
            mock_device_ip = mocker.MagicMock()
            mock_device_ip.group.return_value = next(generator_ip_list)
            return mock_device_ip

    mock_npu_info = mocker.MagicMock()
    mock_npu_info.returncode = 0

    mock_subprocess_run = mocker.patch("subprocess.run",
                                       side_effect=subprocess_run)
    mock_re_match = mocker.patch("re.match", side_effect=re_match)

    device_ip_list = get_device_ips()

    assert mock_subprocess_run.call_count == 1 + world_size
    assert mock_re_match.call_count == 1 + world_size
    assert device_ip_list == ip_list


class TestKVTransferEngine():

    @staticmethod
    def mock_and_generate_kv_transfer_engine(
        world_size,
        n_layer,
        role,
        local_rank,
        mocker: MockerFixture,
    ):
        mock_get_device_ips = mocker.patch(
            "vllm_ascend.distributed.llmdatadist_connector.get_device_ips",
            return_value=[
                '124.0.2.90', '124.0.2.91', '124.0.2.92', '124.0.2.93',
                '124.0.2.94', '124.0.2.95', '124.0.2.96', '124.0.2.97'
            ])
        mock_llm_datadist = mocker.patch("llm_datadist.LLMDataDist")

        os.environ["PROMPT_DEVICE_ID"] = "0, 1, 2, 3"
        os.environ["DECODE_DEVICE_ID"] = "4, 5, 6, 7"

        kv_transfer_engine = KVTransferEngine(world_size, n_layer, role,
                                              local_rank)

        return (
            kv_transfer_engine,
            mock_get_device_ips,
            mock_llm_datadist,
        )

    def test_init(self, mocker: MockerFixture):
        kv_transfer_engine, mock_get_device_ips, mock_llm_datadist = (
            self.mock_and_generate_kv_transfer_engine(
                8, 64, llm_datadist.LLMRole.PROMPT, 0, mocker))

        mock_get_device_ips.assert_called_once()
        mock_llm_datadist.assert_called_once()
        # call args should is ("kv_producer", 0)
        assert mock_llm_datadist.call_args[0][0] == llm_datadist.LLMRole.PROMPT
        assert mock_llm_datadist.call_args[0][1] == 0
        
        assert kv_transfer_engine.prompt_ip_list == [
            '124.0.2.90', '124.0.2.91', '124.0.2.92', '124.0.2.93'
        ]
        assert kv_transfer_engine.decode_ip_list == [
            '124.0.2.94', '124.0.2.95', '124.0.2.96', '124.0.2.97'
        ]

    def test_prepare_data_dist_role_is_PROMPT(self, mocker: MockerFixture):
        world_size = 8
        n_layer = 64
        role = llm_datadist.LLMRole.PROMPT
        local_rank = 1

        kv_transfer_engine, _, _ = self.mock_and_generate_kv_transfer_engine(
            world_size, n_layer, role, local_rank, mocker)
        os.environ["LLMDATADIST_SYNC_CACHE_WAIT_TIME"] = "0"
        mock_data_dist = mocker.patch.object(
            kv_transfer_engine.data_dist,
            "init",
        )
        mock_kv_transfer = mocker.patch.object(
            kv_transfer_engine.data_dist,
            "kv_cache_manager",
        )

        kv_transfer_engine.prepare_data_dist()

        mock_data_dist.assert_called_once()
        assert kv_transfer_engine.kv_transfer

        options = mock_data_dist.call_args[0][0]
        assert options["ge.exec.deviceId"] == str(local_rank)
        assert "llm.listenIpInfo" in options

    def test_prepare_data_dist_role_is_not_PROMPT(self, mocker: MockerFixture):
        world_size = 8
        n_layer = 64
        role = llm_datadist.LLMRole.DECODER
        local_rank = 1

        kv_transfer_engine, _, _ = self.mock_and_generate_kv_transfer_engine(
            world_size, n_layer, role, local_rank, mocker)
        os.environ["LLMDATADIST_SYNC_CACHE_WAIT_TIME"] = "0"
        mock_data_dist = mocker.patch.object(
            kv_transfer_engine.data_dist,
            "init",
        )
        mocker.patch.object(
            kv_transfer_engine.data_dist,
            "kv_cache_manager",
        )

        kv_transfer_engine.prepare_data_dist()

        mock_data_dist.assert_called_once()
        assert kv_transfer_engine.kv_transfer

        options = mock_data_dist.call_args[0][0]
        assert options["ge.exec.deviceId"] == str(local_rank)
        assert "llm.listenIpInfo" not in options

    def test_make_cluster(self, mocker: MockerFixture):
        world_size = 8
        n_layer = 64
        role = None
        local_rank = 1

        kv_transfer_engine, _, _ = self.mock_and_generate_kv_transfer_engine(
            world_size, n_layer, role, local_rank, mocker)

        mock_obj = mocker.MagicMock()
        mock_LLMClusterInfo = mocker.patch(
            "llm_datadist.LLMClusterInfo",
            return_value=mock_obj,
        )

        prefill_ip = "192.168.1.1"
        cluster_id = 3
        cluster = kv_transfer_engine.make_cluster(prefill_ip, cluster_id)

        mock_LLMClusterInfo.assert_called_once()
        assert cluster.remote_cluster_id == cluster_id
        cluster.append_local_ip_info.assert_called_once()
        cluster.append_remote_ip_info.assert_called_once()


class TestLLMDataDistConnector():

    @staticmethod
    def mock_and_generate_LLMDataDistConnector(
        kv_role,
        tensor_parallel_size,
        world_size,
        num_layers,
        rank,
        local_rank,
        mocker: MockerFixture,
    ):
        mock_config = mocker.MagicMock()
        mock_config.kv_transfer_config.kv_role = kv_role
        mock_config.parallel_config.tensor_parallel_size = tensor_parallel_size
        mock_config.parallel_config.world_size = world_size
        mock_config.model_config.get_num_layers.return_value = num_layers

        mock_llm_datadist = mocker.MagicMock()
        mock_llm_datadist.link_clusters.return_value = (None, None)
        mock_kv_transfer = mocker.MagicMock()
        mock_kv_transfer.data_dist = mock_llm_datadist

        mocker.patch(
            "vllm_ascend.distributed.llmdatadist_connector.KVTransferEngine",
            return_value=mock_kv_transfer)

        test_instance = LLMDataDistConnector(
            rank,
            local_rank,
            mock_config,
        )

        return test_instance

    def test_init_kv_producer(self, mocker: MockerFixture):
        # set config
        kv_role = "kv_producer"
        tensor_parallel_size = 2
        world_size = 16
        num_layers = 64

        rank = 8
        local_rank = 0

        test_instance = self.mock_and_generate_LLMDataDistConnector(
            kv_role, tensor_parallel_size, world_size, num_layers, rank,
            local_rank, mocker)

        assert test_instance.tp_size == tensor_parallel_size
        assert test_instance.rank == rank
        assert test_instance.local_rank == local_rank
        assert test_instance.role == llm_datadist.LLMRole.PROMPT
        assert test_instance.world_size == world_size
        assert test_instance.n_layer == num_layers
        test_instance.llm_datadist_engine.prepare_data_dist.assert_called_once(
        )
        test_instance.llm_datadist_engine.make_cluster.assert_not_called()

    def test_init_kv_consumer(self, mocker: MockerFixture):
        # set config
        kv_role = "kv_consumer"
        tensor_parallel_size = 2
        world_size = 16
        num_layers = 64

        rank = 8
        local_rank = 0

        test_instance = self.mock_and_generate_LLMDataDistConnector(
            kv_role, tensor_parallel_size, world_size, num_layers, rank,
            local_rank, mocker)

        assert test_instance.tp_size == tensor_parallel_size
        assert test_instance.rank == rank
        assert test_instance.local_rank == local_rank
        assert test_instance.role == llm_datadist.LLMRole.DECODER
        assert test_instance.world_size == world_size
        assert test_instance.n_layer == num_layers
        test_instance.llm_datadist_engine.prepare_data_dist.assert_called_once(
        )
        test_instance.llm_datadist_engine.make_cluster.assert_called_once()
        test_instance.llm_datadist_engine.data_dist.link_clusters.assert_called_once(
        )

    def test_init_should_raise_error(self, mocker: MockerFixture):
        # set config
        kv_role = " "
        tensor_parallel_size = 2
        world_size = 16
        num_layers = 64

        rank = 8
        local_rank = 0

        with pytest.raises(NotImplementedError) as error:
            test_instance = self.mock_and_generate_LLMDataDistConnector(
                kv_role, tensor_parallel_size, world_size, num_layers, rank,
                local_rank, mocker)
        assert "kv_role should be inside" in str(error.value)

    def test_send_kv_caches_and_hidden_states(self, mocker: MockerFixture):
        # set config
        kv_role = "kv_producer"
        tensor_parallel_size = 2
        world_size = 8
        num_layers = 4
        num_key_value_heads = 2
        hidden_size = 32 * 4
        num_attention_heads = 4
        head_size = int(hidden_size / num_attention_heads)
        kv_cache_num_head = int(num_key_value_heads / tensor_parallel_size)
        dtype = torch.int32
        tokens_num = 16
        start_layer = 0
        end_layer = 2
        seq_lens = [1, 2, 3, 4]

        rank = 8
        local_rank = 0

        # create instance
        test_instance = self.mock_and_generate_LLMDataDistConnector(
            kv_role, tensor_parallel_size, world_size, num_layers, rank,
            local_rank, mocker)

        # should call mock_cache_desc and mock_cache_key 4 times
        # cache_desc_return should have shape attribute
        def mock_make_cache_desc(num_layer, data_shape, data_type,
                                 seq_len_dim_index):
            cache_desc_return = mocker.MagicMock()
            cache_desc_return.shape = data_shape
            return cache_desc_return

        mock_cache_desc = mocker.patch("llm_datadist.CacheDesc",
                                       side_effect=mock_make_cache_desc)
        mock_cache_key = mocker.patch("llm_datadist.CacheKey",
                                      return_value=None)
        # should call mock_kv_transfer.allocate_cache 4 times
        mock_kv_transfer = mocker.MagicMock()
        mock_buffer = mocker.MagicMock()
        mocker.patch(
            "vllm_ascend.distributed.llmdatadist_connector.KVTransferEngine",
            return_value=mock_kv_transfer)
        # should have mock_buffer.per_device_tensor_addrs attribute
        mock_kv_transfer.allocate_cache.return_value = mock_buffer

        # should call 4 times
        def create_npu_tensors(data_shape, kv_hidden_dtype, key_buffer_addr):
            new_kvcache = [torch.zeros(list(data_shape))] * num_layers
            return new_kvcache

        mock_create_npu_tensors = mocker.patch(
            "torchair.llm_datadist.create_npu_tensors",
            side_effect=create_npu_tensors)

        # construct test function input
        mock_model = mocker.MagicMock()
        mock_model_input = mocker.MagicMock()
        mock_model_config = mocker.MagicMock()

        mock_model_input.input_tokens = torch.arange(10).to(dtype)
        mock_model_input.attn_metadata.seq_lens = seq_lens
        mock_model_input.attn_metadata.slot_mapping = torch.arange(9, -1, -1)

        mock_model.model.start_layer = start_layer
        mock_model.model.end_layer = end_layer
        mock_model.model.config = mock_model_config

        mock_model_config.num_key_value_heads = num_key_value_heads
        mock_model_config.hidden_size = hidden_size
        mock_model_config.num_attention_heads = num_attention_heads

        kv_caches = [
            torch.arange(2 * tokens_num * kv_cache_num_head * head_size).view(
                2, -1, kv_cache_num_head, head_size).to(dtype)
        ] * num_layers

        hidden_or_intermediate_states = (torch.arange(
            tokens_num * hidden_size).view([tokens_num,
                                            hidden_size]).to(dtype))

        # should call mock_indices.npu 1 time
        mock_indices = mocker.MagicMock()
        mocker.patch("torch.tensor", return_value=mock_indices)

        # should call len(seq_lens) * (2 * (end_layer - start_layer) + 2)
        mock_scatter_update_ = mocker.patch("torch_npu.scatter_update_")

        # mock torch.distributed
        mock_dis = mocker.patch("torch.distributed.get_rank", return_value=1)

        test_instance.send_kv_caches_and_hidden_states(
            mock_model, mock_model_input, kv_caches,
            hidden_or_intermediate_states)

        assert mock_cache_desc.call_count == 4
        assert mock_cache_key.call_count == 4
        assert mock_buffer.per_device_tensor_addrs
        assert mock_create_npu_tensors.call_count == 4
        assert mock_indices.npu
        scatter_update_call_times = len(seq_lens) * (
            2 * (end_layer - start_layer) + 2)
        assert mock_scatter_update_.call_count == scatter_update_call_times

        for call_args in mock_scatter_update_.call_args_list:
            scatter_update_input0 = call_args[0][0]
            scatter_update_input1 = call_args[0][2]
            assert scatter_update_input0.shape == scatter_update_input1.shape

        mock_dis.assert_called_once()

    def test_recv_kv_caches_and_hidden_states(self, mocker: MockerFixture):
        # set config
        kv_role = "kv_consumer"
        tensor_parallel_size = 2
        world_size = 8
        num_layers = 4
        num_key_value_heads = 2
        hidden_size = 32 * 8
        num_attention_heads = 4
        head_size = int(hidden_size / num_attention_heads)
        kv_cache_num_head = int(num_key_value_heads / tensor_parallel_size)
        dtype = torch.int32
        tokens_num = 16
        start_layer = 0
        end_layer = 2
        seq_lens = [1, 2, 3, 4, 5]
        total_token = sum(seq_lens)

        rank = 8
        local_rank = 0

        # create instance
        test_instance = self.mock_and_generate_LLMDataDistConnector(
            kv_role, tensor_parallel_size, world_size, num_layers, rank,
            local_rank, mocker)

        # should call mock_cache_desc and mock_cache_key 4 times
        # cache_desc_return should have shape attribute
        def mock_make_cache_desc(num_layer, data_shape, data_type,
                                 seq_len_dim_index):
            cache_desc_return = mocker.MagicMock()
            cache_desc_return.shape = data_shape
            return cache_desc_return

        mock_cache_desc = mocker.patch("llm_datadist.CacheDesc",
                                       side_effect=mock_make_cache_desc)
        mock_cache_key = mocker.patch("llm_datadist.CacheKey",
                                      return_value=None)
        # should call mock_kv_transfer.allocate_cache 4 times
        # should call mock_kv_transfer.pull_cache 4 times
        mock_kv_transfer = mocker.MagicMock()
        mock_buffer = mocker.MagicMock()
        mock_cluster = mocker.MagicMock()
        mocker.patch(
            "vllm_ascend.distributed.llmdatadist_connector.KVTransferEngine",
            return_value=mock_kv_transfer)
        # should have mock_buffer.per_device_tensor_addrs attribute
        mock_kv_transfer.allocate_cache.return_value = mock_buffer
        mock_kv_transfer.make_cluster.return_value = mock_cluster

        # should call 4 times
        def create_npu_tensors(data_shape, kv_hidden_dtype, key_buffer_addr):
            new_kvcache = [torch.ones(list(data_shape))] * num_layers
            return new_kvcache

        mock_create_npu_tensors = mocker.patch(
            "torchair.llm_datadist.create_npu_tensors",
            side_effect=create_npu_tensors)

        # should call mock_cache_key_by_id_and_index 4 times
        mock_cache_key = mocker.MagicMock()
        mock_cache_key_by_id_and_index = mocker.patch(
            "llm_datadist.CacheKeyByIdAndIndex",
            side_effect=mock_cache_key,
        )

        # construct test function input
        mock_model = mocker.MagicMock()
        mock_model_input = mocker.MagicMock()
        mock_model_config = mocker.MagicMock()

        mock_model_input.input_tokens = torch.arange(total_token).to(dtype)
        mock_model_input.attn_metadata.seq_lens = seq_lens
        mock_model_input.attn_metadata.slot_mapping = torch.arange(9, -1, -1)

        mock_model.model.start_layer = start_layer
        mock_model.model.end_layer = end_layer
        mock_model.model.config = mock_model_config

        mock_model_config.num_key_value_heads = num_key_value_heads
        mock_model_config.hidden_size = hidden_size
        mock_model_config.num_attention_heads = num_attention_heads

        kv_caches = [
            torch.arange(2 * tokens_num * kv_cache_num_head * head_size).view(
                2, -1, kv_cache_num_head, head_size).to(dtype)
        ] * num_layers

        # should call mock_indices.npu 1 time
        mock_indices = mocker.MagicMock()
        mocker.patch("torch.tensor", return_value=mock_indices)

        # should call len(seq_lens) * (end_layer - start_layer)
        mock_reshape_and_cache = mocker.patch(
            "torch_npu._npu_reshape_and_cache")

        # mock torch.distributed
        mock_dis = mocker.patch("torch.distributed.get_rank", return_value=1)

        hidden_or_intermediate_states, bypass_model_exec, model_input = (
            test_instance.recv_kv_caches_and_hidden_states(
                mock_model,
                mock_model_input,
                kv_caches,
            ))

        assert mock_cache_desc.call_count == 4
        assert mock_cache_key.call_count == 4
        assert mock_buffer.per_device_tensor_addrs
        assert mock_create_npu_tensors.call_count == 4
        assert mock_cache_key_by_id_and_index.call_count == 4

        reshape_and_cache_call_times = len(seq_lens) * (end_layer -
                                                        start_layer)
        assert mock_reshape_and_cache.call_count == reshape_and_cache_call_times

        mock_dis.assert_called_once()

        assert hidden_or_intermediate_states.shape[0] == total_token
        assert hidden_or_intermediate_states.shape[1] == hidden_size
        assert bypass_model_exec == True

    def test_close(self, mocker: MockerFixture):
        # set config
        kv_role = "kv_consumer"
        tensor_parallel_size = 2
        world_size = 16
        num_layers = 64

        rank = 8
        local_rank = 0

        test_instance = self.mock_and_generate_LLMDataDistConnector(
            kv_role, tensor_parallel_size, world_size, num_layers, rank,
            local_rank, mocker)

        test_instance.close()

        test_instance.llm_datadist_engine.data_dist.unlink_clusters.assert_called_once(
        )
