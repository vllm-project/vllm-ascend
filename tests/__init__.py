        runner.ascend_config.is_sparse_li_c4_layer.return_value = False

        runner.ascend_config.is_sparse_li_c4_layer.return_value = False

        runner.ascend_config.is_sparse_li_c4_layer.return_value = False


                runner.ascend_config.is_sparse_li_c4_layer.return_value = False

    @patch(
        "vllm_ascend.worker.model_runner_v1.get_current_hardware_profile",
        return_value=get_hardware_profile(AscendDeviceType.A5),
    )
    @patch("vllm_ascend.worker.model_runner_v1.has_ec_transfer", return_value=False)
    @patch("vllm_ascend.worker.model_runner_v1.get_layers_from_vllm_config")
    def test_a5_sparse_li_c4_specs_keep_main_and_indexer_layouts_separate(
        self,
        mock_get_layers,
        _mock_has_ec_transfer,
        _mock_get_device_type,
    ):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner.kv_cache_dtype = torch.bfloat16
        runner.c8_k_cache_dtype = torch.float8_e4m3fn
        runner.c8_cache_dtype = torch.float8_e4m3fn
        runner.c8_k_scale_cache_dtype = torch.float32
        runner.shared_kv_cache_layers = {}
        runner.ascend_config = MagicMock()
        runner.ascend_config.kvpp_config.size = 1
        runner.ascend_config.is_sparse_li_c8_layer.return_value = False
        runner.ascend_config.is_sparse_li_c4_layer.return_value = True
        runner.model_config.hf_text_config = SimpleNamespace(
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            index_head_dim=128,
        )
        runner.vllm_config.cache_config.cache_dtype = "auto"
        runner.sparse_kv_offload_enabled = False

        attn_module = MLAAttention.__new__(MLAAttention)
        torch.nn.Module.__init__(attn_module)
        attn_module.kv_lora_rank = 512
        attn_module.qk_rope_head_dim = 64
        indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
        torch.nn.Module.__init__(indexer_module)
        attn_layer_name = "model.layers.1.self_attn.attn"
        indexer_layer_name = "model.layers.1.self_attn.indexer.k_cache"
        mock_get_layers.return_value = {
            attn_layer_name: attn_module,
            indexer_layer_name: indexer_module,
        }

        attn_module.impl = SimpleNamespace(
            has_indexer=True,
            enable_sparse_sfa_c8=False,
            enable_sparse_li_c8=False,
        )

        specs = runner.get_kv_cache_spec()
        main_spec = specs[attn_layer_name]
        indexer_spec = specs[indexer_layer_name]

        self.assertEqual(
            runner.ascend_config.is_sparse_li_c4_layer.call_args_list,
            [call(indexer_layer_name)],
        )
        self.assertEqual(main_spec.head_size, 512 + 64)
        self.assertEqual(main_spec.dtype, torch.bfloat16)
        # li_c4: head_size = index_head_dim // 2, dtype = uint8
        self.assertEqual(indexer_spec.head_size, 128 // 2)
        self.assertEqual(indexer_spec.dtype, torch.uint8)
        # li_c4: scale_dim = index_head_dim // 64 * 2
        self.assertEqual(indexer_spec.scale_dim, 128 // 64 * 2)
        # li_c4: scale_dtype = float8_e8m0fnu
        self.assertEqual(indexer_spec.scale_dtype, torch.float8_e8m0fnu)
        self.assertTrue(indexer_spec.cache_sparse_li_c4)
        self.assertFalse(indexer_spec.cache_sparse_li_c8)

    def test_sparse_li_c4_allocate_and_reshape_scale_views(self):
        runner = self._build_runner()
        runner.use_sparse = True
        runner.block_size = 16
        runner._get_attention_kv_cache_dims = lambda _layer_name, _spec: (512, 64)
        runner.sparse_kv_offload_enabled = False

        attn_layer_name = "model.layers.1.self_attn.attn"
        indexer_layer_name = "model.layers.1.self_attn.indexer.k_cache"
        index_head_dim = 128

        main_spec = AscendMLAAttentionSpec(
            block_size=runner.block_size,
            num_kv_heads=1,
            head_size=512 + 64,
            dtype=torch.bfloat16,
            cache_sparse_sfa_c8=False,
        )
        # li_c4 indexer spec: head_size = index_head_dim // 2, dtype = uint8,
        # scale_dim = index_head_dim // 64 * 2, scale_dtype = float8_e8m0fnu
        indexer_spec = AscendSFAIndexerCacheSpec(
            block_size=runner.block_size,
            num_kv_heads=1,
            head_size=index_head_dim // 2,
            dtype=torch.uint8,
            scale_dim=index_head_dim // 64 * 2,
            scale_dtype=torch.float8_e8m0fnu,
            cache_sparse_li_c4=True,
            cache_sparse_li_c8=False,
            sfa_dcp_replicated_indexer_size=1,
        )
        group_spec = UniformTypeKVCacheSpecs.from_specs(
            {
                attn_layer_name: main_spec,
                indexer_layer_name: indexer_spec,
            }
        )
        self.assertIsNotNone(group_spec)
        assert group_spec is not None

        kv_cache_config = KVCacheConfig(
            num_blocks=2,
            kv_cache_tensors=[
                _make_kv_cache_tensor(
                    per_layer_size=main_spec.page_size_bytes * 2,
                    layer_names=[attn_layer_name],
                    page_size=main_spec.page_size_bytes,
                ),
                _make_kv_cache_tensor(
                    per_layer_size=indexer_spec.page_size_bytes * 2,
                    layer_names=[indexer_layer_name],
                    page_size=indexer_spec.page_size_bytes,
                ),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=[attn_layer_name, indexer_layer_name],
                    kv_cache_spec=group_spec,
                )
            ],
        )
        backend = MagicMock()
        backend.get_kv_cache_shape.side_effect = lambda num_blocks, block_size, num_kv_heads, head_size: (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
        )
        runner._kv_cache_spec_attn_group_iterator = MagicMock(
            return_value=[
                SimpleNamespace(
                    kv_cache_group_id=0,
                    kv_cache_spec=main_spec,
                    backend=backend,
                    layer_names=[attn_layer_name],
                ),
                SimpleNamespace(
                    kv_cache_group_id=0,
                    kv_cache_spec=indexer_spec,
                    backend=backend,
                    layer_names=[indexer_layer_name],
                ),
            ]
        )

        raw_caches = runner._allocate_kv_cache_tensors(kv_cache_config)
        caches = runner._reshape_kv_cache_tensors(kv_cache_config, raw_caches, [runner.block_size])

        main_cache = caches[attn_layer_name]
        self.assertEqual(len(main_cache), 2)
        self.assertEqual(main_cache[0].shape, (2, 16, 1, 512))
        self.assertEqual(main_cache[1].shape, (2, 16, 1, 64))

        indexer_cache = caches[indexer_layer_name]
        self.assertEqual(len(indexer_cache), 2)
        self.assertEqual(indexer_cache[0].shape, (2, 16, 1, index_head_dim // 2))
        self.assertEqual(indexer_cache[0].dtype, torch.uint8)
        self.assertEqual(indexer_cache[1].shape, (2, 16, 1, 2, 2))
        self.assertEqual(indexer_cache[1].dtype, torch.float8_e8m0fnu)


@pytest.mark.parametrize(
    ("replicated_indexer", "expected_size"),
    [(False, 1), (True, 4)],
)
@pytest.mark.parametrize("li_c8", [False, True])
@pytest.mark.parametrize("li_c4", [False, True])
@pytest.mark.parametrize("owner", ["unpaired", "static_shared", "mtp", "regular"])
def test_sfa_indexer_cache_spec_runtime_ownership_and_dcp_replication(
    monkeypatch, replicated_indexer, expected_size, li_c8, li_c4, owner
):
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    indexer_module = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
    torch.nn.Module.__init__(indexer_module)
    monkeypatch.setattr(
        indexer_module,
        "get_kv_cache_spec",
        lambda _config: object(),
    )
    layers = {layer_name: indexer_module}
    if owner != "unpaired":
        impl = AscendSFAImpl.__new__(AscendSFAImpl)
        impl.has_indexer = True
        impl._is_mtp_layer = owner == "mtp"
        impl.skip_topk = owner != "regular"
        layers[layer_name.replace(".indexer.k_cache", ".attn")] = SimpleNamespace(
            impl=impl, get_kv_cache_spec=lambda _config: None
        )

    vllm_config = SimpleNamespace(
        additional_config={},
        parallel_config=SimpleNamespace(decode_context_parallel_size=4),
        cache_config=SimpleNamespace(block_size=128, cache_dtype="auto"),
        attention_config=SimpleNamespace(indexer_kv_dtype="int8", hisparse_config=None),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(index_head_dim=128),
        ),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: layers,
    )
    monkeypatch.setattr(
        attn_utils,
        "enable_sfa_dcp_replicated_indexer",
        lambda _config: replicated_indexer,
    )
    monkeypatch.setattr(
        attn_utils,
        "get_current_hardware_profile",
        lambda: get_hardware_profile(AscendDeviceType.A2),
    )
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(
            is_sparse_li_c8_layer=lambda _layer_name: li_c8,
            is_sparse_li_c4_layer=lambda _layer_name: li_c4,
        ),
    )

    specs = attn_utils.get_kv_cache_spec(vllm_config)
    if owner == "static_shared":
        assert layer_name not in specs
        return
    spec = specs[layer_name]

    assert isinstance(spec, AscendSFAIndexerCacheSpec)
    assert spec.sfa_dcp_replicated_indexer_size == expected_size
    if li_c4:
        assert spec.head_size == 128 // 2
        assert spec.dtype == torch.uint8
        assert spec.scale_dim == 128 // 64 * 2
        assert spec.scale_dtype == torch.float8_e8m0fnu
        assert spec.cache_sparse_li_c4 is True
    elif li_c8:
        assert spec.dtype == torch.int8
        assert spec.scale_dim == 1
    else:
        assert spec.dtype == torch.bfloat16
        assert spec.scale_dim == 0


def test_sfa_indexer_li_c4_allocates_and_reshapes_scale_views(monkeypatch):
    """li_c4 indexer: scale cache shape is overridden to
    (*shape[:-1], head_size * 2 // 64, 2)."""
    layer_name = "model.layers.0.self_attn.indexer.k_cache"
    index_head_dim = 128
    spec = AscendSFAIndexerCacheSpec(
        block_size=2,
        num_kv_heads=1,
        head_size=index_head_dim // 2,
        dtype=torch.uint8,
        scale_dim=index_head_dim // 64 * 2,
        scale_dtype=torch.float8_e8m0fnu,
        cache_sparse_li_c4=True,
        cache_sparse_li_c8=False,
        sfa_dcp_replicated_indexer_size=1,
    )
    num_blocks = 2
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[
            _make_kv_cache_tensor(
                num_blocks * spec.page_size_bytes,
                [layer_name],
                spec.page_size_bytes,
            )
        ],
        kv_cache_groups=[KVCacheGroupSpec(layer_names=[layer_name], kv_cache_spec=spec)],
    )
    vllm_config = SimpleNamespace(
        additional_config={},
        kv_transfer_config=None,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1),
    )
    monkeypatch.setattr(attn_utils, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(attn_utils, "enable_sfa", lambda _cfg: False)

    raw = attn_utils._allocate_kv_cache(kv_cache_config, shared_layers={}, device=torch.device("cpu"))
    raw_k, raw_scale = raw[layer_name]
    assert raw_k.dtype == torch.int8
    assert raw_scale.dtype == torch.int8

    backend = SimpleNamespace(
        get_kv_cache_shape=lambda num_blocks_, block_size, num_kv_heads, head_size: (
            num_blocks_,
            block_size,
            num_kv_heads,
            head_size,
        )
    )
    caches = attn_utils._reshape_kv_cache_v2(
        attn_groups=[
            SimpleNamespace(
                kv_cache_group_id=0,
                kv_cache_spec=spec,
                layer_names=[layer_name],
                backend=backend,
            ),
        ],
        kv_cache_raw_tensors=raw,
        cache_dtype="auto",
        kernel_block_sizes=[spec.block_size],
        shared_kv_cache_layers={},
        kv_cache_config=kv_cache_config,
    )
    indexer_k, indexer_scale = caches[layer_name]
    # indexer k: (num_blocks, block_size, 1, head_size)
    assert indexer_k.shape == (num_blocks, spec.block_size, 1, index_head_dim // 2)
    assert indexer_k.dtype == torch.uint8
    assert indexer_scale.shape == (num_blocks, spec.block_size, 1, 2, 2)
    assert indexer_scale.dtype == torch.float8_e8m0fnu




