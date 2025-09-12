is_fast_path_available = False


def apply_mask_to_padding_states(hidden_states, attention_mask):
    """
    Tunes out the hidden states for padding tokens, see https://github.com/state-spaces/mamba/issues/66
    """
    if attion_mask is not None and attention_mastention_mask.shape[0] > 1:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)

    return hidden_states


class RMSNormGated(nn.Module):

    def __init__(
        self,
        hidden_size,
        eps: float = 1e-5,
        group_size: Optional[int] = None,
        norm_before_gate: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """If group_size is not None, we do GroupNorm with each group having group_size elements.
        group_size=None is equivalent to group_size=hidden_size (i.e. there's only 1 group).
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.empty(hidden_size, **factory_kwargs))
        self.register_parameter("bias", None)
        self.group_size = group_size
        self.norm_before_gate = norm_before_gate
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)

    def forward(self, x, z=None):
        """If z is not None, we do norm(x) * silu(z) if norm_before_gate, else norm(x * silu(z))"""
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        # Norm before gate
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(z.to(torch.float32))
 
        return hidden_states.to(input_dtype)


def torch_causal_conv1d_update(
    hidden_states,
    conv_state,
    weight,
    bias=None,
    activation=None,
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]

    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = F.silu(out[:, :, -seq_len:])
    out = out.to(hidden_states.dtype)
    return out


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
 
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - num_heads % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    tot_heads = num_heads + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
    
    print(f"value.shape: {value.shape}, key.shape: {key.shape}")
    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    # reshape to chunks
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)
 
    # chunk decay
    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)
 
    # for each chunk
    for i in range(0, tot_heads // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )
 
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :num_heads]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def torch_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
 
    g = g.unsqueeze(0)
    beta = beta.unsqueeze(0)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
 
    batch_size, sequence_length, num_heads, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale
 
    core_attn_out = torch.zeros(batch_size, sequence_length, num_heads, v_head_dim).to(value)
    last_recurrent_state = (
        torch.zeros(batch_size, sequence_length, k_head_dim, v_head_dim).to(value)
        if initial_state is None
        else initial_state.to(value)
    )
 
    for i in range(num_heads):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)
 
        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        # TODO: check shape
        # core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=1)
 
    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


class Qwen3NextGatedDeltaNet(nn.Module, MambaBase):

    @property
    def mamba_type(self) -> str:
        return "linear_attention"

    def get_attn_backend(self) -> type["AttentionBackend"]:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionBackend
        return GDNAttentionBackend

    def get_state_dtype(self) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            self.model_config.dtype, self.cache_config.mamba_cache_dtype)

    def get_state_shape(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            self.tp_size,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            self.conv_kernel_size,
            self.num_spec,
            use_v1=True)

    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: Optional[ModelConfig] = None,
        cache_config: Optional[CacheConfig] = None,
        quant_config: Optional[QuantizationConfig] = None,
        speculative_config: Optional[SpeculativeConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = extract_layer_index(prefix)
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.prefix = prefix

        self.config = config
        self.model_config = model_config
        self.cache_config = cache_config
        self.quant_config = quant_config
        self.speculative_config = speculative_config
        self.num_spec = (self.speculative_config.num_speculative_tokens
                         if self.speculative_config else 0)

        # QKV
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            prefix=f"{prefix}.conv1d",
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        self.projection_size_ba = self.num_v_heads * 2
        self.in_proj = MergedColumnParallelLinear(
            input_size=self.hidden_size,
            output_sizes=[self.projection_size_qkvz, self.projection_size_ba],
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.in_proj",
        )

        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        delattr(self.conv1d.weight, "weight_loader")
        set_weight_attrs(
            self.conv1d.weight, {
                "weight_loader":
                mamba_v2_sharded_weight_loader([
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ], self.tp_size, self.tp_rank)
            })

        # selective projection used to make dt, B and C input dependant

        # time step projection (discretization)
        # instantiate once and copy inv_dt in init_weights of PretrainedModel
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.tp_size), )
        self.A_log = nn.Parameter(
            torch.empty(
                divide(self.num_v_heads, self.tp_size),
                dtype=torch.float32,
            ))

        set_weight_attrs(self.A_log,
                         {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias,
                         {"weight_loader": sharded_weight_loader(0)})

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            # group_size=None,
            # norm_before_gate=True,
            # device=torch.npu.current_device(),
            # dtype=config.torch_dtype,
        )

        self.out_proj = RowParallelLinear(self.value_dim,
                                          self.hidden_size,
                                          bias=False,
                                          input_is_parallel=True,
                                          quant_config=quant_config,
                                          prefix=f"{prefix}.out_proj")

        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz,
        mixed_ba,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        new_tensor_shape_qkvz = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            (self.head_k_dim + self.head_k_dim +
             (self.head_v_dim + self.head_v_dim) * self.num_v_heads //
             self.num_k_heads),
        )
        new_tensor_shape_ba = mixed_qkvz.size()[:-1] + (
            self.num_k_heads // self.tp_size,
            2 * self.num_v_heads // self.num_k_heads,
        )

        mixed_qkvz = mixed_qkvz.view(*new_tensor_shape_qkvz)
        mixed_ba = mixed_ba.view(*new_tensor_shape_ba)

        split_arg_list_qkvz = [
            self.head_k_dim,
            self.head_k_dim,
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
            (self.num_v_heads // self.num_k_heads * self.head_v_dim),
        ]
        split_arg_list_ba = [
            self.num_v_heads // self.num_k_heads,
            self.num_v_heads // self.num_k_heads
        ]

        # [b, sq, ng, (hn + hn + np/ng * hn + np/ng + np/ng)]
        # --> [b, sq, ng, hn], [b, sq, ng, hn], [b, sq, ng, np/ng * hn],
        #  [b, sq, ng, np/ng * hn], [b, sq, ng, np/ng], [b, sq, ng, np/ng]
        (query, key, value, z) = torch.split(mixed_qkvz,
                                             split_arg_list_qkvz,
                                             dim=2)
        (b, a) = torch.split(mixed_ba, split_arg_list_ba, dim=2)

        # [b, sq, ng, np/ng * hn] -> [b, sq, np, hn]
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b = b.reshape(b.size(0), self.num_v_heads // self.tp_size)
        a = a.reshape(a.size(0), self.num_v_heads // self.tp_size)

        return query, key, value, z, b, a

    def rearrange_mixed_qkv(self, mixed_qkv):
        if mixed_qkv is None:
            return None, None, None
        query, key, value = torch.split(
            mixed_qkv,
            [
                self.key_dim // self.tp_size,
                self.key_dim // self.tp_size,
                self.value_dim // self.tp_size,
            ],
            dim=-1,
        )
        query, key = map(
            lambda x: rearrange(x, 'l (h d) -> 1 l h d', d=self.head_k_dim),
            (query, key))
        value = rearrange(value, 'l (h d) -> 1 l h d', d=self.head_v_dim)
        return query, key, value

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
        cache_params: Optional[MambaCacheParams] = None,
    ):
        return torch.ops.vllm.gdn_attention(
            hidden_states,
            output,
            self.prefix,
        )

    def _forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        forward_context = get_forward_context()
        attn_metadata: AttentionMetadata = forward_context.attn_metadata

        if attn_metadata is None:
            # V1 profile run
            return

        attn_metadata = attn_metadata.gdn_attn_metadata
        # assert isinstance(attn_metadata, dict)
        # attn_metadata = attn_metadata[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        has_initial_state = attn_metadata.has_initial_state
        spec_query_start_loc = attn_metadata.spec_query_start_loc
        non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
        spec_sequence_masks = attn_metadata.spec_sequence_masks
        spec_token_masks = attn_metadata.spec_token_masks
        spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor  # noqa: E501
        non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor  # noqa: E501
        self_kv_cache = self.kv_cache[forward_context.virtual_engine]

        conv_state = self_kv_cache[0].transpose(-1, -2)
        ssm_state = self_kv_cache[1]

        print(f"conv_state.shape: {conv_state.shape}, ssm_state.shape: {ssm_state.shape}, hidden_states.shape: {hidden_states.shape}")

        num_actual_tokens = (attn_metadata.num_prefill_tokens +
                             attn_metadata.num_decode_tokens +
                             attn_metadata.num_spec_decode_tokens)
        num_accepted_tokens = attn_metadata.num_accepted_tokens

        # 1. Set up dimensions for reshapes later
        projected_states, _ = self.in_proj(hidden_states[:num_actual_tokens])
        if spec_token_masks is not None:
            spec_token_masks = spec_token_masks[:num_actual_tokens]
        projected_states_qkvz, projected_states_ba = torch.split(
            projected_states,
            [
                self.projection_size_qkvz // self.tp_size,
                self.projection_size_ba // self.tp_size
            ],
            dim=-1,
        )
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba)
        query, key, value = map(lambda x: rearrange(x, 'l p d -> l (p d)'),
                                (query, key, value))
        mixed_qkv = torch.cat((query, key, value), dim=-1)

        # 2. Convolution sequence transformation
        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0),
                                               self.conv1d.weight.size(2))

        if spec_sequence_masks is not None:
            if (attn_metadata.num_prefills == 0
                    and attn_metadata.num_decodes == 0):
                mixed_qkv_spec = mixed_qkv
                mixed_qkv_non_spec = None
            else:
                mixed_qkv_spec = mixed_qkv[spec_token_masks]
                mixed_qkv_non_spec = mixed_qkv[~spec_token_masks]
        else:
            mixed_qkv_spec = None
            mixed_qkv_non_spec = mixed_qkv

        # 2.1: process the mutli-query part
        if spec_sequence_masks is not None:
            mixed_qkv_spec = mixed_qkv_spec.view(
                attn_metadata.num_spec_decodes, -1, mixed_qkv_spec.size(-1))
            mixed_qkv_spec = rearrange(mixed_qkv_spec, 'b l d -> b d l')
            mixed_qkv_spec = causal_conv1d_update(
                mixed_qkv_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=spec_state_indices_tensor[:, 0]
                [:attn_metadata.num_spec_decodes],
                num_accepted_tokens=num_accepted_tokens,
                validate_data=False,
            )
            mixed_qkv_spec = rearrange(mixed_qkv_spec, 'b d l -> (b l) d')

        # 2.2: process the remaining part
        if attn_metadata.num_prefills > 0:
            # - "cache_indices" updates the conv_state cache in positions
            #   pointed to by "mamba_cache_params.state_indices_tensor"
            mixed_qkv_non_spec = causal_conv1d_fn(
                mixed_qkv_non_spec.transpose(0, 1),
                conv_weights,
                self.conv1d.bias,
                activation=self.activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=non_spec_state_indices_tensor,
                query_start_loc=non_spec_query_start_loc,
            ).transpose(0, 1)
        elif attn_metadata.num_decodes > 0:
            mixed_qkv_non_spec = causal_conv1d_update(
                mixed_qkv_non_spec,
                conv_state,
                conv_weights,
                self.conv1d.bias,
                self.activation,
                conv_state_indices=non_spec_state_indices_tensor[:attn_metadata
                                                                 .num_decodes],
                validate_data=True,
            )
        else:
            mixed_qkv_non_spec = None

        query_spec, key_spec, value_spec = self.rearrange_mixed_qkv(
            mixed_qkv_spec)
        query_non_spec, key_non_spec, value_non_spec = self.rearrange_mixed_qkv(
            mixed_qkv_non_spec)

        beta = b.sigmoid()
        # g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        # g = fused_gdn_gating(self.A_log, a, self.dt_bias)
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        g, beta = map(lambda x: rearrange(x, 'l d -> 1 l d'), (g, beta))

        if spec_sequence_masks is not None:
            if (attn_metadata.num_prefills == 0
                    and attn_metadata.num_decodes == 0):
                g_spec = g
                beta_spec = beta
                g_non_spec = None
                beta_non_spec = None
            else:
                g_spec = g[:, spec_token_masks]
                beta_spec = beta[:, spec_token_masks]
                g_non_spec = g[:, ~spec_token_masks]
                beta_non_spec = beta[:, ~spec_token_masks]
        else:
            g_spec = None
            beta_spec = None
            g_non_spec = g
            beta_non_spec = beta

        # 3. Recurrent attention

        # 3.1: process the mutlti-query part
        if spec_sequence_masks is not None:
            core_attn_out_spec, last_recurrent_state = (
                fused_recurrent_gated_delta_rule(
                    q=query_spec,
                    k=key_spec,
                    v=value_spec,
                    g=g_spec,
                    beta=beta_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=spec_query_start_loc[:attn_metadata.
                                                    num_spec_decodes + 1],
                    ssm_state_indices=spec_state_indices_tensor,
                    num_accepted_tokens=num_accepted_tokens,
                    use_qk_l2norm_in_kernel=True,
                ))
        else:
            core_attn_out_spec, last_recurrent_state = None, None

        # 3.2: process the remaining part
        if attn_metadata.num_prefills > 0:
            initial_state = ssm_state[
                non_spec_state_indices_tensor].contiguous()
            initial_state[~has_initial_state, ...] = 0
            # (
            #     core_attn_out_non_spec,
            #     last_recurrent_state,
            # ) = chunk_gated_delta_rule(
            #     q=query_non_spec,
            #     k=key_non_spec,
            #     v=value_non_spec,
            #     g=g_non_spec,
            #     beta=beta_non_spec,
            #     initial_state=initial_state,
            #     output_final_state=True,
            #     cu_seqlens=non_spec_query_start_loc,
            #     head_first=False,
            #     use_qk_l2norm_in_kernel=True,
            # )
            
            (
                core_attn_out_non_spec,
                last_recurrent_state,
            ) = torch_chunk_gated_delta_rule(
                query=query_non_spec,
                key=key_non_spec,
                value=value_non_spec,
                g=g_non_spec,
                beta=beta_non_spec,
                initial_state=initial_state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )

            # Init cache
            ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(
                ssm_state.dtype)
        elif attn_metadata.num_decodes > 0:
            # core_attn_out_non_spec, last_recurrent_state = (
            #     fused_recurrent_gated_delta_rule(
            #         q=query_non_spec,
            #         k=key_non_spec,
            #         v=value_non_spec,
            #         g=g_non_spec,
            #         beta=beta_non_spec,
            #         initial_state=ssm_state,
            #         inplace_final_state=True,
            #         cu_seqlens=non_spec_query_start_loc[:attn_metadata.
            #                                             num_decodes + 1],
            #         ssm_state_indices=non_spec_state_indices_tensor,
            #         use_qk_l2norm_in_kernel=True,
            #     ))
            
            core_attn_out_non_spec, last_recurrent_state = (
                torch_recurrent_gated_delta_rule(
                    q=query_non_spec,
                    k=key_non_spec,
                    v=value_non_spec,
                    g=g_non_spec,
                    beta=beta_non_spec,
                    initial_state=ssm_state,
                    inplace_final_state=True,
                    cu_seqlens=non_spec_query_start_loc[:attn_metadata.
                                                        num_decodes + 1],
                    ssm_state_indices=non_spec_state_indices_tensor,
                    use_qk_l2norm_in_kernel=True,
                ))
        else:
            core_attn_out_non_spec, last_recurrent_state = None, None

        # Merge core attention output
        if (spec_sequence_masks is not None
                and core_attn_out_non_spec is not None):
            core_attn_out = torch.empty(
                (1, num_actual_tokens, *core_attn_out_spec.shape[2:]),
                dtype=core_attn_out_non_spec.dtype,
                device=core_attn_out_non_spec.device,
            )
            core_attn_out[:, spec_token_masks] = core_attn_out_spec
            core_attn_out[:, ~spec_token_masks] = core_attn_out_non_spec
        elif spec_sequence_masks is not None:
            core_attn_out = core_attn_out_spec
        else:
            core_attn_out = core_attn_out_non_spec

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = rearrange(core_attn_out, '... h d -> ... (h d)')

        output[:num_actual_tokens], _ = self.out_proj(core_attn_out)


