# SPDX-License-Identifier: Apache-2.0
"""A5 operator selection and adapters for Qwen gated delta networks."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Mapping

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class GDNBackendMode(StrEnum):
    AUTO = "auto"
    FLA_NPU = "fla_npu"
    NATIVE = "native"


class GDNOperator(StrEnum):
    CAUSAL_CONV1D = "causal_conv1d"
    L2NORM_FWD = "l2norm_fwd"
    CHUNK_LOCAL_CUMSUM = "chunk_local_cumsum"
    CHUNK_SCALED_DOT_KKT = "chunk_scaled_dot_kkt"
    SOLVE_TRI = "solve_tri"
    RECOMPUTE_W_U_FWD = "recompute_w_u_fwd"
    CHUNK_GATED_DELTA_RULE_FWD_H = "chunk_gated_delta_rule_fwd_h"
    CHUNK_FWD_O = "chunk_fwd_o"
    RECURRENT_GATED_DELTA_RULE = "recurrent_gated_delta_rule"


@dataclass(frozen=True)
class GDNBackendConfig:
    mode: GDNBackendMode
    overrides: dict[GDNOperator, GDNBackendMode]

    def mode_for(self, operator: GDNOperator) -> GDNBackendMode:
        return self.overrides.get(operator, self.mode)


@dataclass(frozen=True)
class GDNRuntimeSignature:
    soc: str
    dtype: str
    state_dtype: str
    num_key_heads: int
    num_value_heads: int
    key_dim: int
    value_dim: int
    chunk_size: int = 64
    mtp: bool = False
    acl_graph: bool = False


@dataclass(frozen=True)
class GDNOperatorSelection:
    backend: GDNBackendMode
    operator: Callable[..., Any]
    symbol: str
    reason: str | None = None


@dataclass(frozen=True)
class GDNPrefillMetadata:
    cu_seqlens: torch.Tensor | None
    cu_seqlens_host: tuple[int, ...] | None
    chunk_indices: torch.Tensor | None
    chunk_indices_host: tuple[int, ...] | None
    cu_seqlens_kern: tuple[int, ...] | None = None
    keep_meta: torch.Tensor | None = None
    block_indices_cumsum: torch.Tensor | None = None
    chunk_indices_large_block: torch.Tensor | None = None


def run_gdn_prefill_pipeline(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    has_initial_state: torch.Tensor,
    scale: float,
    metadata: GDNPrefillMetadata,
    operators: Mapping[GDNOperator, Callable[..., Any]],
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the inference-only forward ordering from flash_gated_delta_rule.py."""

    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("GDN prefill expects q/k/v with shape [B, T, H, D].")
    if q.shape != k.shape:
        raise ValueError(f"GDN prefill q/k shapes must match, got q={tuple(q.shape)} k={tuple(k.shape)}.")
    if q.shape[:2] != v.shape[:2]:
        raise ValueError(f"GDN prefill token layouts must match, got q={tuple(q.shape)} v={tuple(v.shape)}.")

    num_key_heads = q.shape[2]
    num_value_heads = v.shape[2]
    if num_value_heads % num_key_heads != 0:
        raise ValueError(
            "GDN prefill value heads must be an integer multiple of key heads, "
            f"got key_heads={num_key_heads} value_heads={num_value_heads}."
        )
    if g.shape != beta.shape or g.shape != (q.shape[0], q.shape[1], num_value_heads):
        raise ValueError(
            "GDN prefill g/beta must both have shape [B, T, Nv], "
            f"got g={tuple(g.shape)} beta={tuple(beta.shape)}."
        )

    l2norm = operators[GDNOperator.L2NORM_FWD]
    q = l2norm(q)
    k = l2norm(k)
    repeat = num_value_heads // num_key_heads
    if repeat > 1:
        q = q.repeat_interleave(repeat, dim=2)
        k = k.repeat_interleave(repeat, dim=2)

    # fla_npu chunk forward operators use [B, H, T, D].
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()

    initial_state = initial_state.clone()
    initial_state[~has_initial_state, ...] = 0
    initial_state = initial_state.transpose(-1, -2).contiguous()

    g = operators[GDNOperator.CHUNK_LOCAL_CUMSUM](
        g,
        chunk_size=chunk_size,
        cu_seqlens=metadata.cu_seqlens,
        chunk_indices=metadata.chunk_indices,
    )
    a = operators[GDNOperator.CHUNK_SCALED_DOT_KKT](
        k,
        g,
        beta,
        chunk_size=chunk_size,
        cu_seqlens=metadata.cu_seqlens,
        chunk_indices=metadata.chunk_indices,
    )
    a = operators[GDNOperator.SOLVE_TRI](
        a,
        cu_seqlens=metadata.cu_seqlens,
        cu_seqlens_host=metadata.cu_seqlens_host,
        chunk_indices=metadata.chunk_indices,
        chunk_indices_host=metadata.chunk_indices_host,
        output_dtype=k.dtype,
    )

    g_head = g.transpose(1, 2).contiguous()
    beta_head = beta.transpose(1, 2).contiguous().float()
    a_head = a.transpose(1, 2).contiguous()
    w, u = operators[GDNOperator.RECOMPUTE_W_U_FWD](
        k,
        v,
        beta_head,
        a_head,
        g_head,
        chunk_size=chunk_size,
        cu_seqlens=metadata.cu_seqlens_host,
        chunk_indices=metadata.chunk_indices_host,
    )

    initial_state_kern = initial_state
    if metadata.keep_meta is not None:
        initial_state_kern = initial_state[metadata.keep_meta]
    h, v_new, final_state = operators[GDNOperator.CHUNK_GATED_DELTA_RULE_FWD_H](
        k,
        w,
        u,
        g_head,
        initial_state_kern,
        chunk_size=chunk_size,
        cu_seqlens=metadata.cu_seqlens_kern or metadata.cu_seqlens_host,
        chunk_indices=metadata.chunk_indices_host,
    )
    if metadata.keep_meta is not None:
        full_state = initial_state.clone()
        full_state[metadata.keep_meta] = final_state
        final_state = full_state

    output = operators[GDNOperator.CHUNK_FWD_O](
        q,
        k,
        v_new,
        h,
        g_head,
        scale=scale,
        chunk_size=chunk_size,
        cu_seqlens=metadata.cu_seqlens_host,
        chunk_indices=metadata.chunk_indices_host,
    )
    return output.transpose(1, 2).contiguous(), final_state.transpose(-1, -2).contiguous()


def run_gdn_decode_pipeline(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    state: torch.Tensor,
    scale: float,
    actual_seq_lengths: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    l2norm: Callable[[torch.Tensor], torch.Tensor],
    recurrent: Callable[..., Any],
) -> torch.Tensor:
    """Run ordinary (non-MTP) recurrent decode with native cache semantics."""

    q = l2norm(q)
    k = l2norm(k)
    result = recurrent(
        query=q.squeeze(0),
        key=k.squeeze(0),
        value=v.squeeze(0),
        g=g.squeeze(0),
        beta=beta.squeeze(0),
        state=state,
        scale=scale,
        actual_seq_lengths=actual_seq_lengths,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=None,
    )
    if isinstance(result, (tuple, list)):
        output, final_state = result
        state.copy_(final_state)
    else:
        output = result
    return output.unsqueeze(0)


_FLA_OPERATOR_PATHS: dict[GDNOperator, tuple[str, str]] = {
    GDNOperator.CAUSAL_CONV1D: ("fla_npu.ops.ascendc", "causal_conv1d"),
    GDNOperator.L2NORM_FWD: ("fla_npu.ops.triton", "l2norm_fwd"),
    GDNOperator.CHUNK_LOCAL_CUMSUM: ("fla_npu.ops.triton", "chunk_local_cumsum"),
    GDNOperator.CHUNK_SCALED_DOT_KKT: ("fla_npu.ops.triton", "chunk_scaled_dot_kkt_fwd"),
    GDNOperator.SOLVE_TRI: ("fla_npu.ops.ascendc", "solve_tri"),
    GDNOperator.RECOMPUTE_W_U_FWD: ("fla_npu.ops.ascendc", "recompute_w_u_fwd"),
    GDNOperator.CHUNK_GATED_DELTA_RULE_FWD_H: (
        "fla_npu.ops.ascendc",
        "chunk_gated_delta_rule_fwd_h",
    ),
    GDNOperator.CHUNK_FWD_O: ("fla_npu.ops.ascendc", "chunk_fwd_o"),
}


def _first_line(exc: BaseException) -> str:
    return str(exc).splitlines()[0] if str(exc) else type(exc).__name__


def resolve_fla_operator(operator: GDNOperator) -> tuple[Callable[..., Any], str]:
    """Resolve a public fla_npu operator without importing it on native paths."""

    if operator is GDNOperator.RECURRENT_GATED_DELTA_RULE:
        return _resolve_fla_recurrent_operator()
    module_name, attribute = _FLA_OPERATOR_PATHS[operator]
    module = importlib.import_module(module_name)
    resolved = getattr(module, attribute)
    return resolved, f"{module_name}.{attribute}"


def _resolve_fla_recurrent_operator() -> tuple[Callable[..., Any], str]:
    raise AttributeError(
        "flash-linear-attention-npu does not expose recurrent_gated_delta_rule; "
        "use the vllm-ascend native operator"
    )


class A5GDNOperatorDispatcher:
    """Select and cache normalized A5 GDN operator implementations."""

    def __init__(self, config: GDNBackendConfig, *, is_a5: bool) -> None:
        self.config = config
        self.is_a5 = is_a5
        self._selections: dict[tuple[GDNOperator, GDNRuntimeSignature], GDNOperatorSelection] = {}

    def select(
        self,
        operator: GDNOperator,
        signature: GDNRuntimeSignature,
        *,
        native: Callable[..., Any],
        native_symbol: str,
        fla_resolver: Callable[[], tuple[Callable[..., Any], str]] | None = None,
        probe: Callable[[Callable[..., Any]], bool | None] | None = None,
    ) -> GDNOperatorSelection:
        cache_key = (operator, signature)
        if cache_key in self._selections:
            return self._selections[cache_key]

        requested = self.config.mode_for(operator)
        if not self.is_a5 or requested is GDNBackendMode.NATIVE:
            selection = GDNOperatorSelection(GDNBackendMode.NATIVE, native, native_symbol)
            self._remember(operator, signature, selection)
            return selection

        resolver = fla_resolver or (lambda: resolve_fla_operator(operator))
        try:
            resolved, symbol = resolver()
        except Exception as exc:
            return self._fallback_or_raise(
                operator,
                signature,
                requested,
                native,
                native_symbol,
                stage="resolve",
                exc=exc,
            )

        if probe is not None:
            try:
                probe_result = probe(resolved)
                if probe_result is False:
                    raise RuntimeError("smoke probe returned false")
            except Exception as exc:
                return self._fallback_or_raise(
                    operator,
                    signature,
                    requested,
                    native,
                    native_symbol,
                    stage="smoke_probe",
                    exc=exc,
                )

        selection = GDNOperatorSelection(GDNBackendMode.FLA_NPU, resolved, symbol)
        self._remember(operator, signature, selection)
        return selection

    def execute(
        self,
        operator: GDNOperator,
        selection: GDNOperatorSelection,
        *args: Any,
        phase: str,
        layer_name: str,
        state_may_be_mutated: bool,
        **kwargs: Any,
    ) -> Any:
        try:
            return selection.operator(*args, **kwargs)
        except Exception:
            logger.exception(
                "GDN A5 operator execution failed: op=%s backend=%s symbol=%s phase=%s "
                "layer=%s state_may_be_mutated=%s",
                operator.value,
                selection.backend.value,
                selection.symbol,
                phase,
                layer_name,
                state_may_be_mutated,
            )
            raise

    def select_native_only(
        self,
        operator: GDNOperator,
        signature: GDNRuntimeSignature,
        *,
        native: Callable[..., Any],
        native_symbol: str,
    ) -> GDNOperatorSelection:
        """Select an intentionally retained native implementation and log it."""

        cache_key = (operator, signature)
        if cache_key in self._selections:
            return self._selections[cache_key]
        requested_override = self.config.overrides.get(operator)
        if requested_override is GDNBackendMode.FLA_NPU:
            raise RuntimeError(
                f"GDN operator {operator.value} has no Stage 1 fla_npu replacement; "
                "remove its per-operator override"
            )
        selection = GDNOperatorSelection(
            GDNBackendMode.NATIVE,
            native,
            native_symbol,
            "retained native implementation",
        )
        self._remember(operator, signature, selection)
        return selection

    def _fallback_or_raise(
        self,
        operator: GDNOperator,
        signature: GDNRuntimeSignature,
        requested: GDNBackendMode,
        native: Callable[..., Any],
        native_symbol: str,
        *,
        stage: str,
        exc: BaseException,
    ) -> GDNOperatorSelection:
        reason = _first_line(exc)
        if requested is GDNBackendMode.FLA_NPU:
            raise RuntimeError(
                f"GDN operator {operator.value} failed during {stage} in strict fla_npu mode: {reason}"
            ) from exc
        selection = GDNOperatorSelection(GDNBackendMode.NATIVE, native, native_symbol, reason)
        logger.warning(
            "GDN A5 operator fallback: op=%s requested=%s selected=native stage=%s "
            "exception=%s reason=%s",
            operator.value,
            requested.value,
            stage,
            type(exc).__name__,
            reason,
        )
        self._remember(operator, signature, selection)
        return selection

    def _remember(
        self,
        operator: GDNOperator,
        signature: GDNRuntimeSignature,
        selection: GDNOperatorSelection,
    ) -> None:
        self._selections[(operator, signature)] = selection
        logger.info(
            "GDN A5 operator selected: op=%s backend=%s symbol=%s soc=%s dtype=%s "
            "state_dtype=%s nk=%d nv=%d dk=%d dv=%d chunk_size=%d mtp=%s acl_graph=%s",
            operator.value,
            selection.backend.value,
            selection.symbol,
            signature.soc,
            signature.dtype,
            signature.state_dtype,
            signature.num_key_heads,
            signature.num_value_heads,
            signature.key_dim,
            signature.value_dim,
            signature.chunk_size,
            signature.mtp,
            signature.acl_graph,
        )


class A5GDNAdapter:
    """Normalize fla_npu/native operator contracts for one Qwen GDN layer."""

    def __init__(
        self,
        config: GDNBackendConfig,
        signature: GDNRuntimeSignature,
        *,
        layer_name: str,
        is_a5: bool,
    ) -> None:
        self.signature = signature
        self.layer_name = layer_name
        self.dispatcher = A5GDNOperatorDispatcher(config, is_a5=is_a5)

    def prefill(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        has_initial_state: torch.Tensor,
        scale: float,
        metadata: GDNPrefillMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        operators = self._prefill_operators(metadata)
        return run_gdn_prefill_pipeline(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            has_initial_state=has_initial_state,
            scale=scale,
            metadata=metadata,
            operators=operators,
            chunk_size=self.signature.chunk_size,
        )

    def causal_conv1d(
        self,
        *,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        conv_state: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        initial_state_mode: torch.Tensor | None,
        activation_mode: int,
        pad_slot_id: int,
        run_mode: int,
    ) -> torch.Tensor:
        def native(
            input_tensor,
            conv_weight,
            conv_bias,
            conv_states,
            **kwargs,
        ):
            output = torch.empty_like(input_tensor)
            torch.ops._C_ascend.npu_causal_conv1d_custom(
                output,
                input_tensor,
                conv_weight,
                conv_state=conv_states,
                bias_opt=conv_bias,
                query_start_loc_opt=kwargs["query_start_loc"],
                cache_indices_opt=kwargs["cache_indices"],
                initial_state_mode_opt=kwargs["initial_state_mode"],
                num_accepted_tokens_opt=kwargs["num_accepted_tokens"],
                activation_mode=kwargs["activation_mode"],
                pad_slot_id=kwargs["pad_slot_id"],
                run_mode=kwargs["run_mode"],
            )
            return output

        selection = self.dispatcher.select(
            GDNOperator.CAUSAL_CONV1D,
            self.signature,
            native=native,
            native_symbol="torch.ops._C_ascend.npu_causal_conv1d_custom",
        )
        operator = self._logged_operator(
            GDNOperator.CAUSAL_CONV1D,
            selection,
            phase="prefill" if run_mode == 0 else "decode",
            stateful=True,
        )
        return operator(
            x,
            weight,
            bias,
            conv_state,
            query_start_loc=query_start_loc,
            cache_indices=cache_indices,
            initial_state_mode=initial_state_mode,
            num_accepted_tokens=None,
            activation_mode=activation_mode,
            pad_slot_id=pad_slot_id,
            run_mode=run_mode,
            head_num=0,
        )

    def decode(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        state: torch.Tensor,
        scale: float,
        actual_seq_lengths: torch.Tensor,
        ssm_state_indices: torch.Tensor,
    ) -> torch.Tensor:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd

        l2_selection = self.dispatcher.select_native_only(
            GDNOperator.L2NORM_FWD,
            self.signature,
            native=l2norm_fwd,
            native_symbol="vllm.third_party.fla.l2norm_fwd",
        )
        recurrent_selection = self.dispatcher.select_native_only(
            GDNOperator.RECURRENT_GATED_DELTA_RULE,
            self.signature,
            native=torch.ops._C_ascend.npu_recurrent_gated_delta_rule,
            native_symbol="torch.ops._C_ascend.npu_recurrent_gated_delta_rule",
        )
        return run_gdn_decode_pipeline(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            state=state,
            scale=scale,
            actual_seq_lengths=actual_seq_lengths,
            ssm_state_indices=ssm_state_indices,
            l2norm=self._logged_operator(
                GDNOperator.L2NORM_FWD,
                l2_selection,
                phase="decode",
                stateful=False,
            ),
            recurrent=self._logged_operator(
                GDNOperator.RECURRENT_GATED_DELTA_RULE,
                recurrent_selection,
                phase="decode",
                stateful=True,
            ),
        )

    def _prefill_operators(self, metadata: GDNPrefillMetadata) -> dict[GDNOperator, Callable[..., Any]]:
        from vllm.third_party.flash_linear_attention.ops.l2norm import l2norm_fwd as native_l2norm

        from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import (
            chunk_scaled_dot_kkt_fwd as native_kkt,
        )
        from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum as native_cumsum
        from vllm_ascend.ops.triton.fla.solve_tril import solve_tril as native_solve
        from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd as native_recompute

        def native_cumsum_normalized(gate, *, chunk_size, cu_seqlens, chunk_indices):
            del chunk_indices
            return native_cumsum(
                gate,
                chunk_size=chunk_size,
                cu_seqlens=cu_seqlens,
                block_indices=metadata.block_indices_cumsum,
                head_first=False,
            )

        def fla_cumsum(raw):
            def call(gate, *, chunk_size, cu_seqlens, chunk_indices):
                indices = None if chunk_indices is None else {str(chunk_size): chunk_indices}
                return raw(
                    gate,
                    chunk_size=chunk_size,
                    cu_seqlens=cu_seqlens,
                    chunk_indices_out=indices,
                    head_first=False,
                )

            return call

        def native_kkt_normalized(key, gate, beta_value, *, chunk_size, cu_seqlens, chunk_indices):
            return native_kkt(
                k=key.transpose(1, 2).contiguous(),
                beta=beta_value,
                g_cumsum=gate,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                output_dtype=torch.float32,
            )

        def fla_kkt(raw):
            def call(key, gate, beta_value, *, chunk_size, cu_seqlens, chunk_indices):
                return raw(
                    k=key,
                    g=gate,
                    beta=beta_value,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    chunk_size=chunk_size,
                    output_dtype=torch.float32,
                )

            return call

        def native_solve_normalized(
            a,
            *,
            cu_seqlens,
            cu_seqlens_host,
            chunk_indices,
            chunk_indices_host,
            output_dtype,
        ):
            del cu_seqlens_host, chunk_indices_host
            return native_solve(
                A=a,
                cu_seqlens=cu_seqlens,
                chunk_indices_large_block=metadata.chunk_indices_large_block,
                chunk_indices_bt=chunk_indices,
                output_dtype=output_dtype,
            )

        def fla_solve(raw):
            def call(
                a,
                *,
                cu_seqlens,
                cu_seqlens_host,
                chunk_indices,
                chunk_indices_host,
                output_dtype,
            ):
                del cu_seqlens, chunk_indices
                a = a.to(output_dtype).contiguous()
                if cu_seqlens_host is None:
                    return raw(a, layout="bsnd")
                return raw(
                    a.squeeze(0),
                    cu_seqlens=cu_seqlens_host,
                    chunk_indices=chunk_indices_host,
                    layout="tnd",
                ).unsqueeze(0)

            return call

        def native_recompute_normalized(
            key,
            value,
            beta_value,
            a,
            gate,
            *,
            chunk_size,
            cu_seqlens,
            chunk_indices,
        ):
            del chunk_size, cu_seqlens, chunk_indices
            w, u = native_recompute(
                k=key.transpose(1, 2).contiguous(),
                v=value.transpose(1, 2).contiguous(),
                beta=beta_value.transpose(1, 2).contiguous(),
                g_cumsum=gate.transpose(1, 2).contiguous(),
                A=a.transpose(1, 2).contiguous(),
                cu_seqlens=metadata.cu_seqlens,
                chunk_indices=metadata.chunk_indices,
            )
            return w.transpose(1, 2).contiguous(), u.transpose(1, 2).contiguous()

        def fla_recompute(raw):
            def call(
                key,
                value,
                beta_value,
                a,
                gate,
                *,
                chunk_size,
                cu_seqlens,
                chunk_indices,
            ):
                return raw(
                    key,
                    value,
                    beta_value,
                    a,
                    chunk_size,
                    g=gate,
                    gk=None,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                )

            return call

        def native_fwd_h(key, w, u, gate, initial_state, *, chunk_size, cu_seqlens, chunk_indices):
            return torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h(
                key,
                w,
                u,
                g=gate,
                gk=None,
                initial_state=initial_state,
                output_final_state=True,
                chunk_size=chunk_size,
                save_new_value=True,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                use_exp2=False,
                transpose_state_layout=False,
            )

        def fla_fwd_h(raw):
            def call(key, w, u, gate, initial_state, *, chunk_size, cu_seqlens, chunk_indices):
                return raw(
                    key,
                    w,
                    u,
                    g=gate,
                    gk=None,
                    initial_state=initial_state,
                    output_final_state=True,
                    chunk_size=chunk_size,
                    save_new_value=True,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    use_exp2=False,
                    transpose_state_layout=False,
                )

            return call

        def native_fwd_o(query, key, value, h, gate, *, scale, chunk_size, cu_seqlens, chunk_indices):
            return torch.ops._C_ascend.chunk_fwd_o(
                query,
                key,
                value,
                h,
                scale,
                g=gate,
                g_gamma=None,
                cu_seqlens=cu_seqlens,
                chunk_indices=chunk_indices,
                chunk_size=chunk_size,
                transpose_state_layout=False,
            )

        def fla_fwd_o(raw):
            def call(query, key, value, h, gate, *, scale, chunk_size, cu_seqlens, chunk_indices):
                return raw(
                    query,
                    key,
                    value,
                    h,
                    scale,
                    g=gate,
                    g_gamma=None,
                    cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices,
                    chunk_size=chunk_size,
                    transpose_state_layout=False,
                )

            return call

        specs = {
            GDNOperator.L2NORM_FWD: (native_l2norm, "vllm.third_party.fla.l2norm_fwd", None),
            GDNOperator.CHUNK_LOCAL_CUMSUM: (
                native_cumsum_normalized,
                "vllm_ascend.triton.chunk_local_cumsum",
                fla_cumsum,
            ),
            GDNOperator.CHUNK_SCALED_DOT_KKT: (
                native_kkt_normalized,
                "vllm_ascend.triton.chunk_scaled_dot_kkt_fwd",
                fla_kkt,
            ),
            GDNOperator.SOLVE_TRI: (
                native_solve_normalized,
                "vllm_ascend.triton.solve_tril",
                fla_solve,
            ),
            GDNOperator.RECOMPUTE_W_U_FWD: (
                native_recompute_normalized,
                "vllm_ascend.triton.recompute_w_u_fwd",
                fla_recompute,
            ),
            GDNOperator.CHUNK_GATED_DELTA_RULE_FWD_H: (
                native_fwd_h,
                "torch.ops._C_ascend.chunk_gated_delta_rule_fwd_h",
                fla_fwd_h,
            ),
            GDNOperator.CHUNK_FWD_O: (
                native_fwd_o,
                "torch.ops._C_ascend.chunk_fwd_o",
                fla_fwd_o,
            ),
        }
        selected: dict[GDNOperator, Callable[..., Any]] = {}
        for operator, (native, native_symbol, normalize_fla) in specs.items():
            if normalize_fla is None:
                selection = self.dispatcher.select_native_only(
                    operator,
                    self.signature,
                    native=native,
                    native_symbol=native_symbol,
                )
            else:
                selection = self.dispatcher.select(
                    operator,
                    self.signature,
                    native=native,
                    native_symbol=native_symbol,
                    fla_resolver=self._normalized_fla_resolver(operator, normalize_fla),
                )
            selected[operator] = self._logged_operator(operator, selection, phase="prefill", stateful=False)
        return selected

    @staticmethod
    def _normalized_fla_resolver(
        operator: GDNOperator,
        normalizer: Callable[[Callable[..., Any]], Callable[..., Any]],
    ) -> Callable[[], tuple[Callable[..., Any], str]]:
        def resolve() -> tuple[Callable[..., Any], str]:
            raw, symbol = resolve_fla_operator(operator)
            return normalizer(raw), symbol

        return resolve

    def _logged_operator(
        self,
        operator: GDNOperator,
        selection: GDNOperatorSelection,
        *,
        phase: str,
        stateful: bool,
    ) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            return self.dispatcher.execute(
                operator,
                selection,
                *args,
                phase=phase,
                layer_name=self.layer_name,
                state_may_be_mutated=stateful,
                **kwargs,
            )

        return call


def _parse_mode(value: str) -> GDNBackendMode:
    try:
        return GDNBackendMode(value.strip().lower())
    except ValueError as exc:
        valid = ", ".join(mode.value for mode in GDNBackendMode)
        raise ValueError(f"Invalid GDN backend mode {value!r}; expected one of: {valid}.") from exc


def parse_gdn_backend_config(mode: str, operator_overrides: str) -> GDNBackendConfig:
    """Parse global and per-operator backend configuration."""

    parsed_mode = _parse_mode(mode)
    overrides: dict[GDNOperator, GDNBackendMode] = {}
    if not operator_overrides.strip():
        return GDNBackendConfig(parsed_mode, overrides)

    for raw_entry in operator_overrides.split(","):
        entry = raw_entry.strip()
        if not entry or entry.count("=") != 1:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}; expected operator=backend.")
        raw_operator, raw_backend = (part.strip().lower() for part in entry.split("=", 1))
        try:
            operator = GDNOperator(raw_operator)
        except ValueError as exc:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: unknown operator.") from exc
        try:
            backend = GDNBackendMode(raw_backend)
        except ValueError as exc:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: unknown backend.") from exc
        if backend is GDNBackendMode.AUTO:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: auto is only a global mode.")
        if operator in overrides:
            raise ValueError(f"Invalid GDN operator backend override {raw_entry!r}: duplicate operator.")
        overrides[operator] = backend

    return GDNBackendConfig(parsed_mode, overrides)
