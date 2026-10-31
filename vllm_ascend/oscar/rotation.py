"""OSCAR Rotation Matrix Loader for vllm-ascend.

Loads pre-computed orthogonal rotation matrices R_k and R_v from .pt files.
These matrices are computed offline by the OSCAR calibration process.
"""

import os
from typing import Optional

import torch
from vllm.logger import logger


class OscarRotationLoader:
    """Loads and caches OSCAR rotation matrices."""

    _instance: Optional["OscarRotationLoader"] = None
    _from_env_signature: tuple | None = None

    def __init__(
        self,
        k_rotation_path: str,
        v_rotation_path: str,
        num_layers: int = 36,
        head_dim: int = 128,
        device: str = "npu",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.num_layers = num_layers
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype

        logger.info("[OSCAR] Loading rotation matrices:\n  K: %s\n  V: %s", k_rotation_path, v_rotation_path)

        k_data = torch.load(k_rotation_path, map_location="cpu", weights_only=True)
        v_data = torch.load(v_rotation_path, map_location="cpu", weights_only=True)

        # Handle OSCAR .pt format: dict with 'layers' key
        self._R_k = self._extract_rotation(k_data, num_layers, head_dim).to(dtype=dtype, device=device)
        self._R_v = self._extract_rotation(v_data, num_layers, head_dim).to(dtype=dtype, device=device)

        assert self._R_k.shape == (num_layers, head_dim, head_dim), (
            f"K rotation shape mismatch: expected ({num_layers}, {head_dim}, {head_dim}), got {self._R_k.shape}"
        )
        assert self._R_v.shape == (num_layers, head_dim, head_dim), (
            f"V rotation shape mismatch: expected ({num_layers}, {head_dim}, {head_dim}), got {self._R_v.shape}"
        )

        # Pre-compute R_v.T (inverse rotation for output)
        self._R_v_inv = self._R_v.transpose(-1, -2).contiguous()

        logger.info("[OSCAR] Rotation matrices loaded: R_k=%s, R_v=%s", self._R_k.shape, self._R_v.shape)

    @staticmethod
    def _extract_rotation(data, num_layers: int, head_dim: int) -> torch.Tensor:
        """Extract rotation matrix from .pt file (handles dict or tensor format)."""
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, dict) and "layers" in data:
            # OSCAR format: dict with layers[i]['rotation']
            rotations = []
            layers = data["layers"]
            for i in range(num_layers):
                if i not in layers and str(i) not in layers:
                    raise ValueError(f"rotation checkpoint is missing layer {i} of {num_layers}")
                layer_data = layers[i] if i in layers else layers[str(i)]
                if isinstance(layer_data, dict) and "rotation" in layer_data:
                    rotations.append(layer_data["rotation"])
                elif isinstance(layer_data, torch.Tensor):
                    rotations.append(layer_data)
                else:
                    raise ValueError(f"Unexpected layer data format at layer {i}")
            return torch.stack(rotations, dim=0)
        else:
            raise ValueError(f"Unsupported rotation file format: {type(data)}")

    @classmethod
    def from_env(
        cls,
        num_layers: int,
        head_dim: int,
        device: str = "npu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> Optional["OscarRotationLoader"]:
        """Create loader from environment variables.

        Required env vars:
            VLLM_OSCAR_K_ROTATION_PATH
            VLLM_OSCAR_V_ROTATION_PATH
        """
        k_path = os.environ.get("VLLM_OSCAR_K_ROTATION_PATH", "")
        v_path = os.environ.get("VLLM_OSCAR_V_ROTATION_PATH", "")

        if not k_path or not v_path:
            return None
        if not os.path.exists(k_path) or not os.path.exists(v_path):
            logger.warning("[OSCAR] Rotation files not found: k=%s, v=%s", k_path, v_path)
            return None

        # normalize device: callers pass either "npu" or torch.device("npu"),
        # and torch.device("npu") == "npu" is False — the guard would reject a
        # semantically identical reconfiguration
        requested = (num_layers, head_dim, str(device), str(dtype))
        if cls._instance is None:
            cls._instance = cls(
                k_rotation_path=k_path,
                v_rotation_path=v_path,
                num_layers=num_layers,
                head_dim=head_dim,
                device=device,
                dtype=dtype,
            )
            cls._instance._from_env_signature = requested
        elif getattr(cls._instance, "_from_env_signature", None) != requested:
            # The loader is a singleton: silently ignoring conflicting
            # parameters would hand one model a rotation stack sized for
            # another. Fail closed instead.
            raise RuntimeError(
                "OscarRotationLoader.from_env already initialized with "
                f"{cls._instance._from_env_signature}, refusing conflicting "
                f"reconfiguration {requested}"
            )
        return cls._instance

    def get_k_rotation(self, layer_idx: int) -> torch.Tensor:
        """Get K rotation matrix for a layer. Shape: [head_dim, head_dim]."""
        return self._R_k[layer_idx]

    def get_v_rotation(self, layer_idx: int) -> torch.Tensor:
        """Get V rotation matrix for a layer. Shape: [head_dim, head_dim]."""
        return self._R_v[layer_idx]

    def get_v_inverse_rotation(self, layer_idx: int) -> torch.Tensor:
        """Get V inverse rotation (R_v.T). Shape: [head_dim, head_dim]."""
        return self._R_v_inv[layer_idx]

    def rotate_k(self, k: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """K_rot = K @ R_k[layer]"""
        return torch.matmul(k, self._R_k[layer_idx])

    def rotate_v(self, v: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """V_rot = V @ R_v[layer]"""
        return torch.matmul(v, self._R_v[layer_idx])

    def rotate_q(self, q: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Q_rot = Q @ R_k[layer]  (same rotation as K)"""
        return torch.matmul(q, self._R_k[layer_idx])

    def inverse_rotate_output(self, output: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """out = output @ R_v.T[layer]"""
        return torch.matmul(output, self._R_v_inv[layer_idx])


# --- backend integration loader (relocated verbatim from the monolith) ----
# Used by the INT2 backend (`AscendINT2AttentionBackendImpl`) to load the
# per-layer rotation pair at backend init. The write/read/output contract:
#   write: K/V quantized AFTER `k @ R_k[l]`, `v @ R_v[l]` (cache stores
#          rotated KV);
#   read:  query rotated with the SAME matrix `q @ R_k[l]` at attention
#          time;
#   output: attention output inverse-rotated `out @ R_v[l].T`.
# Enabling rotation without all three legs (or the inverse) corrupts
# retrieval; R218_ROT_REQUIRED=1 makes a missing asset fail-closed.
def load_oscar_rotations(
    path: str,
    layer_num: int,
    start_layer: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Load per-layer Oscar rotation matrices from ``path``.

    The checkpoint schema is the one produced by the offline Oscar pipeline::

        {"layers": {layer_id: {"rotation": Tensor[head_dim, head_dim]}, ...}}

    Returns a stacked tensor of shape ``[layer_num, head_dim, head_dim]`` in
    ``dtype`` on ``device``, indexed by local layer index
    (``global_layer_id - start_layer``). Raises ``ValueError`` if any layer in
    ``[start_layer, start_layer + layer_num)`` is missing or has mismatched
    head_dim.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    if "layers" not in state:
        raise ValueError(f"Oscar rotation checkpoint at {path} missing 'layers' key")
    layers = state["layers"]
    out = torch.empty((layer_num, head_dim, head_dim), dtype=dtype)
    for local in range(layer_num):
        global_lid = start_layer + local
        if global_lid not in layers and str(global_lid) not in layers:
            raise ValueError(f"Oscar rotation checkpoint at {path} missing layer {global_lid}")
        ldata = layers.get(global_lid, layers.get(str(global_lid)))
        R = ldata["rotation"]
        if R.shape != (head_dim, head_dim):
            raise ValueError(
                f"Oscar rotation layer {global_lid} has shape {tuple(R.shape)}, expected ({head_dim}, {head_dim})"
            )
        out[local] = R.to(dtype)
    logger.info(
        "Loaded Oscar rotation from %s for layers [%d, %d) head_dim=%d dtype=%s",
        path,
        start_layer,
        start_layer + layer_num,
        head_dim,
        dtype,
    )
    return out.to(device)
