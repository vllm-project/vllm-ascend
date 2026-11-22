from dataclasses import dataclass

import torch


@dataclass
class TreeAcceptance:
    """Device-side result shared by verification and accepted-state transport.

    ``path_node_ids`` is [R, K], excludes the root and uses -1 for its unused
    suffix. The other fields are [R]. ``num_sampled`` includes the bonus token
    and ``last_accepted_node_id`` is the last accepted node, or zero for the root.
    An inactive row has zero sampled tokens and uses the root node ID.
    Buffers belong to the verifier and remain valid until its next call.
    """

    path_node_ids: torch.Tensor
    num_sampled: torch.Tensor
    last_accepted_node_id: torch.Tensor
    active: torch.Tensor
