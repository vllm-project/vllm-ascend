from functools import wraps
from unittest.mock import MagicMock, patch

from vllm.distributed.parallel_state import all_gather_fake


def patch_distributed_groups(dcp_size=1,
                             dcp_rank=0,
                             pcp_size=1,
                             pcp_rank=0,
                             needs_mocks=True):
    """
    Decorator to patch common distributed group mocks with configuration
    
    Args:
        dcp_size: DCP world size (default: 1)
        dcp_rank: DCP rank (default: 0)
        pcp_size: PCP world size (default: 1)
        pcp_rank: PCP rank (default: 0)
        needs_mocks: Whether to pass mock objects as the first arguments 
             after 'self' to the decorated function. 
             If True, the decorated function receives: 
                 func(self, mock_all_to_all_single, mock_dcp, mock_pcp, *args, **kwargs)
             If False, mocks are not passed and function receives: 
                 func(self, *args, **kwargs)
             (default: True)
    """

    def decorator(func):

        @wraps(func)
        @patch('torch.distributed.all_to_all_single')
        @patch('vllm.distributed.parallel_state._PCP')
        @patch('vllm.distributed.parallel_state._DCP')
        def wrapper(self, mock_dcp, mock_pcp, mock_all_to_all_single, *args,
                    **kwargs):
            mock_dcp.rank_in_group = dcp_rank
            mock_dcp.world_size = dcp_size
            mock_dcp.device_group = MagicMock()

            mock_dcp.all_gather = MagicMock()
            mock_dcp.all_gather.side_effect = lambda input_, dim: all_gather_fake(
                input_, dim, mock_dcp.world_size, "mock_dcp_group")

            mock_pcp.rank_in_group = pcp_rank
            mock_pcp.world_size = pcp_size
            mock_pcp.device_group = MagicMock()

            mock_pcp.all_gather = MagicMock()
            mock_pcp.all_gather.side_effect = lambda input_, dim: all_gather_fake(
                input_, dim, mock_pcp.world_size, "mock_pcp_group")

            mock_all_to_all_single.side_effect = lambda output, input, *a, **kw: output.copy_(
                input)

            if needs_mocks:
                return func(self, mock_all_to_all_single, mock_dcp, mock_pcp,
                            *args, **kwargs)
            else:
                return func(self, *args, **kwargs)

        return wrapper

    return decorator
