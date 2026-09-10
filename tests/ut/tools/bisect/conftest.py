import importlib.util
import sys
import types
from unittest.mock import MagicMock

if importlib.util.find_spec("psutil") is None:
    psutil = types.ModuleType("psutil")
    psutil.__spec__ = importlib.util.spec_from_loader("psutil", loader=None)
    psutil.Error = RuntimeError  # type: ignore[attr-defined]
    psutil.Process = MagicMock()  # type: ignore[attr-defined]
    psutil.process_iter = MagicMock(return_value=[])  # type: ignore[attr-defined]
    sys.modules["psutil"] = psutil
