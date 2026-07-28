def __getattr__(name):
    if name == "ExpertOffloadManager":
        from .expert_offload_manager import ExpertOffloadManager

        return ExpertOffloadManager
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["ExpertOffloadManager"]
