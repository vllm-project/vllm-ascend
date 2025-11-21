from abc import ABCMeta, abstractmethod


class DummyAttentionMetadataBuilder(metaclass=ABCMeta):
    """
    When Model DP is turned on, the idle DP needs to build fake data to run with it.
    At this time, the attention metadata builder needs to inherit this interface to implement build_dummy method.
    """

    @abstractmethod
    def build_dummy(self, *args, **kwargs):
        pass

    @abstractmethod
    def mark_static_for_attn_metadata(self, *args, **kwargs):
        pass
