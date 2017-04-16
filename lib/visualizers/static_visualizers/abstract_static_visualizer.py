from .. import AbstractVisualizer


class AbstractStaticVisualizer(AbstractVisualizer):
    """
    The AbstractStaticVisualizer acts as a base class for all static visualizers. Automatically calls _plot method 
    on __init__ call.
    """
    def __init__(self, axis, *args, **kwargs):
        super(AbstractStaticVisualizer, self).__init__(axis)
        self._plot(*args, **kwargs)

    def _key(self):
        raise NotImplementedError

    def _plot(self, *args, **kwargs):
        raise NotImplementedError
