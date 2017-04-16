from .. import AbstractSet


class ControllerSet(AbstractSet):
    """
    The ControllerSet is a wrapper around a list of controllers. Its main use comes to light when training 
    multiple actor-critic agents using different seeds. The distribution of training results can easily be visualized 
    through on of the ConcrollerCollection methods.
    """

    def __init__(self, controller_list, force_id=None):
        super(ControllerSet, self).__init__(controller_list)
        self._id = force_id if force_id is not None else self._id

    def get_controller_ids(self):
        """
        :return: Returns list of controller ids 
        """
        return [c.get_id() for c in self]

    def get_train_results(self):
        """
        :return: Returns list of training results 
        """
        return [c.get_train_results() for c in self]

    def shrink(self, n):
        """
        Shrinks controller set to n controllers 
        :param n: number of controllers to shrink to
        :return: shrunk ControllerSet
        """
        return self if n >= len(self) else ControllerSet(self[:n], force_id=self.get_id())

    def lookback_result(self, lookback, look_back_metric="median"):
        """
        Creates list of SimulationResults for lookback window
        :param lookback: number of episode to lookback on
        :param look_back_metric: 'median' or 'mean'
        :return: 
        """
        return [c.get_last_results(lookback, look_back_metric=look_back_metric) for c in self]

    def get_best(self, lookback, look_back_metric="median", s=1):
        """
        Returns best Controller according to a lookback window 
        :param lookback: number of episode to lookback on
        :param look_back_metric: 'median' or 'mean'
        :param s: index of state variable 
        :return: 
        """
        return min(
            self,
            key=lambda x: x.get_last_results(lookback, look_back_metric=look_back_metric).get_cum_state_error()[s].sum()
        )
