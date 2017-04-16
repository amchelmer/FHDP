from ..abstract_object import AbstractObject


class AbstractRewardFunction(AbstractObject):
    """
    Abstract class for RewardFunctions.
    """

    def __init__(self):
        super(AbstractRewardFunction, self).__init__()

    def _key(self):
        raise NotImplementedError

    def get_reward(self, *args, **kwargs):
        """
        Computes reward given a state, action and future state 
        :return: reward as a float
        """
        raise NotImplementedError

    def get_derivative_to_action(self, *args, **kwargs):
        """
        Computes derivative with respect to action
        :return: Derivative as numpy.array of shape(nactions, 1)
        """
        raise NotImplementedError
