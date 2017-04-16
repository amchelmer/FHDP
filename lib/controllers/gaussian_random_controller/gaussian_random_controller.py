import numpy as np

from ..abstract_controller import AbstractController
from ...tools.math_tools import saturate


class GaussianRandomController(AbstractController):
    """
    Controller class that yields randomly distributed action according to a Gaussian distribution with 
    sigma=actuator_limits/3.
    """

    def __init__(self, seed, actuator_limits):
        super(GaussianRandomController, self).__init__(actuator_limits)
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        self._std = self._actuator_limits[:, 1:] / 3.

    def _key(self):
        raise NotImplementedError

    def get_action(self, state):
        """
        Compute the action given a state
        :param state: numpy.array
        :return: numpy.array of shape(naction, 1)
        """
        random_action = self._rng.randn(self._actuator_limits.shape[0], 1)
        return saturate(
            random_action,
            self._actuator_limits
        )

    def reset(self):
        """
        Resets the controller to the original state
        """
        self._rng = np.random.RandomState(self._seed)
