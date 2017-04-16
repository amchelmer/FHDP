import numpy as np

from ..abstract_controller import AbstractController
from ...tools.math_tools import saturate


class FixedInputController(AbstractController):
    """
    The FixedInputController always returns an action of 'values', no matter the state.
    """
    def __init__(self, values, actuator_limits=None):
        super(FixedInputController, self).__init__(
            actuator_limits if actuator_limits is not None else np.tile([-np.inf, +np.inf], (values.shape[0], 1))
        )
        self._values = values

    def _key(self):
        raise NotImplementedError

    def get_action(self, state):
        """
        Compute the action given a state
        :param state: numpy.array
        :return: numpy.array of shape(naction, 1)
        """
        return saturate(
            self._values,
            self._actuator_limits
        )

    def reset(self):
        """
        Resets the controller to the original state
        """
        pass
