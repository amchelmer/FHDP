import numpy as np

from ..abstract_object import AbstractObject


class AbstractController(AbstractObject):
    """
    The AbstractController class is a basis for any controller. It ensures that all classes inhereting from it have 
    the get_action and reset methods. 
    """

    def __init__(self, actuator_limits):
        super(AbstractController, self).__init__()
        self._actuator_limits = actuator_limits.astype(float)
        self._n_actions = self._actuator_limits.shape[0]

    def _key(self):
        raise NotImplementedError

    def get_action(self, state):
        """
        Compute the action given a state
        :param state: numpy.array
        :return: numpy.array of shape(naction, 1)
        """
        raise NotImplementedError

    def get_zero_action(self):
        """
        :return: Returns a zero action as a numpy.array of shape(nactions,1) 
        """
        action = np.zeros((self._n_actions, 1))
        self.logger.debug("Action is {}".format(action))
        return action

    def reset(self):
        """
        Resets the controller to the original state
        """
        raise NotImplementedError
