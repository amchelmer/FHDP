from ..abstract_controller import AbstractController


class ZeroInputController(AbstractController):
    """
    The ZeroInputController always returns a zero action for any state. Useful for simulating plant dynamics given an 
    initial state, but without controller actions.
    """

    def __init__(self, actuator_limits):
        super(ZeroInputController, self).__init__(actuator_limits)

    def _key(self):
        raise NotImplementedError

    def get_action(self, state):
        """
        Compute the action given a state
        :param state: numpy.array
        :return: numpy.array of shape(naction, 1)
        """
        return self.get_zero_action()

    def reset(self):
        """
        Resets the controller to the original state
        """
        pass
