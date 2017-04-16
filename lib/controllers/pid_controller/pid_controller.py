import numpy as np

from ..abstract_controller import AbstractController
from ...tools.math_tools import saturate
from ...validation.format_validation import assert_shape_like


class PIDController(AbstractController):
    """
    Simple implementation of a PID controller.
    """

    def __init__(self, actuator_limits, ref_state, frequency, p_gain=None, i_gain=None, d_gain=None):
        super(PIDController, self).__init__(actuator_limits)
        self._ref_state = ref_state
        self._acc_error = np.zeros_like(ref_state)
        self._last_error = self._acc_error.copy()
        self._dt = 1. / frequency
        self._p_gain, self._i_gain, self._d_gain = self._assert_gains(p_gain, i_gain, d_gain)

    def _key(self):
        raise NotImplementedError

    def _assert_gains(self, *gains):
        """
        Assert proper definition of gains
        :param gains: Gain values for PID parts, each one must be a numpy.array of shape(nactions, nstates)
        :return: list of 3 gain numpy.arrays of shape(nactions, nstates)
        """
        out_gains = [gain if gain is not None else np.zeros(
            (self._actuator_limits.shape[0], self._ref_state.shape[0])
        ) for gain in gains]
        if len(out_gains) != 1:
            for gain in out_gains[1:]:
                assert_shape_like(gain, out_gains[0])
        return out_gains

    def reset(self):
        """
        Resets the controller to the original state 
        """
        self._acc_error = np.zeros_like(self._ref_state)
        self._last_error = np.zeros_like(self._ref_state)

    def get_action(self, state):
        """
        Compute the action given a state
        :param state: numpy.array
        :return: numpy.array of shape(naction, 1)
        """
        state_error = self.compute_error_signal(state)
        error_diff = (state_error - self._last_error) / self._dt
        sat_action = saturate(
            self._p_gain.dot(state_error) +
            self._i_gain.dot(self._acc_error) +
            self._d_gain.dot(error_diff),
            self._actuator_limits
        )
        self.logger.debug("Action is {:s}".format(sat_action.flatten()))

        self._accumulate_error(state_error)
        self._set_last_error(state_error)

        return sat_action

    def compute_error_signal(self, state):
        """
        Computes error with reference state
        :param state: state as numpy.array of shape(nstates, 1)
        :return: error as numpy.array of shape(nstates, 1)
        """
        assert_shape_like(state, self._ref_state)
        err = self._ref_state - state
        self.logger.debug("Error is {}".format(err.flatten()))
        return err

    def _set_last_error(self, error):
        """
        Sets last_error to error
        :param error: error as numpy.array of shape(nstates, 1)
        """
        self._last_error = error.copy()
        self.logger.debug("Set last error to {}".format(self._last_error.flatten()))

    def _accumulate_error(self, error):
        """
        Increment cumulative error with error
        :param error: error as numpy.array of shape(nstates, 1)
        """
        self._acc_error += error
        self.logger.debug("Set accumulated error to {}".format(self._acc_error.flatten()))
