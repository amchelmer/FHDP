import numpy as np

from ..abstract_reward_function import AbstractRewardFunction
from ...tools.math_tools import hashable, center_mod


class QuadraticErrorRewardFunction(AbstractRewardFunction):
    """
    Quadratic reward function class. Main purpose is to return a reward based on a given state and action. Also supplies
     the derivative of the reward to the action.
    """

    def __init__(self, action_weights, state_weights, desired_state=None, state_mod=None):
        super(QuadraticErrorRewardFunction, self).__init__()
        self._action_weights = np.array(np.diag(action_weights), dtype=np.float64)
        self._state_weights = np.array(np.diag(state_weights), dtype=np.float64)

        desired_state = desired_state if desired_state is not None else np.zeros((self._state_weights.shape[0], 1))
        state_mod = state_mod if state_mod is not None else [False] * desired_state.shape[0]

        self._des_state = desired_state
        self._state_mod = state_mod

    def _key(self):
        return (
            tuple(self.get_state_weights()),
            tuple(self.get_action_weights()),
            hashable(self._des_state),
        )

    def get_reward(self, state, action):
        """
        Compute reward for given state and action
        :param state: state as a np.array of shape(nstates, 1)
        :param action: action as a np.array of shape(nactions, 1)
        :return: reward as a float
        """
        state_error = self.compute_state_error(state)
        separate_rewards = self._compute_quadratic_reward(state_error, action)
        self.logger.debug(
            "State reward: {:.2f}, action reward: {:.2f}, total reward: {:.2f}".format(separate_rewards.flatten()[0],
                                                                                       separate_rewards.flatten()[1],
                                                                                       separate_rewards.sum())
        )
        return separate_rewards.sum()

    def get_derivative_to_action(self, action):
        """
        Computes derivative with respect to action
        :return: Derivative as numpy.array of shape(nactions, 1)
        """
        deriv = - self._action_weights.dot(action)
        self.logger.debug(
            "dR/du is {} for action {}".format(deriv.flatten(), action.flatten())
        )
        return deriv

    def _compute_quadratic_reward(self, state_error, action):
        """
        Computes quadratic reward with separate entries for state and action part
        :return: numpy.array of shape(2,1)
        """
        return -0.5 * np.vstack([
            state_error.T.dot(self._state_weights).dot(state_error),
            action.T.dot(self._action_weights).dot(action),
        ])

    def compute_state_error(self, state):
        """
        Computes state error with respect to desired state.
        :param state: numpy.array of shape(nstates, 1)
        :return: numpy.array of shape(nstates, 1)
        """
        return center_mod(self._des_state - state, self._state_mod)

    def get_state_weights(self):
        """
        Returns the state weights as a 1D numpy.array 
        """
        return np.diag(self._state_weights)

    def get_action_weights(self):
        """
        Returns the action weights as a 1D numpy.array 
        """
        return np.diag(self._action_weights)

    def get_state_mod(self):
        """
        Returns the state mod variable as list 
        """
        return self._state_mod

    def add_weights(self, action_weights, state_weights):
        """
        Update the state and action weights by adding an additional quanity
        :param action_weights: numpy.array of shape (self._action_weights, )
        :param state_weights: numpy.array of shape (self._state_weights, )
        """
        self._action_weights += np.array(np.diag(action_weights))
        self._state_weights += np.array(np.diag(state_weights))
