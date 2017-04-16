import numpy as np

from .simulation import Simulation
from ..abstract_object import AbstractObject
from ..reward_functions import AbstractRewardFunction
from ..tools.math_tools import hashable
from ..validation.object_validation import assert_in
from ..validation.type_validation import assert_list_of_type, assert_is_type


class SimulationResult(AbstractObject):
    """
    Object to summarize the most import characteristics of a Simulation or a list of other SimulationResults. 
    Required for efficiently storing the learning process of an ActorCriticController.
    """
    METRICS = {
        "mean": np.mean,
        "median": np.median,
    }

    def __init__(self, simulation_obj, metric=None):
        super(SimulationResult, self).__init__()
        if isinstance(simulation_obj, Simulation):
            assert_is_type(metric, AbstractRewardFunction)
            (self._cum_reward,
             self._mean_reward,
             self._last_reward,
             self._cum_state_error,
             self._mean_state_error,
             self._last_state_error,
             self._last_state) = self._handle_simulation(simulation_obj, metric)
        elif isinstance(simulation_obj, list):
            assert_in(metric, self.METRICS)
            assert_list_of_type(simulation_obj, SimulationResult)
            (self._cum_reward,
             self._mean_reward,
             self._last_reward,
             self._cum_state_error,
             self._mean_state_error,
             self._last_state_error,
             self._last_state) = self._handle_simulation_result_list(simulation_obj, self.METRICS[metric])
        else:
            raise ValueError("Unknown simulation object type {}".format(type(simulation_obj)))

    @staticmethod
    def _handle_simulation(simulation, reward_function):
        """
        Extracts required parameters from a Simulation object
        :param simulation: Simulation object
        :param reward_function: RewardFunction object for supplying state error.
        :return: total reward, mean reward, last reward, cum state error, mean state error, last state error, last state 
        """
        state_errors = simulation.get_state_error(reward_function)
        return (
            simulation.get_total_reward(),
            simulation.get_rewards().mean(),
            simulation.get_rewards()[0, -1],
            state_errors.sum(axis=1).reshape(-1, 1),
            state_errors.mean(axis=1).reshape(-1, 1),
            state_errors[:, -1].reshape(-1, 1),
            simulation.get_states()[:, -1].reshape(-1, 1)
        )

    @staticmethod
    def _handle_simulation_result_list(simulation_result_list, metric):
        """
        Extracts required parameters from a set of SimulationResult objects
        :param simulation_result_list: Iterable of SimulationResult objects
        :param metric: Metric for summarizing results, either mean or median
        :return: total reward, mean reward, last reward, cum state error, mean state error, last state error, last state 
        """
        return (
            metric([sr.get_cum_reward() for sr in simulation_result_list]),
            metric([sr.get_mean_reward() for sr in simulation_result_list]),
            metric([sr.get_last_reward() for sr in simulation_result_list]),
            metric(np.hstack([sr.get_cum_state_error() for sr in simulation_result_list]), axis=1).reshape(-1, 1),
            metric(np.hstack([sr.get_mean_state_error() for sr in simulation_result_list]), axis=1).reshape(-1, 1),
            metric(np.hstack([sr.get_last_state_error() for sr in simulation_result_list]), axis=1).reshape(-1, 1),
            metric(np.hstack([sr.get_last_state() for sr in simulation_result_list]), axis=1).reshape(-1, 1),
        )

    def _key(self):
        return (
            self._cum_reward,
            self._mean_reward,
            self._last_reward,
            hashable(self._cum_state_error),
            hashable(self._mean_state_error),
            hashable(self._last_state_error),
            hashable(self._last_state),
        )

    def get_cum_reward(self):
        """
        :return: Returns cumulative reward as float
        """
        return self._cum_reward

    def get_mean_reward(self):
        """
        :return: Returns mean reward as float
        """
        return self._mean_reward

    def get_last_reward(self):
        """
        :return: Returns last reward as float
        """
        return self._last_reward

    def get_cum_state_error(self):
        """
        :return: Returns cumulative state error as numpy.array of shape(nstates,1)
        """
        return self._cum_state_error

    def get_mean_state_error(self):
        """
        :return: Returns mean state error as numpy.array of shape(nstates,1)
        """
        return self._mean_state_error

    def get_last_state_error(self):
        """
        :return: Returns last state error as numpy.array of shape(nstates,1)
        """
        return self._last_state_error

    def get_last_state(self):
        """
        :return: Returns last state as numpy.array of shape(nstates,1)
        """
        return self._last_state
