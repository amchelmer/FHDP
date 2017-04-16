import numpy as np
import pickle

from ..abstract_object import AbstractObject
from ..env import DUMP_PATH
from ..tools.math_tools import hashable
from ..reward_functions import AbstractRewardFunction
from ..validation.object_validation import assert_true
from ..visualizers.static_visualizers.simulation_static_visualizer import SimulationStaticVisualizer


class Simulation(AbstractObject):
    """
    The simulation object describes an RL episode. It contains the states visited during the episode, the actions taken,
    and possibly also the values and rewards obtained. Simulation objects are central to interpreting results of the 
    runs. They can be animated or a time-series overview of all parameters can be plotted at once.
    """

    def __init__(self, plant, states, actions, values=None, rewards=None):
        super(Simulation, self).__init__()

        self._plant = plant
        self._states = states.copy()
        self._actions = actions.copy()
        self._nstates, self._nsteps = states.shape
        self._nactions, _ = actions.shape
        assert_true(states.shape[1] == actions.shape[1], "Inconsistent dimensions of state and actions")

        self._rewards = np.zeros((1, self._nsteps))
        self._values = np.zeros((1, self._nsteps))

        if isinstance(rewards, AbstractRewardFunction):
            self._rewards = np.array([rewards.get_reward(*r[:2]) for r in self]).reshape(1, -1)
        elif rewards is not None:
            self._rewards = rewards

        if hasattr(values, "get_value"):
            self._values = np.array([values.get_value(r[0])[0] for r in self]).reshape(1, -1)
        elif values is not None:
            self._values = values

    def __len__(self):
        return self._nsteps

    def __iter__(self):
        for i in range(len(self)):
            yield (
                self._states[:, i].reshape(-1, 1),
                self._actions[:, i].reshape(-1, 1),
                self._values[:, i].reshape(-1, 1),
                self._rewards[:, i].reshape(-1, 1),
            )

    def _key(self):
        return (
            self._plant,
            hashable(self._states),
            hashable(self._actions),
            hashable(self._values),
            hashable(self._rewards)
        )

    def get_actions(self):
        """
        :return: Returns actions as np.array of shape (n_actions, nsteps) 
        """
        return self._actions

    def get_states(self):
        """
        :return: Returns states as np.array of shape (n_states, nsteps) 
        """
        return self._states

    def get_values(self):
        """
        :return: Returns values as np.array of shape (1, nsteps)
        """
        return self._values

    def get_rewards(self):
        """
        :return: Returns values as np.array of shape (1, nsteps)
        """
        return self._rewards

    def get_total_reward(self):
        """
        :return: Sums reward signal as float 
        """
        return self._rewards.sum()

    def get_plant(self):
        """
        :return: Returns plant object 
        """
        return self._plant

    def cum_state_error(self, reward_function):
        """
        Computes the cumulative state error over the entire simulation
        :param reward_function: RewardFunction object
        :return: np.array of shape (nstates, 1)
        """
        return self.get_state_error(reward_function).sum(axis=1).reshape(-1, 1)

    def get_state_error(self, reward_function):
        """
        Computes the state errors during the simulation
        :param reward_function: RewardFunction object
        :return: np.array of shape (nstates, nsteps)
        """
        return np.hstack([np.abs(reward_function.compute_state_error(s)) for (s, _, _, _) in self])

    def get_time_vector(self):
        """
        Generates time vector
        :return: np.array of shape (nsteps, )
        """
        return np.arange(self._nsteps) * self._plant.get_time_step()

    def plot_time_series(self, *args, **kwargs):
        """
        Plots a set of time series describing the states, actions, values and rewards.
        :param args: 
        :param kwargs: 
        :return: Visualizer object
        """
        return SimulationStaticVisualizer(self, *args, **kwargs)

    def replay(self):
        """
        Animate the simulation (in a loop)
        :return: Matplotlib Animation
        """
        vis_cls = self._plant.get_visualizer_cls()
        vis = vis_cls(self)
        return vis.run()

    def save_replay(self, name="simulation", *args, **kwargs):
        """
        Save animation to an MP4 movie.
        :param name: Filename
        :param args: additional args
        :param kwargs: additional kwargs
        """
        vis_cls = self._plant.get_visualizer_cls()
        vis = vis_cls(self)
        vis.save_movie(name, *args, **kwargs)

    def static_replay(self, axis=None, stepsize=3):
        """
        Visualizes the motion of the simulation in a static plot
        :param axis: Matplotlib axis object
        :param stepsize: Frequency of states that are printed
        :return: StaticVisualizer
        """
        vis_cls = self._plant.get_visualizer_cls()
        vis = vis_cls(self, axis=axis)
        vis.plot_static(stepsize=stepsize)
        return vis

    def dump(self, file_handle=None):
        """
        Dumps object to file in DUMP_PATH, requires hashable object
        :param file_handle: python file handle object
        """
        if not isinstance(file_handle, file):
            file_handle = open(
                "{:s}{:s}-{:d}.pckl".format(
                    DUMP_PATH,
                    self.__class__.__name__,
                    self.get_id()
                ),
                "wb")
        pickle.dump(self, file_handle)

    @classmethod
    def load(cls, file_id):
        """
        Load object from DUMP_PATH
        :param file_id: python file handle object
        :return: Simulation object
        """
        if not isinstance(file_id, file):
            handle = open(
                "{:s}{:s}-{:d}.pckl".format(
                    DUMP_PATH,
                    cls.__name__,
                    file_id
                ),
                "rb")
        else:
            handle = file_id
        return pickle.load(handle)
