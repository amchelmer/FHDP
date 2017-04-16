import matplotlib.pyplot as plt
import numpy as np

from .. import AbstractSet
from ...simulations.simulation import Simulation
from ...validation.format_validation import assert_list_of_value
from ...validation.type_validation import assert_list_of_type


class SimulationSet(AbstractSet):
    """
    The SimulationCollection object is a wrapper around a list of Simulation objects. It requires all Simulations to be 
    from the same plant. 
    """

    def __init__(self, simulations):
        self._validate_parameters(simulations)
        super(SimulationSet, self).__init__(simulations)

    @staticmethod
    def _validate_parameters(simulations):
        """
        Validates the list of simulations.
        :param simulations: list of Simulation objects
        """
        assert_list_of_type(simulations, Simulation)
        assert_list_of_value([sim.get_plant() for sim in simulations])

    def get_plant(self):
        """
        :return: Returns plant object 
        """
        return self[0].get_plant()

    def get_best_simulation(self):
        """
        Returns simulation with highest reward
        :return: Simulation object
        """
        return max(
            self,
            key=lambda x: x.get_total_reward()
        )

    def get_worst_simulation(self):
        """
        Returns simulation with lowest reward
        :return: Simulation object
        """
        return min(
            self,
            key=lambda x: x.get_total_reward()
        )

    def get_total_reward_list(self):
        """
        Returns array with cumulative rewards of all Simulation objects
        :return: np.array of shape (n_simulations, )
        """
        return np.array([sim.get_total_reward() for sim in self])

    def plot_total_rewards(self):
        """
        Plots total rewards as a function of Simulation number
        :return: Matplotlib Axis object
        """
        f, axis = plt.subplots()
        axis.plot(self.get_total_reward_list())
        axis.set_ylabel("Total reward [-]")
        axis.set_xlabel("Learning episode [-]")
        return axis

    def replay(self, n=5):
        """
        Replays n Simulations starting with the last
        :param n: Number of Simulations to be replayed

        """
        n = min(n, len(self))
        for i in range(len(self) - 1, len(self) - n - 1, -1):
            print("Replaying simulation {:d}".format(i + 1))
            _ = self[i].replay()
            plt.show()
