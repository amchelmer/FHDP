import numpy as np

from .. import AbstractSet
from ...visualizers.static_visualizers.learning_process_static_visualizer import LearningProcessStaticVisualizer


class RewardSet(AbstractSet):
    """
    The RewardSet class describes the reward signal over a number of episodes. It can do this for a set of controllers 
    to show the distribution of results over the set.
    """

    def __init__(self, controller_set):
        super(RewardSet, self).__init__(controller_set.get_controller_ids())
        self._training_results = controller_set.get_train_results()
        self._id = controller_set.get_id()

    def get_controller_ids(self):
        """
        :return: Returns list of controller ids 
        """
        return [s for s in self]

    def get_training_rewards(self):
        """
        :return: Returns the results as an 2D numpy.array of shape(n_controllers, n_episodes) 
        """
        return np.array([[sr.get_cum_reward() for sr in cr] for cr in self._training_results])

    def plot(self, *args, **kwargs):
        """
        Plots the results in a graph. 
        :return: LearningProcessStaticVisualizer instance
        """
        return LearningProcessStaticVisualizer(self, *args, **kwargs)
