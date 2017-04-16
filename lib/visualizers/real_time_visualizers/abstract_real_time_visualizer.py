import matplotlib.animation as am
import numpy as np

from ..abstract_visualizer import AbstractVisualizer
from ...env import FIGURE_PATH
from ...validation.type_validation import assert_is_type
from ...simulations import Simulation


class AbstractRealTimeVisualizer(AbstractVisualizer):
    """
    The AbstractRealTimeVisualizer is a class for visualizing motion and animating objects. It can create movies from 
    Simulations and static plots containing frozen moments of an animation.
    """
    X_SPACING = 1.
    Z_SPACING = 1.

    def __init__(self, simulation, axis):
        super(AbstractRealTimeVisualizer, self).__init__(axis)
        assert_is_type(simulation, Simulation)
        self._sim = simulation
        self._rewards = self._sim.get_rewards()
        self._states = self._sim.get_states()
        self._actions = self._sim.get_actions()
        self._dt = self._sim.get_plant().get_time_step()
        self._time_template = "time: {:.2f}s"
        self._reward_template = "reward: {:+.1f}"
        self.get_axis().autoscale(enable=False)

    def _key(self):
        raise NotImplementedError

    def get_n_steps(self):
        """
        :return: Return number of time steps in Simulation 
        """
        return self._states.shape[1]

    def init_draw(self):
        """
        Method for drawing the first, blank canvas of an animation
        """
        raise NotImplementedError

    def animate(self, i):
        """
        Method for iteratively drawing frames of the animation. Draws frame i.
        :param i: Index of frame to draw
        """
        raise NotImplementedError

    def run(self):
        """
        Run animation.
        :return: Matplotlib Animation
        """
        return am.FuncAnimation(
            self._figure,
            self.animate,
            frames=self.get_n_steps(),
            interval=self._dt * 1000,
            blit=False,
            init_func=self.init_draw()
        )

    def save_movie(self, name, bitrate=2000, *args, **kwargs):
        """
        Save MP4 movie of Simulation to disk in figs_path 
        :param name: filename as string
        :param bitrate: desired bitrate as integer
        :param args: 
        :param kwargs: 
        """
        self.run().save(
            "{:s}{:s}.mp4".format(FIGURE_PATH, name),
            fps=int(1. / self._dt),
            bitrate=bitrate,
            *args,
            **kwargs
        )
