import numpy as np

from ..abstract_real_time_visualizer import AbstractRealTimeVisualizer


class InvertedPendulumRealTimeVisualizer(AbstractRealTimeVisualizer):
    """
    RealTimeVisualizer class for the InvertedPendulum plant.
    """

    def __init__(self, simulation, axis=None):
        super(InvertedPendulumRealTimeVisualizer, self).__init__(simulation, axis)

        self._pole_length = self._sim.get_plant().get_pole_length()
        self._pole_start = (0, 0)

        self.get_axis().set_xlim([-self._pole_length * 2, self._pole_length * 2])
        self.get_axis().set_ylim([-self._pole_length * 2, self._pole_length * 2])
        self.get_axis().autoscale(enable=False)
        self.get_axis().set_aspect("equal")
        self.get_axis().grid(True)

        self._pole_line, = self.get_axis().plot([], [], 'o-', lw=2, color="tomato")
        self._ground_line = self.get_axis().plot(
            self.get_axis().get_xlim(),
            (0, 0),
            color="k",
            lw=1
        )
        self._force_vector, = self.get_axis().plot([], [], ':', lw=3, color="k")

        self.get_axis().set_title("Total reward for run is {:+.2f}".format(self._sim.get_total_reward()))

        self._time_text = self.get_axis().text(0.05, 0.9, "", transform=self.get_axis().transAxes)
        self._reward_text = self.get_axis().text(0.75, 0.9, "", transform=self.get_axis().transAxes)

    def _key(self):
        raise NotImplementedError

    def init_draw(self):
        """
        Method for drawing the first, blank canvas of an animation
        """
        self._pole_line.set_data([], [])
        self._force_vector.set_data([], [])
        self._time_text.set_text("")
        self._reward_text.set_text("")

    def animate(self, i):
        """
        Method for iteratively drawing frames of the animation. Draws frame i.
        :param i: Index of frame to draw
        """
        theta, theta_dot = self._states[:, i]
        action = -.01 * self._actions[0, i]

        pole_end = (
            +self._pole_length * np.sin(theta),
            +self._pole_length * np.cos(theta)
        )
        self._pole_line.set_data(
            np.array([self._pole_start[0], pole_end[0]]),
            np.array([self._pole_start[1], pole_end[1]]),
        )

        self._force_vector.set_data(
            np.array([pole_end[0], pole_end[0] - action * np.cos(theta)]),
            np.array([pole_end[1], pole_end[1] + action * np.sin(theta)]),
        )

        self._time_text.set_text(self._time_template.format(i * self._dt))
        self._reward_text.set_text(self._reward_template.format(self._rewards[0, i]))

    def plot_static(self, stepsize=1):
        """
        Make static plot that shows frozen motion. 
        :param stepsize: Interval at which snapshots are taken
        """
        for i in np.arange(1, self.get_n_steps(), stepsize):
            theta, theta_dot = self._states[:, i]

            pole_end = (
                +self._pole_length * np.sin(theta),
                +self._pole_length * np.cos(theta)
            )
            self._pole_line, = self.get_axis().plot(
                np.array([self._pole_start[0], pole_end[0]]),
                np.array([self._pole_start[1], pole_end[1]]),
                'o-',
                lw=1,
                color="{:.2f}".format(1 - (float(i) / self.get_n_steps()))
            )
            self.get_axis().set_title("")
        self._name = "static-replay"
