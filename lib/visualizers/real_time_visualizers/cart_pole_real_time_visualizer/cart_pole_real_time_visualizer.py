import matplotlib.patches as pt
import numpy as np

from ..abstract_real_time_visualizer import AbstractRealTimeVisualizer
from ....simulations import Simulation
from ....validation.type_validation import assert_is_type


class CartPoleRealTimeVisualizer(AbstractRealTimeVisualizer):
    """
    RealTimeVisualizer class for the CartPole plant.
    """
    _CART_SIZE = (0.5, 0.4, 0.3)

    def __init__(self, simulation, axis=None):
        super(CartPoleRealTimeVisualizer, self).__init__(simulation, axis)

        assert_is_type(simulation, Simulation)

        self.get_axis().set_xlim([-10, 10])
        self.get_axis().set_ylim([-6, 6])
        self.get_axis().autoscale(enable=False)
        self.get_axis().set_aspect("equal")
        self.get_axis().grid(True)

        self._cart_vertices_x, self._cart_vertices_y = self.get_vertices()
        self._pole_length = self._sim.get_plant().get_pole_length()
        self._wheel_diameter = self.get_wheel_diameter()

        self._pole_line, = self.get_axis().plot([], [], 'o-', lw=2, color="tomato")
        self._cart_line, = self.get_axis().plot([], [], lw=2, color="slategrey")
        self._ground_line = self.get_axis().plot(
            self.get_axis().get_xlim(),
            (self._cart_vertices_y.min() - self._wheel_diameter) * np.ones(2),
            color="k",
            lw=1
        )
        self._left_wheel = self.get_axis().add_patch(pt.Circle(
            (0, 0),
            0.1,
            linewidth=0,
            color="slategrey"
        ))
        self._right_wheel = self.get_axis().add_patch(pt.Circle(
            (0, 0),
            0.1,
            linewidth=0,
            color="slategrey"
        ))
        self._force_vector, = self.get_axis().plot([], [], ':', lw=3, color="dodgerblue")

        self._time_text = self.get_axis().text(0.05, 0.9, "", transform=self.get_axis().transAxes)
        self._reward_text = self.get_axis().text(0.75, 0.9, "", transform=self.get_axis().transAxes)

    def _key(self):
        raise NotImplementedError

    def init_draw(self):
        """
        Method for drawing the first, blank canvas of an animation
        """
        self._pole_line.set_data([], [])
        self._cart_line.set_data([], [])
        self._force_vector.set_data([], [])
        self._time_text.set_text("")
        self._reward_text.set_text("")

    def animate(self, i):
        """
        Method for iteratively drawing frames of the animation. Draws frame i.
        :param i: Index of frame to draw
        """
        x, x_dot, theta, theta_dot = self._states[:, i]
        action = self._actions[0, i]

        self._pole_line.set_data(
            np.array([0, +self._pole_length * np.sin(theta)]) + x,
            np.array([0, +self._pole_length * np.cos(theta)]),
        )

        self._cart_line.set_data(
            self._cart_vertices_x + x,
            self._cart_vertices_y,
        )
        new_left = (x - self._cart_vertices_x.max() / 2., self._cart_vertices_y.min() - self._wheel_diameter / 2.)
        new_right = (x + self._cart_vertices_x.max() / 2., self._cart_vertices_y.min() - self._wheel_diameter / 2.)
        self._left_wheel.center = new_left
        self._right_wheel.center = new_right

        self._force_vector.set_data(
            np.array([0, action / 8.]) + x,
            np.array([0, 0])

        )

        self._time_text.set_text(self._time_template.format(i * self._dt))
        self._reward_text.set_text(self._reward_template.format(self._rewards[0, i]))

    def get_cart_size(self):
        """
        :return: Return cart sizes as tuple 
        """
        return self._CART_SIZE

    def get_vertices(self):
        """
        Compute xy vertices of rectangle representing cart
        :return: 
        """
        x, y, z = self._CART_SIZE
        return x * np.array([-1, +1, +1, -1, -1]), z * np.array([+1, +1, -1, -1, +1])

    def get_wheel_diameter(self):
        """
        :return: Return wheel diameter 
        """
        return self._CART_SIZE[1] / 2.
