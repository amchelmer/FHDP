import matplotlib.patches as pt
import numpy as np

from ..abstract_real_time_visualizer import AbstractRealTimeVisualizer
from ....tools.math_tools import euler2quat


class QuadRotor2DRealTimeVisualizer(AbstractRealTimeVisualizer):
    """
    RealTimeVisualizer object for a 2-dimensional Quad-Rotor
    """
    _SPAN = 0.517 / 2.
    _ROTOR_HEIGHT = 0.020
    _ROTOR_CLEARANCE = 0.127 - _ROTOR_HEIGHT - 0.05
    _ROTOR_DIAMETER = 0.10 * 2
    _BODY_WIDTH = 2 * (_SPAN - 0.15)
    _BODY_HEIGHT = _BODY_WIDTH / 3.
    _ACTION_SCALE = 1000. * 2

    _BODY_TRUSS_XYZ_BODY = [
        np.array([[-_SPAN, 0, 0]]).T,
        np.array([[+_SPAN, 0, 0]]).T,
    ]
    _TRUSS_1_XYZ_BODY = [
        np.array([[+_SPAN, 0, 0]]).T,
        np.array([[+_SPAN, 0, -_ROTOR_CLEARANCE]]).T,
    ]
    _TRUSS_3_XYZ_BODY = [
        np.array([[-_SPAN, 0, 0]]).T,
        np.array([[-_SPAN, 0, -_ROTOR_CLEARANCE]]).T,
    ]
    _ROTOR_1_XYZ_BODY = _TRUSS_1_XYZ_BODY[1]
    _ROTOR_3_XYZ_BODY = _TRUSS_3_XYZ_BODY[1]
    _BODY_CENTER_BODY = np.array([[0, 0, 0]]).T

    Z_SPACING = 0.2

    def __init__(self, simulation, axis=None):
        super(QuadRotor2DRealTimeVisualizer, self).__init__(simulation, axis)

        self._plant = simulation.get_plant()

        x_array, z_array, _, _, _, _ = simulation.get_states()

        self.get_axis().set_xlim(
            x_array.min() - self.X_SPACING,
            x_array.max() + self.X_SPACING,
        )
        self.get_axis().set_ylim(
            z_array.max() + self.Z_SPACING,
            z_array.min() - self.Z_SPACING,

        )
        self.get_axis().autoscale(enable=False)
        self.get_axis().set_aspect("equal")
        self.get_axis().grid(True, zorder=1)

        self._body_truss, = self.get_axis().plot([], [], '-', lw=1, color="k", zorder=2)

        self._truss_1, = self.get_axis().plot([], [], '-', lw=1, color="k", zorder=2)
        self._truss_3, = self.get_axis().plot([], [], '-', lw=1, color="k", zorder=2)

        self._rotor_3 = self.get_axis().add_patch(
            pt.Ellipse(
                xy=self.convert_2d(self.body_to_earth(
                    np.zeros((6, 1)),
                    self._ROTOR_3_XYZ_BODY
                )).flatten(),
                width=self._ROTOR_DIAMETER,
                height=self._ROTOR_HEIGHT,
                facecolor="slategrey",
                linewidth=0.1,
                zorder=3,
            )
        )

        self._rotor_1 = self.get_axis().add_patch(
            pt.Ellipse(
                xy=self.convert_2d(self.body_to_earth(
                    np.zeros((6, 1)),
                    self._ROTOR_1_XYZ_BODY
                )).flatten(),
                width=self._ROTOR_DIAMETER,
                height=self._ROTOR_HEIGHT,
                facecolor="slategrey",
                linewidth=0.1,
                zorder=3,
            )
        )

        self._body_blob = self.get_axis().add_patch(
            pt.Ellipse(
                xy=self.convert_2d(self.body_to_earth(
                    np.zeros((6, 1)),
                    self._BODY_CENTER_BODY
                )).flatten(),
                width=self._BODY_WIDTH,
                height=self._BODY_HEIGHT,
                facecolor="k",
                alpha=None,
                zorder=3,
            )
        )
        self._time_text = self.get_axis().text(
            0.05,
            0.9,
            "",
            transform=self.get_axis().transAxes
        )
        self._reward_text = self.get_axis().text(
            0.65,
            0.9,
            "",
            transform=self.get_axis().transAxes
        )

        self.body_truss_locations = []
        self.body_blob_centers = []
        self.thetas = []
        self.truss_1_locations, self.truss_3_locations = [], []
        self.rotor_1_locations, self.rotor_3_locations = [], []

        self._prerun_simulation()

    def _key(self):
        raise NotImplementedError

    def _prerun_simulation(self):
        """
        Compute all positions of the shapes in advance to speed up animation quality

        """
        for i in range(self.get_n_steps()):
            state = self._states[:, i]
            x, z, _, _, theta, _ = state
            body_truss_location = self.convert_2d(self.get_body_truss_location(state))

            truss_1 = self.get_truss_1_location(state)
            truss_3 = self.get_truss_3_location(state)

            self.body_truss_locations.append(body_truss_location)
            self.truss_1_locations.append(self.convert_2d(truss_1))
            self.truss_3_locations.append(self.convert_2d(truss_3))
            self.body_blob_centers.append(self.convert_2d(self.body_to_earth(state, self._BODY_CENTER_BODY)).flatten())
            self.thetas.append(-theta)

    def init_draw(self):
        """
        Method for drawing the first, blank canvas of an animation
        """
        self._truss_3.set_data([], [])
        self._truss_1.set_data([], [])
        self._body_truss.set_data([], [])
        self._time_text.set_text("")
        self._reward_text.set_text("")

    def animate(self, i):
        """
        Method for iteratively drawing frames of the animation. Draws frame i.
        :param i: Index of frame to draw
        """
        self._body_truss.set_data(*self.body_truss_locations[i])
        theta = self.thetas[i]

        self._body_blob.center = self.body_blob_centers[i]
        self._body_blob.angle = np.rad2deg(theta)

        truss_3 = self.truss_3_locations[i]
        self._truss_3.set_data(*truss_3)
        self._rotor_3.center = truss_3[:, -1]
        self._rotor_3.angle = np.rad2deg(theta)

        truss_1 = self.truss_1_locations[i]
        self._truss_1.set_data(*truss_1)
        self._rotor_1.center = truss_1[:, -1]
        self._rotor_1.angle = np.rad2deg(theta)

        self._time_text.set_text(self._time_template.format(i * self._dt))
        self._reward_text.set_text(self._reward_template.format(self._rewards[0, i]))

    def get_body_truss_location(self, state):
        """
        Computes the location of the body truss in Earth coordinates
        :param state: numpy.array of shape(6,1)
        :return: xyz position in E-frame as numpy.array of shape(3,1)
        """
        return np.hstack([
            self.body_to_earth(state, self._BODY_TRUSS_XYZ_BODY[0]),
            self.body_to_earth(state, self._BODY_TRUSS_XYZ_BODY[1]),
        ])

    def get_truss_3_location(self, state):
        """
        Computes the location of the right truss in Earth coordinates
        :param state: numpy.array of shape(6,1)
        :return: xyz position in E-frame as numpy.array of shape(3,1)
        """
        return np.hstack([
            self.body_to_earth(state, self._TRUSS_1_XYZ_BODY[0]),
            self.body_to_earth(state, self._TRUSS_1_XYZ_BODY[1]),
        ])

    def get_truss_1_location(self, state):
        """
        Computes the location of the left truss in Earth coordinates
        :param state: numpy.array of shape(6,1)
        :return: xyz position in E-frame as numpy.array of shape(3,1)
        """
        return np.hstack([
            self.body_to_earth(state, self._TRUSS_3_XYZ_BODY[0]),
            self.body_to_earth(state, self._TRUSS_3_XYZ_BODY[1]),
        ])

    def plot_static(self, stepsize=1):
        """
        Make static plot that shows frozen motion. 
        :param stepsize: Interval at which snapshots are taken
        """
        self._name = "static-replay"
        self.get_axis().set_title("")
        for i in np.arange(1, self.get_n_steps(), stepsize):
            x_pos, z_pos = self.body_truss_locations[i]
            center = self.body_blob_centers[i]
            color = "{:.2f}".format(1 - (float(i) / self.get_n_steps()))
            self._body_truss, = self.get_axis().plot(
                np.array([x_pos[0], center[0], x_pos[1]]),
                np.array([z_pos[0], center[1], z_pos[1]]),
                'o-',
                lw=1,
                color=color,
                markeredgecolor="black",
            )
        self.get_axis().set_xlabel("Lateral position x [m]")
        self.get_axis().set_ylabel("Vertical position z [m]")

    @staticmethod
    def convert_2d(array):
        """
        Project 3D array to 2D surface
        :param array: numpy.array of shape(3, n) 
        :return: numpy.array of shape(2, n)
        """
        return np.array([
            array[0, :],
            array[2, :]
        ])

    def body_to_earth(self, state, vector):
        """
        Transforms body reference frame to Earth reference frame.
        :param state: numpy.array of shape(6, 1)
        :param vector: xyz numpy.array of shape (3, 1)
        :return: 
        """
        x, z, _, _, theta, _ = state.flatten()
        quaternions = euler2quat(np.array([[0, theta, 0]]).T)
        return self._plant.rotate_body_to_earth(quaternions, vector) + np.array([[x, 0, z]]).T
