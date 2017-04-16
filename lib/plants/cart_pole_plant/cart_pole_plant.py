import numpy as np

from ..abstract_plant import AbstractPlant
from ...constants.physical_constants import GRAVITATIONAL_ACCELERATION
from ...features import Feature
from ...sets import FeatureSet
from ...tools.math_tools import hashable
from ...visualizers.real_time_visualizers.cart_pole_real_time_visualizer import CartPoleRealTimeVisualizer


class CartPolePlant(AbstractPlant):
    """
    Simple non-linear dynamics for a cart-pole system. State vector is [x, x_dot, theta, theta_dot], input is [F].
    Translations and forces are defined positive to the right. Angles and moments are positive in clockwise direction.
    Theta is defined 0 when pole is standing up!

    Pole mass is assumed to act at the end of the pole, hence the length of the pole defined here, is actually only
    half the length of the actual pole.
    """

    _INITIAL_STATE_MEAN = np.array([[0, 0, np.pi, 0]]).T
    _INITIAL_STATE_SIGMA = np.array([[1, 0, np.pi / 3, 0]]).T
    _FEATURE_SET = FeatureSet([
        Feature(r"$x$ [m]", bounds=15 * np.array([-1, 1])),
        Feature(r"$\theta$", bounds=20 * np.array([-1, 1])),
        Feature(r"$\dot{x}$ [m/s]"),
        Feature(r"$\dot{\theta}$", scale=10., bounds=20 * np.pi * np.array([-1, 1])),
        Feature("force", feature_type="action", bounds=3 * np.array([-1, 1]))
    ])
    _VISUALIZER_CLS = CartPoleRealTimeVisualizer
    _MOD = [False, False, 2 * np.pi, False]

    def __init__(self, dt, cart_mass, cart_friction, pole_mass, pole_length, integrator="RK4", feature_set=None):
        super(CartPolePlant, self).__init__(
            dt,
            integrator,
            self._FEATURE_SET if feature_set is None else feature_set,
            self._INITIAL_STATE_MEAN,
            self._INITIAL_STATE_SIGMA
        )
        self._cart_mass = cart_mass
        self._cart_friction = cart_friction
        self._pole_mass = pole_mass
        self._pole_length = pole_length

    def _key(self):
        return (
            self._FEATURE_SET,
            self._time_step,
            hashable(self._state_modulus),
            self._cart_mass,
            self._cart_friction,
            self._pole_mass,
            self._pole_length,
            self._integrator_type
        )

    def compute_derivative(self, state, action):
        """
        Method for computing the time-derivative of the state variables.
        :param state: np.array of shape(nstates, 1)
        :param action: np.array of shape(naction, 1)
        :return: derivatives in np.array of shape(nstates, 1)  
        """
        x, x_dot, theta, theta_dot = state.flatten()
        force, = action.flatten()
        ct, st = np.cos(theta), np.sin(theta)
        friction_force = self._cart_friction * x_dot
        div = self._cart_mass + self._pole_mass * st ** 2
        x_dot_dot = np.divide(
            force - friction_force - self._pole_mass * st * (-self._pole_length * theta_dot ** 2 +
                                                             GRAVITATIONAL_ACCELERATION * ct),
            div
        )
        theta_dot_dot = np.divide(
            ct * (friction_force - force) - st * (
                ct * self._pole_mass * self._pole_length * theta_dot - GRAVITATIONAL_ACCELERATION * (
                    self._pole_mass + self._cart_mass
                )
            ),
            self._pole_length * div
        )
        return np.array(
            [[x_dot, x_dot_dot, theta_dot, theta_dot_dot]]
        ).T

    def get_pole_length(self):
        """
        :return: Returns pole length as float
        """
        return self._pole_length

    def get_pole_mass(self):
        """
        :return: Returns pole mass as float
        """
        return self._pole_mass
