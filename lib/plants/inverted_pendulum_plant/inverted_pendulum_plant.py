import numpy as np

from ..abstract_plant import AbstractPlant
from ...constants.physical_constants import GRAVITATIONAL_ACCELERATION
from ...features import Feature
from ...sets import FeatureSet
from ...tools.math_tools import hashable
from ...visualizers.real_time_visualizers.inverted_pendulum_real_time_visualizer import \
    InvertedPendulumRealTimeVisualizer


class InvertedPendulumPlant(AbstractPlant):
    """
    Simple non-linear dynamics for a inverted pendulum system. State vector is [theta, theta_dot], input is [u].
    Angles and moments are positive in clockwise direction. Theta is defined 0 when pole is standing up!

    Pole mass is assumed to act at the end of the pole, hence the length of the pole defined here, is actually only
    half the length of the actual pole.
    """

    _INITIAL_STATE_MEAN = np.array([[np.pi, 0]]).T
    _INITIAL_STATE_SIGMA = np.array([[np.pi / 3, 0]]).T
    _FEATURE_SET = FeatureSet([
        Feature(r"$\theta$ [rad]"),
        Feature(r"$\dot{\theta} [rad/s]$", scale=10., bounds=np.array([-20 * np.pi, 20 * np.pi])),
        Feature("volt [v]", feature_type="action", bounds=3 * np.array([-1, 1])),
    ])
    _VISUALIZER_CLS = InvertedPendulumRealTimeVisualizer
    _MOD = [2 * np.pi, False]

    def __init__(self, dt, pole_mass, pole_length, pole_damping, torque_constant, integrator="rk4", feature_set=None):
        super(InvertedPendulumPlant, self).__init__(
            dt,
            integrator,
            self._FEATURE_SET if feature_set is None else feature_set,
            self._INITIAL_STATE_MEAN,
            self._INITIAL_STATE_SIGMA
        )
        self._pole_mass = pole_mass
        self._pole_length = pole_length
        self._pole_damping = pole_damping
        self._torque_constant = torque_constant

    def _key(self):
        return (
            self._time_step,
            self._FEATURE_SET,
            hashable(self._state_modulus),
            self._pole_mass,
            self._pole_length,
            self._pole_damping,
            self._torque_constant,
            self._integrator_type,
        )

    def compute_derivative(self, state, action):
        """
        Method for computing the time-derivative of the state variables.
        :param state: np.array of shape(nstates, 1)
        :param action: np.array of shape(naction, 1)
        :return: derivatives in np.array of shape(nstates, 1)  
        """
        theta, theta_dot = state.flatten()
        voltage, = action.flatten()

        theta_dot_dot = np.divide(
            self._pole_mass * GRAVITATIONAL_ACCELERATION * self._pole_length * np.sin(theta)
            - self._pole_damping * theta_dot
            + self._torque_constant * voltage,
            self._pole_mass * (self._pole_length ** 2)
        )
        return np.array([[theta_dot, theta_dot_dot]]).T

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
