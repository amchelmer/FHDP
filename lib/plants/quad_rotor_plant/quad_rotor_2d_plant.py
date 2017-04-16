import numpy as np

from .quad_rotor_3d_plant import QuadRotor3DPlant
from ...features import Feature
from ...sets import FeatureSet
from ...tools.math_tools import saturate, euler2quat, quat2euler
from ...visualizers.real_time_visualizers.quad_rotor_real_time_visualizer import QuadRotor2DRealTimeVisualizer


class QuadRotor2DPlant(QuadRotor3DPlant):
    """
    Plant dynamics of the Parrot AR drone mapped onto the xz plane. Plant uses euler angles for attitude
    representation (as opposed to quaternions). Ported from the Simulink model by Djim Molenkamp.
    """
    _FEATURE_SET = FeatureSet([
        Feature(r"$x$ [m]", scale=2.7257602277),
        Feature(r"$z$ [m]", scale=2.7257602277, bounds=np.array([-25, 0])),
        Feature(r"$\dot{x}$ [m/s]", derivative=True, scale=5., bounds=15 * np.array([-1, 1])),
        Feature(r"$\dot{z}$ [m/s]", derivative=True, scale=5., bounds=15 * np.array([-1, 1])),
        Feature(r"$\theta$ [rad]", scale=2.301822),
        Feature(r"$\dot{\theta}$ [rad/s]", scale=6.357142, derivative=True),
        Feature(r"$a_1$ [-]", feature_type="action", bounds=np.array([-1, 1])),
        Feature(r"$a_2$ [-]", feature_type="action", scale=0.760859, bounds=0.3 * np.array([-1, 1])),
    ])
    _INITIAL_STATE_MEAN = np.zeros((len(_FEATURE_SET.get_state_set()), 1))
    _INITIAL_STATE_SIGMA = np.zeros((len(_FEATURE_SET.get_state_set()), 1))

    _VISUALIZER_CLS = QuadRotor2DRealTimeVisualizer

    def __init__(self, dt, integrator="euler", blade_flapping=True, init_mean=None, init_std=None, feature_set=None):
        feature_set = self._FEATURE_SET if feature_set is None else feature_set
        super(QuadRotor2DPlant, self).__init__(
            dt,
            integrator=integrator,
            blade_flapping=blade_flapping,
            feature_set=feature_set,
            init_mean=self._INITIAL_STATE_MEAN if init_mean is None else init_mean,
            init_std=self._INITIAL_STATE_SIGMA if init_std is None else init_std
        )

    def preprocess_state_action(self, state, action):
        """
        Preprocess action, by mapping from alternative rotor speeds to regular rotor speeds and by saturating the input.
        :param state: np.array with 2D state, shape (6,1)
        :param action: np.array of shape (4,1) in space [-1, 1]^4
        :return: np.array of shape (4,1)
        """
        x, z, xdot, zdot, theta, q = state.flatten()
        q0, _, q2, _ = euler2quat(np.array([0, theta, 0]))

        state_3d = np.array([[
            x, 0, z,
            xdot, 0, zdot,
            q0, 0, q2, 0,
            0, q, 0,
        ]]).T

        u1, u3 = action.flatten() * self._action_scale
        action_out = saturate(
            self.map_input(
                np.array([[u1 + self._rpm_lift_off, 0, u3, 0]]).T
            ),
            self._ROTOR_SPEED_SATURATION
        )

        return state_3d, action_out

    def postprocess_next_state(self, next_state):
        """
        :param next_state: state vector for 3D quad-rotor, shape (13,1)
        :return: state_2d: state vector for 2D quad-rotor, shape (6,1)
        """
        x, _, z, xdot, _, zdot, q0, _, q2, _, _, q, _ = next_state.flatten()
        _, theta, _ = quat2euler(np.array([q0, 0, q2, 0])).flatten()
        state_2d = np.array([[
            x, z,
            xdot, zdot,
            theta,
            q
        ]]).T

        return state_2d
