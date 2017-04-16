import numpy as np

from ..abstract_plant import AbstractPlant
from ...constants.physical_constants import AIR_DENSITY_AT_SEA_LEVEL, GRAVITATIONAL_ACCELERATION
from ...features import Feature
from ...sets import FeatureSet
from ...tools.math_tools import saturate, normalize_quaternions, hashable


class QuadRotor3DPlant(AbstractPlant):
    """
    Plant dynamics of the Parrot AR drone. Ported from the Simulink model by Djim Molenkamp.
    """

    _FEATURE_SET = FeatureSet([
        Feature(r"$x$ [m]"),
        Feature(r"$y$ [m]"),
        Feature(r"$z$ [m]", scale=5., bounds=np.array([-25, 0])),
        Feature(r"$\dot{x}$ [m/s]"),
        Feature(r"$\dot{y}$ [m/s]"),
        Feature(r"$\dot{z}$ [m/s]", scale=5.),
        Feature(r"$q_0$ [-]"),
        Feature(r"$q_1$ [-]"),
        Feature(r"$q_2$ [-]"),
        Feature(r"$q_3$ [-]"),
        Feature(r"$\dot{\phi}$ [rad/s]", scale=10.),
        Feature(r"$\dot{\theta}$ [rad/s]", scale=10.),
        Feature(r"$\dot{\psi}$ [rad/s]", scale=10.),
        Feature(r"$u_1$ [rpm]", feature_type="action", bounds=np.array([-1, 1])),
        Feature(r"$u_2$ [rpm]", feature_type="action", bounds=np.array([-1, 1])),
        Feature(r"$u_3$ [rpm]", feature_type="action", bounds=np.array([-1, 1])),
        Feature(r"$u_4$ [rpm]", feature_type="action", bounds=np.array([-1, 1])),
    ])
    _INITIAL_STATE_MEAN = np.array([[0, 0, 0, 0, 0, 0, 1., 0, 0, 0, 0, 0, 0]]).T
    _INITIAL_STATE_SIGMA = np.zeros_like(_INITIAL_STATE_MEAN)

    _MOD = None
    _VISUALIZER_CLS = None

    _POWER_FACTOR = 5.3e-10
    _VOLTAGE_CONSTANT = 42.5
    _OMEGA_CONSTANT = 0.1
    _DRAG_CONSTANT = 0.1
    _ROTOR_INERTIA = 0.0184
    _CENTER_MASS = 0.22
    _ARM_MASS = 0.05
    _ARM_LENGTH = 0.15
    _TOTAL_MASS = _CENTER_MASS + 4 * _ARM_MASS
    _INERTIA_3D = np.array([
        [2 * _ARM_LENGTH ** 2 * _ARM_MASS, 0, 0],
        [0, 2 * _ARM_LENGTH ** 2 * _ARM_MASS, 0],
        [0, 0, 4 * _ARM_LENGTH ** 2 * _ARM_MASS]
    ])
    _SURFACE = 0.04
    _TORQUE_COEFFICIENT = 0.005
    _EDGE_ANGLE = np.deg2rad(85.)
    _CL_ANGLES_RADIANS = np.deg2rad(np.arange(0, 92, 2, dtype=np.float64))
    _CL_COEFFICIENTS = np.array([
        0, 0.20, 0.40, 0.60, 0.40, 0.36, 0.35, 0.35, 0.36, 0.37, 0.38, 0.40, 0.42, 0.44, 0.47, 0.50, 0.53, 0.56, 0.58,
        0.60, 0.62, 0.63, 0.64, 0.65, 0.65, 0.65, 0.64, 0.63, 0.62, 0.60, 0.58, 0.56, 0.53, 0.50, 0.47, 0.43, 0.39,
        0.35, 0.31, 0.27, 0.23, 0.19, 0.15, 0.10, 0.05, 0
    ])
    _HOVER_COEFFICIENT = 0.10 ** 2 * np.pi
    _BETA_COEFFICIENT = 0.25
    _MBF_COEFFICIENT = 0.1 / 10.
    _ROTOR_SPEED_SATURATION = np.array([[0, 3.4e+03]]).repeat(len(_FEATURE_SET.get_action_set()), axis=0)
    _INPUT_VOLTAGE_SATURATION = np.array([[0, 16]]).repeat(len(_FEATURE_SET.get_action_set()), axis=0)

    def __init__(self, dt, integrator="euler", init_mean=None, init_std=None, blade_flapping=True, feature_set=None):
        super(QuadRotor3DPlant, self).__init__(
            dt,
            integrator,
            self._FEATURE_SET if feature_set is None else feature_set,
            self._INITIAL_STATE_MEAN if init_mean is None else init_mean,
            self._INITIAL_STATE_SIGMA if init_std is None else init_std
        )
        self._rpm_lift_off = self._compute_hover_thrust()
        self._action_scale = min(
            self._rpm_lift_off - self._ROTOR_SPEED_SATURATION[0, 0],
            self._ROTOR_SPEED_SATURATION[0, 1] - self._rpm_lift_off,
        )
        self._blade_flapping_flag = blade_flapping

    def _key(self):
        return (
            self._time_step,
            self._feature_set,
            hashable(self._state_modulus),
            self._integrator_type,
            hashable(self._bounds),
            hashable(self._initial_state_mean),
            hashable(self._initial_state_sigma),
            self._rpm_lift_off,
            self._blade_flapping_flag,
            self._action_scale,
        )

    def _compute_hover_thrust(self):
        """
        Computes rpm at which the quad-rotor lifts off.
        :return: float of rpm (per engine
        """
        lift_off_power_per_rotor = np.divide(
            (self._TOTAL_MASS * GRAVITATIONAL_ACCELERATION / 4.) ** (3. / 2.),
            np.sqrt(2 * AIR_DENSITY_AT_SEA_LEVEL * self._HOVER_COEFFICIENT)
        )
        hover_rpm = (lift_off_power_per_rotor / self._POWER_FACTOR) ** (1. / 3.)
        self.logger.debug("Computed hover rpm: {}".format(hover_rpm))
        return hover_rpm

    def compute_derivative(self, state, action):
        """
        Compute the time derivative of the state x_dot
        :param state: state as a numpy.array of shape (13,1)
        :param action: mapped rotor speeds as a numpy.array of shape(4,1)
        :return: time-derivatives for state as numpy.array of shape (13,1)
        """
        saturated_rotor_speeds = action

        _, velocities, attitude_quaternions, omega_body = self.split_state(state)
        forces, moments = self.get_forces_and_moments(state, saturated_rotor_speeds)

        body_forces = np.add(
            self.rotate_body_to_earth(attitude_quaternions, forces),
            np.array([[0, 0, self._TOTAL_MASS * GRAVITATIONAL_ACCELERATION]], dtype=np.float64).T
        )

        accelerations = body_forces / self._TOTAL_MASS
        cross = np.cross(
            omega_body.flatten(),
            np.dot(self._INERTIA_3D, omega_body).flatten()
        )
        omega_body_dot = np.linalg.lstsq(
            self._INERTIA_3D,
            moments - cross.reshape(3, 1)
        )[0]
        qdot = 0.5 * self.get_wb(attitude_quaternions).T.dot(omega_body)

        return np.vstack([velocities, accelerations, qdot, np.nan_to_num(omega_body_dot)])

    def get_power_by_voltage(self, rotor_speeds, input_voltage):
        """
        Compute power given the current rotor speeds and the
        :param rotor_speeds: numpy.array with shape (4,1)
        :param input_voltage: numpy.array with shape (4,1)
        :return: power: numpy.array with shape (4,1)
        """
        moment = self._VOLTAGE_CONSTANT * input_voltage - self._OMEGA_CONSTANT * rotor_speeds
        drag_moment = rotor_speeds * self._DRAG_CONSTANT
        new_rotor_speeds = rotor_speeds + (moment - drag_moment) / self._ROTOR_INERTIA * self._time_step
        return (
            new_rotor_speeds,
            self._POWER_FACTOR * np.power(new_rotor_speeds, 3)
        )

    def get_power_by_omega(self, rotor_speeds):
        """
        Compute power given the current rotor speeds and the
        :param rotor_speeds: numpy.array with shape (4,1)
        :return: power: numpy.array with shape (4,1)
        """
        return self._POWER_FACTOR * np.power(rotor_speeds, 3)

    def get_lift_coefficient(self, alpha):
        """
        Compute Life coefficient by interpolating lookup table in class variable
        :param alpha: alpha in radians (float)
        :return: float
        """
        return np.interp(alpha, self._CL_ANGLES_RADIANS, self._CL_COEFFICIENTS)

    @staticmethod
    def split_state(state):
        """
        Splits state into position,velocity,attitude and rotational velocity
        :param state: numpy.array of shape (13,1) with states
        :return:
        """
        return state[:3], state[3:6], state[6:10], state[10:]

    def get_forces_and_moments(self, state, rotor_speeds):
        """
        Compute aerodynamic forces and moments from state and action
        :param state: numpy.array with shape (13,1)
        :param rotor_speeds: numpy.array with shape (4,1)
        :return: two numpy.arrays each with shape (3,1)
        """
        forces_moments = (
            self._get_propeller_forces_and_moments(state, rotor_speeds) +
            self._get_fuselage_forces_and_moments(state)
        )
        return forces_moments[:3], forces_moments[3:]

    def _compute_thrust(self, engine_power):
        """
        Computes thrust from engine power
        :param engine_power: engine power as numpy.array of shape(4,1)
        :return: thrust as numpy.array of shape(4,1)
        """
        s = np.sign(engine_power)
        return s * (s * engine_power * np.sqrt(2 * AIR_DENSITY_AT_SEA_LEVEL * self._HOVER_COEFFICIENT)) ** (2. / 3)

    def _compute_total_velocity(self, velocities_body, omega_body):
        """
        Computes total velocity
        :param velocities_body: Body velocities as numpy.array of  
        :param omega_body: 
        :return: 
        """
        omega_flat = omega_body.flatten()
        total_velocity = velocities_body.T + np.multiply(
            self._ARM_LENGTH,
            np.array([
                [0, 0, -omega_flat[1]],
                [0, 0, +omega_flat[0]],
                [0, 0, +omega_flat[1]],
                [0, 0, -omega_flat[0]],
            ], dtype=np.float64)
        )
        return total_velocity

    @staticmethod
    def _compute_thrust_ratio(velocities):
        """
        
        :param velocities: 
        :return: 
        """
        vinf = np.linalg.norm(velocities, ord=2, axis=1).reshape(4, 1)
        return np.array([1.15 - 0.15 * np.cos(vinf.flatten() / 20 * np.pi) + 0.05 * velocities[:, 2]]).reshape(4, 1)

    def _get_propeller_forces_and_moments(self, state, rotor_speeds):
        """
        Method for computing propeller forces and moments from state and rotor speeds 
        :param state: numpy.array of shape(13,1)
        :param rotor_speeds: 
        :return: numpy.array of shape (6,1) representing 3 forces and 3 moments
        """
        _, velocities, attitude_quaternions, omega_body = self.split_state(state)

        engine_power = self.get_power_by_omega(rotor_speeds)
        thrust = self._compute_thrust(engine_power)

        velocities_body = self.rotate_earth_to_body(attitude_quaternions, velocities)
        velocities = self._compute_total_velocity(velocities_body, omega_body)

        thrust_ratio = self._compute_thrust_ratio(velocities)
        current_thrust = np.multiply(thrust, thrust_ratio)
        ct_1, ct_2, ct_3, ct_4 = current_thrust.flatten()

        try:
            tau_engine = 0.5 * np.divide(engine_power, rotor_speeds)
        except FloatingPointError:
            tau_engine = 0.5 * np.array(
                [e / r if r != 0 else 0 for (e, r) in zip(engine_power, rotor_speeds)]
            ).reshape(4, 1)

        if self._blade_flapping_flag:
            betax_rad = np.deg2rad(np.multiply(-self._BETA_COEFFICIENT, velocities[:, 1])).reshape(4, 1)
            betay_rad = np.deg2rad(np.multiply(+self._BETA_COEFFICIENT, velocities[:, 0])).reshape(4, 1)

            force_bfx = np.multiply(current_thrust, np.sin(betax_rad))
            force_bfy = np.multiply(current_thrust, np.sin(betay_rad))

            moment_bfx = np.multiply(self._MBF_COEFFICIENT, betax_rad)
            moment_bfy = np.multiply(self._MBF_COEFFICIENT, betay_rad)
        else:
            force_bfx, force_bfy = np.zeros((3, 1)), np.zeros((3, 1))
            moment_bfx, moment_bfy = np.zeros((3, 1)), np.zeros((3, 1))

        forces = np.array([[
            -force_bfy.sum(),
            +force_bfx.sum(),
            -current_thrust.sum()
        ]], dtype=np.float64).T

        moments = np.array([[
            (ct_4 - ct_2) * self._ARM_LENGTH + moment_bfx.sum(),
            (ct_1 - ct_3) * self._ARM_LENGTH + moment_bfy.sum(),
            tau_engine[1] + tau_engine[3] - tau_engine[0] - tau_engine[2]
        ]], dtype=np.float64).T

        return np.vstack([forces, moments])

    def _get_fuselage_forces_and_moments(self, state):
        """
        Compute aerodynamic forces and moments based on state.
        :param state:
        :return: 3 forces,3 moments as numpy.array with shape(6,1)
        """
        _, velocities, attitude_quaternions, omega_body = self.split_state(state)

        velocity_body = self.rotate_earth_to_body(attitude_quaternions, velocities)
        velocity_mag = np.linalg.norm(velocities, ord=2)

        try:
            alpha_ratio = np.clip(velocity_body[2] / velocity_mag, -1, 1)
        except FloatingPointError:
            alpha_ratio = 1

        try:
            alpha, alpha_c = (0, 0) if velocity_mag < 0.01 else (np.arcsin(alpha_ratio), np.abs(np.arcsin(alpha_ratio)))
        except FloatingPointError:
            raise FloatingPointError("Invalid value encountered with values: {}".format(alpha_ratio))

        drag_coefficient = -np.cos(2 * alpha) * 0.75 + 1.25
        drag = drag_coefficient * AIR_DENSITY_AT_SEA_LEVEL * velocity_mag ** 2 * self._SURFACE

        lift = np.zeros((3, 1), dtype=np.float64)
        if 0 < alpha_c < self._EDGE_ANGLE:
            lift_coefficient = 0.1 * self.get_lift_coefficient(alpha_c)
            lift_magnitude = lift_coefficient * AIR_DENSITY_AT_SEA_LEVEL * velocity_mag ** 2 * self._SURFACE

            velocity_xy = velocity_mag * np.cos(alpha)
            lift += np.array([[0, 0, -np.sign(velocity_body[2]) * np.cos(alpha) * lift_magnitude]]).T

            if not velocity_xy == 0:
                lift += np.multiply(
                    np.array([[1, 1, 0]]).T,
                    np.abs(lift_magnitude * np.sin(alpha)) * velocity_body / velocity_xy
                )
        try:
            forces = lift - np.divide(drag * velocity_body, velocity_mag)
        except FloatingPointError:
            forces = np.zeros((3, 1), dtype=np.float64)

        moments = np.array([[
            0,
            0,
            np.sign(omega_body[-1]) * -0.5 * AIR_DENSITY_AT_SEA_LEVEL * omega_body[-1] ** 2 * self._TORQUE_COEFFICIENT
        ]], dtype=np.float64).T

        return np.vstack([forces, moments])

    @classmethod
    def rotate_earth_to_body(cls, quaternions, vector, reverse=False):
        """
        Rotate vector in earth frame to body frame
        :param quaternions: numpy.array of shape (4,1) with quaternions
        :param vector: numpy.array of shape (3,1) of to-be-rotated vector
        :param reverse: boolean whether to transpose rotation matrix. When True rotation is done from body frame to 
        earth frame. 
        :return:
        """
        q0, q1, q2, q3 = quaternions.flatten()
        rq = np.array([
            [q0 ** 2 + q1 ** 2 - q2 ** 2 - q3 ** 2, 2 * q1 * q2 + 2 * q0 * q3, 2 * q1 * q3 - 2 * q0 * q2],
            [2 * q1 * q2 - 2 * q0 * q3, q0 ** 2 - q1 ** 2 + q2 ** 2 - q3 ** 2, 2 * q2 * q3 + 2 * q0 * q1],
            [2 * q1 * q3 + 2 * q0 * q2, 2 * q2 * q3 - 2 * q0 * q1, q0 ** 2 - q1 ** 2 - q2 ** 2 + q3 ** 2],
        ])
        if reverse:
            rq = rq.T
        return np.dot(rq, vector)

    @classmethod
    def rotate_body_to_earth(cls, quaternions, vector):
        """
        Rotate vector in body frame to Earth frame
        :param quaternions: vector with quaternions as numpy.array of shape(4,1)
        :param vector: numpy.array of shape (3,1) of to-be-rotated vector
        :return: 
        """
        return cls.rotate_earth_to_body(quaternions, vector, reverse=True)

    @staticmethod
    def map_input(inputs):
        """
        Input mapping from artificial inputs to rotor speeds
        :param inputs: artificial rotor speeds:
                        u1 is collective speed,
                        u2 controls the roll: + rotor 4 - rotor 2
                        u3 controls the pitch: + rotor 1 - rotor 3
                        u4 controls the yaw: + (rotor 2 + rotor 4) - (rotor 1 + rotor 3)
        :return: numpy.array of shape (4,1) with rotor speeds
        """
        u1, u2, u3, u4 = inputs.flatten()
        return np.array([[
            u1 + u3 - u4,
            u1 - u2 + u4,
            u1 - u3 - u4,
            u1 + u2 + u4,
        ]]).T

    def preprocess_state_action(self, state, action):
        """
        Preprocess state. Function is overridden in 2D quad-rotor case to map from 2D to 3D.
        :param state: numpy.array of shape (12,1)
        :param action: numpy.array of shape (4,1)
        :return: numpy.array of shape (12,1) and numpy.array of shape (4,1)
        """
        position, velocities, attitude_quaternions, omega_body = self.split_state(state)

        state_out = np.vstack([
            position,
            velocities,
            normalize_quaternions(attitude_quaternions),
            omega_body
        ])

        scaled_action = action * self._action_scale + np.array([[self._rpm_lift_off, 0, 0, 0]]).T
        action_out = saturate(
            self.map_input(scaled_action),
            self._ROTOR_SPEED_SATURATION
        )

        return state_out, action_out

    def postprocess_next_state(self, next_state):
        """
        Postprocess state. Function is overridden in 2D quad-rotor case to map from 3D to 2D.
        :param next_state: numpy.array of shape (12,1)
        :return: numpy.array of shape (12,1)
        """
        position, velocities, attitude_quaternions, omega_body = self.split_state(next_state)
        position_saturated = saturate(
            position,
            self.get_bounds()[:3, :]
        )
        return np.vstack([
            position_saturated,
            velocities,
            normalize_quaternions(attitude_quaternions),
            omega_body
        ])

    @staticmethod
    def get_wb(quaternions):
        """
        Compute matrix for converting euler rates to quaternion rates
        :param quaternions: quaternions vector as numpy.array with shape(4,1)
        :return: numpy.array of shape(3,4) 
        """
        q0, q1, q2, q3 = quaternions.flatten()
        return np.array([
            [-q1, q0, q3, -q2],
            [- q2, -q3, q0, q1],
            [- q3, q2, -q1, q0],
        ])
