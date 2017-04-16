import numpy as np

from ..plant_base_test import PlantBaseTest
from ....controllers import FixedInputController
from ....features import Feature
from ....plants.inverted_pendulum_plant import InvertedPendulumPlant
from ....sets import FeatureSet


class TestInvertedPendulumPlant(PlantBaseTest):
    POLE_MASS = 3.0e-2  # kg
    POLE_LENGTH = 8.4e-2  # m
    POLE_DAMPING = 3.33e-4  # Nms/rad
    TORQUE_CONSTANT = 6.16e-3  # N m / V
    STATE = np.array([[0.9 * np.pi, -0.1 * np.pi]]).T
    PREPROCESSED_STATE = STATE.copy()
    ACTION = np.array([[1.5]]).T
    PREPROCESSED_ACTION = ACTION.copy()
    TIME_STEP = 0.01
    MOD = [2 * np.pi, False]
    INTEGRATOR = "rk4"
    OUT_OF_BOUNDS_STATE = np.array([[0.9 * np.pi, 21 * np.pi]]).T
    INITIAL_STATE = np.array([
        [3.031058],
        [0.]
    ])
    DERIVATIVE = np.array([
        [-0.31415927],
        [80.2214533]
    ])
    NEXT_STATE = np.array([
        [2.828284],
        [0.48203862]
    ])
    LIKE_ME_ARRAY = np.array([[0.15, -1.2]]).T
    LIKE_ME_OTHER_FEATURE_SET = FeatureSet([
        Feature(r"$\theta$ [rad]"),
        Feature("volt [v]", feature_type="action"),
    ])
    SIMULATION_STATES = np.array([
        [1.57079633, 1.60178075, 1.65251882, 1.7226662, 1.81182141],
        [2.1, 4.09157016, 6.05035133, 7.97249675, 9.8502371],
    ])
    SIMULATION_ACTIONS = 3 * np.ones((1, 5))
    SIMULATION_LENGTH = 5 * TIME_STEP
    SIMULATION_CONTROLLER = FixedInputController(np.array([[3]]))
    SIMULATION_INITIAL_STATE = np.array([[np.pi / 2, 2.1]]).T

    @staticmethod
    def _get_plant_cls():
        return InvertedPendulumPlant

    def _get_plant_parameters(self):
        return {
            "pole_length": self.POLE_LENGTH,
            "pole_mass": self.POLE_MASS,
            "pole_damping": self.POLE_DAMPING,
            "torque_constant": self.TORQUE_CONSTANT,
            "dt": self.TIME_STEP,
            "integrator": self.INTEGRATOR,
        }

    def _get_other_plant_parameters(self):
        return {
            "pole_length": self.POLE_LENGTH,
            "pole_mass": self.POLE_MASS,
            "pole_damping": self.POLE_DAMPING + 0.1,
            "torque_constant": self.TORQUE_CONSTANT,
            "dt": self.TIME_STEP,
            "integrator": self.INTEGRATOR,
        }

    def plant_base_test(self):
        self._plant_base_test()

    def test_get_pole_length(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_pole_length(),
            plant._pole_length
        )

    def test_get_pole_mass(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_pole_mass(),
            plant._pole_mass
        )
