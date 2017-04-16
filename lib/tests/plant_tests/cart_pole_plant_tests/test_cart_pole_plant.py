import numpy as np

from ..plant_base_test import PlantBaseTest
from ....controllers import FixedInputController
from ....features import Feature
from ....plants.cart_pole_plant import CartPolePlant
from ....sets import FeatureSet


class TestCartPolePlant(PlantBaseTest):
    CART_MASS = 1.2
    POLE_LENGTH = 1.4
    POLE_MASS = 0.5
    FRICTION_COEFFICIENT = 1
    STATE = np.array([[
        1.1, 0.5,
        0.8 * np.pi, -0.05 * np.pi
    ]]).T
    PREPROCESSED_STATE = STATE.copy()
    ACTION = np.array([[0.5]]).T
    PREPROCESSED_ACTION = ACTION.copy()
    TIME_STEP = 0.01
    MOD = [False, False, 2 * np.pi, False]
    INTEGRATOR = "rk4"
    OUT_OF_BOUNDS_STATE = np.array([[
        -30, -0.01,
        -2 * np.pi, 0.01
    ]]).T
    DERIVATIVE = np.array([[
        0.5, 1.70593987,
        -0.15707963, 5.07161866
    ]]).T
    NEXT_STATE = np.array([[
        1.10508503, 0.51697978,
        2.51195706, -0.10631853
    ]]).T
    INITIAL_STATE = np.array([[
        -0.105553, 0.,
        2.562034, 0.
    ]]).T
    LIKE_ME_ARRAY = np.array([[1.1, 0.8 * np.pi, 0.5]]).T
    LIKE_ME_OTHER_FEATURE_SET = FeatureSet([
        Feature(r"$x$ [m]"),
        Feature(r"$\theta$"),
        Feature("force", feature_type="action")
    ])
    SIMULATION_STATES = np.array([
        [0, 1.24650461e-04, 4.97185419e-04, 1.11543947e-03, 1.97719376e-03],
        [0, 2.48949110e-02, 4.95759906e-02, 7.40378392e-02, 9.82751647e-02],
        [3.14159265e+00, -3.14150362e+00, -3.14123760e+00, -3.14079633e+00, -3.14018170e+00],
        [0, 1.77799972e-02, 3.53947903e-02, 5.28281277e-02, 7.00639265e-02]
    ])
    SIMULATION_ACTIONS = np.array([[3., 3., 3., 3., 3.]])
    SIMULATION_LENGTH = 5 * TIME_STEP
    SIMULATION_CONTROLLER = FixedInputController(np.array([[3]]))
    SIMULATION_INITIAL_STATE = np.array([[0, 0, np.pi, 0]]).T

    @staticmethod
    def _get_plant_cls():
        return CartPolePlant

    def _get_plant_parameters(self):
        return {
            "cart_mass": self.CART_MASS,
            "cart_friction": self.FRICTION_COEFFICIENT,
            "pole_length": self.POLE_LENGTH,
            "pole_mass": self.POLE_MASS,
            "dt": self.TIME_STEP,
            "integrator": self.INTEGRATOR,
        }

    def _get_other_plant_parameters(self):
        return {
            "cart_mass": 1.4,
            "cart_friction": 2.1,
            "pole_length": self.POLE_LENGTH,
            "pole_mass": self.POLE_MASS,
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
