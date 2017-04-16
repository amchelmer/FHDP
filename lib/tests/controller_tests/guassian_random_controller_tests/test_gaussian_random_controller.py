import numpy as np

from ..controller_base_test import ControllerBaseTest
from ....controllers.gaussian_random_controller import GaussianRandomController

from numpy.testing import assert_array_almost_equal


class TestGaussianRandomController(ControllerBaseTest):
    SEED = 123
    ACTUATOR_LIMITS = np.array([
        [-5, 5],
        [-0.5, 0.4],
    ])
    RANDOM_NUMBERS = np.random.RandomState(SEED).randn(10)

    def _get_controller_cls(self):
        return GaussianRandomController

    def _get_controller_kwargs(self):
        return {
            "seed": self.SEED,
            "actuator_limits": self.ACTUATOR_LIMITS,
        }

    def test_controller_base(self):
        self._controller_base_test()

    def _test_reset(self):
        controller = self._generate_controller()
        for x in range(5):
            controller.get_action(None)
        controller.reset()
        assert_array_almost_equal(
            controller.get_action(None),
            np.array([[-1.0856306, 0.4]]).T,
            decimal=8
        )

    def _test_get_action(self):
        controller = self._generate_controller()
        assert_array_almost_equal(
            controller.get_action(None),
            np.array([[-1.0856306, 0.4]]).T,
            decimal=8
        )
