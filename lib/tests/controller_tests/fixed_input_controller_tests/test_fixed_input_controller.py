import numpy as np

from ..controller_base_test import ControllerBaseTest
from ....controllers.fixed_input_controller import FixedInputController

from numpy.testing import assert_array_equal


class TestFixedInputController(ControllerBaseTest):
    ACTION = np.array([[4.3, 1.2]]).T
    STATE = np.array([[123, 456]]).T
    ACTUATOR_LIMITS = np.array([
        [-5, 5],
        [-5, 5]
    ])

    def _get_controller_cls(self):
        return FixedInputController

    def _get_controller_kwargs(self):
        return {
            "actuator_limits": self.ACTUATOR_LIMITS,
            "values": self.ACTION,
        }

    def test_controller_base(self):
        self._controller_base_test()

    def _test_get_action(self):
        controller = self._generate_controller()
        assert_array_equal(
            controller.get_action(self.STATE),
            self.ACTION
        )

    def _test_reset(self):
        pass
