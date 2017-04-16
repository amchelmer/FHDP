import numpy as np

from ..controller_base_test import ControllerBaseTest
from ....controllers.zero_input_controller import ZeroInputController

from numpy.testing import assert_array_equal


class TestZeroInputController(ControllerBaseTest):
    ACTUATOR_LIMITS = np.array([
        [-5, 5],
        [-5, 5],
    ])

    def _get_controller_cls(self):
        return ZeroInputController

    def _get_controller_kwargs(self):
        return {
            "actuator_limits": self.ACTUATOR_LIMITS
        }

    def controller_base_test(self):
        self._test_get_action()
        self._test_get_zero_action()

    def _test_get_action(self):
        controller = self._generate_controller()
        state = None
        assert_array_equal(
            controller.get_action(state),
            controller.get_zero_action()
        )

    def _test_reset(self):
        pass
