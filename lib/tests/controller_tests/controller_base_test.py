import numpy as np
import unittest

from numpy.testing import assert_array_equal


class ControllerBaseTest(unittest.TestCase):
    ACTUATOR_LIMITS = None

    def _get_controller_cls(self):
        raise NotImplementedError

    def _get_controller_kwargs(self):
        raise NotImplementedError

    def _generate_controller(self):
        controller_cls = self._get_controller_cls()
        return controller_cls(**self._get_controller_kwargs())

    def _controller_base_test(self):
        self._test_get_action()
        self._test_get_zero_action()
        self._test_reset()

    def _test_get_action(self):
        raise NotImplementedError

    def _test_get_zero_action(self):
        controller = self._generate_controller()
        assert_array_equal(
            controller.get_zero_action(),
            np.zeros((self.ACTUATOR_LIMITS.shape[0], 1))
        )

    def _test_reset(self):
        raise NotImplementedError
