import numpy as np

from ..controller_base_test import ControllerBaseTest
from ....controllers.pid_controller import PIDController
from numpy.testing import assert_array_almost_equal, assert_array_equal


class TestPIDController(ControllerBaseTest):
    REFSTATE = np.array([[-6.3, 1.2, 3]]).T
    FREQUENCY = 100
    ACTUATOR_LIMITS = np.array([[-2, 2], [-8, 8]])
    P_GAIN = np.array([[3, 4, 5], [1, 2, 3]])
    I_GAIN = 0.1 * np.array([[3, 4, 5], [1, 2, 3]])
    D_GAIN = 0.1 * np.array([[4, 5, 6], [0, 1, 0]])
    LAST_ERROR = np.array([[-0.03, .008, 1.2]]).T
    ACC_ERROR = np.array([[-1.3, .98, 10.21]]).T
    STATE = np.array([[-6.2, 1.05, 2]]).T

    def _get_controller_cls(self):
        return PIDController

    def _get_controller_kwargs(self):
        return {
            "ref_state": self.REFSTATE,
            "frequency": self.FREQUENCY,
            "actuator_limits": self.ACTUATOR_LIMITS,
            "p_gain": self.P_GAIN,
            "i_gain": self.I_GAIN,
            "d_gain": self.D_GAIN,
        }

    def test_controller_base(self):
        self._controller_base_test()

    def _test_reset(self):
        controller = self._generate_controller()
        controller._set_last_error(self.LAST_ERROR)
        controller._accumulate_error(self.ACC_ERROR)
        controller.reset()
        assert_array_equal(
            controller._last_error,
            np.zeros((len(self.STATE), 1))
        )
        assert_array_equal(
            controller._acc_error,
            np.zeros((len(self.STATE), 1))
        )

    def _test_get_action(self):
        controller = self._generate_controller()
        controller._set_last_error(self.LAST_ERROR)
        controller._accumulate_error(self.ACC_ERROR)
        action = controller.get_action(self.STATE)
        assert_array_almost_equal(
            action,
            np.array([[2.0, 7.7490]]).T,
            decimal=8
        )

    def test_set_last_error(self):
        controller = self._generate_controller()
        assert_array_equal(
            controller._last_error,
            np.zeros((len(self.STATE), 1))
        )
        controller._set_last_error(self.LAST_ERROR)
        assert_array_almost_equal(
            controller._last_error,
            self.LAST_ERROR
        )

    def test_accumulate_error(self):
        controller = self._generate_controller()
        assert_array_equal(
            controller._acc_error,
            np.zeros((len(self.STATE), 1))
        )
        controller._accumulate_error(self.ACC_ERROR)
        assert_array_almost_equal(
            controller._acc_error,
            self.ACC_ERROR,
            decimal=8
        )

    def test_assert_gains(self):
        controller = self._generate_controller()
        self.assertRaises(
            AssertionError,
            controller._assert_gains,
            self.P_GAIN,
            self.I_GAIN,
            self.D_GAIN[:, 1]
        )
        gains = [self.P_GAIN, self.I_GAIN, self.D_GAIN, None]
        for gain_1, gain2 in zip(
                [self.P_GAIN, self.I_GAIN, self.D_GAIN, np.zeros((2, 3))],
                controller._assert_gains(*gains)
        ):
            assert_array_equal(gain_1, gain2)

    def test_compute_error_signal(self):
        controller = self._generate_controller()
        assert_array_almost_equal(
            controller.compute_error_signal(self.STATE),
            np.array([[-0.1, 0.15, 1.]]).T
        )
