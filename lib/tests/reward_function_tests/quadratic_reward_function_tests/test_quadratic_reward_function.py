import numpy as np

from ..reward_function_base_test import RewardFunctionBaseTest
from ....reward_functions.quadratic_error_reward_function import QuadraticErrorRewardFunction

from numpy.testing import assert_array_almost_equal, assert_array_equal


class TestQuadraticRewardFunction(RewardFunctionBaseTest):
    ACTION_WEIGHT_LIST = [2, 1]
    STATE_WEIGHT_LIST = [5, 0.1]
    OTHER_STATE_WEIGHT_LIST = [0.5, 1]
    DESIRED_STATE = np.array([[0, 0]]).T

    STATE_MOD = [2 * np.pi, False]

    QS1 = np.array([[+np.pi, +0.]]).T
    QS2 = np.array([[+np.pi, +0.3]]).T
    QS3 = np.array([[+np.pi, -0.3]]).T
    QS4 = np.array([[-np.pi, -0.3]]).T
    QS5 = np.array([[-3.5 * np.pi, -30]]).T
    QS6 = np.array([[+1.5 * np.pi, -30]]).T
    QS7 = np.array([[+3.5 * np.pi, +30]]).T

    QA1 = np.array([[+0.5, -0.2]]).T
    QA2 = np.array([[-0.5, +0.2]]).T
    QA3 = np.array([[-0.8, +0.3]]).T

    def _get_reward_function_cls(self):
        return QuadraticErrorRewardFunction

    def _get_reward_kwargs(self):
        return {
            "action_weights": self.ACTION_WEIGHT_LIST,
            "state_weights": self.STATE_WEIGHT_LIST,
            "desired_state": self.DESIRED_STATE,
            "state_mod": self.STATE_MOD,
        }

    def _get_other_reward_kwargs(self):
        return {
            "action_weights": self.ACTION_WEIGHT_LIST,
            "state_weights": self.OTHER_STATE_WEIGHT_LIST,
            "desired_state": self.DESIRED_STATE,
            "state_mod": self.STATE_MOD,
        }

    def reward_function_base_test(self):
        self._reward_function_base_test()

    def _test_get_reward(self):
        reward_function = self._generate_reward_function()
        RA1 = -0.5 * 0.54
        RA2 = RA1
        RA3 = -0.5 * 1.37
        RS1 = -0.5 * 49.34802201
        RS2 = -0.5 * 49.35702201
        RS5 = -0.5 * 102.337005501

        assert_array_almost_equal(
            reward_function.get_reward(self.QS1, self.QA1),
            np.array([RS1 + RA1]),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS1, self.QA2),
            np.array([RS1 + RA2]),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS1, self.QA3),
            np.array([RS1 + RA3]),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS2, self.QA1),
            np.array([RS2 + RA1]),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS2, self.QA2),
            np.array([RS2 + RA2]),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS2, self.QA1),
            reward_function.get_reward(self.QS3, self.QA1),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS2, self.QA1),
            reward_function.get_reward(self.QS4, self.QA1),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS5, self.QA1),
            np.array([RS5 + RA1]),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS6, self.QA1),
            reward_function.get_reward(self.QS5, self.QA1),
            decimal=8
        )
        assert_array_almost_equal(
            reward_function.get_reward(self.QS6, self.QA1),
            reward_function.get_reward(self.QS7, self.QA1),
            decimal=8
        )

    def _test_get_derivative_to_action(self):
        delta = 1e-5
        reward_function = self._generate_reward_function()
        reward = reward_function.get_reward(
            self.QS5,
            self.QA3
        )
        reward_plus_1 = reward_function.get_reward(
            self.QS5,
            self.QA3 + np.array([[delta, 0]]).T
        )
        reward_plus_2 = reward_function.get_reward(
            self.QS5,
            self.QA3 + np.array([[0, delta]]).T
        )
        gradient = np.divide(
            np.array([[reward_plus_1, reward_plus_2]]).T - reward,
            delta
        )
        assert_array_almost_equal(
            reward_function.get_derivative_to_action(self.QA3),
            gradient,
            decimal=4
        )

    def test_quadratic_reward(self):
        reward_function = self._generate_reward_function()
        state_error = reward_function.compute_state_error(self.QS5)
        assert_array_almost_equal(
            reward_function._compute_quadratic_reward(
                state_error,
                self.QA3
            ),
            reward_function._compute_quadratic_reward(
                -state_error,
                self.QA3
            ),
            decimal=8
        )

    def test_compute_state_error(self):
        reward_function = self._generate_reward_function()
        assert_array_almost_equal(
            reward_function.compute_state_error(self.QS5),
            np.array([[-1.5707963267948966, +30.]]).T,
            decimal=8
        )

    def test_get_state_weights(self):
        reward_function = self._generate_reward_function()
        assert_array_equal(
            reward_function.get_state_weights(),
            self.STATE_WEIGHT_LIST
        )

    def test_get_action_weights(self):
        reward_function = self._generate_reward_function()
        assert_array_equal(
            reward_function.get_action_weights(),
            self.ACTION_WEIGHT_LIST
        )

    def test_get_state_mod(self):
        reward_function = self._generate_reward_function()
        self.assertEqual(
            reward_function.get_state_mod(),
            reward_function._state_mod
        )

    def test_add_weights(self):
        reward_function = self._generate_reward_function()
        reward_function.add_weights(
            np.array([-1, 0.5]),
            np.array([1.5, -3]),
        )
        assert_array_equal(
            reward_function.get_action_weights(),
            np.array([1, 1.5])
        )
        assert_array_equal(
            reward_function.get_state_weights(),
            np.array([6.5, -2.9])
        )

