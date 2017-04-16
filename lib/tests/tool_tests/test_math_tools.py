import numpy as np
import unittest

from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...tools.math_tools import (center_mod,
                                 hashable,
                                 normalize_quaternions,
                                 saturate, euler2quat, quat2euler)


class TestMathTools(unittest.TestCase):
    def test_center_mod(self):
        array = np.array([0.5, -0.8, 1.3])
        assert_array_almost_equal(
            center_mod(array, [False, 0.3, 1.1]),
            np.array([0.5, 0.1, 0.2]),
            decimal=8
        )
        assert_array_equal(
            center_mod(array, None),
            array
        )

    def test_saturate(self):
        state = np.array([[1, 0, -2, 5]]).T
        assert_array_almost_equal(
            saturate(
                state,
                np.array([
                    [0, 2],
                    [-1, 1],
                    [-1, 3],
                    [0, 3],
                ])
            ),
            np.array([[1, 0, -1, 3]]).T,
            decimal=8
        )
        assert_array_equal(
            saturate(state, None),
            state
        )

    def test_euler2quat(self):
        assert_array_almost_equal(
            euler2quat(
                np.array([[
                    1.622963855375272,
                    -0.147244085494920,
                    0.401364254133409,
                ]]).T
            ),
            np.array([[0.662146038842327, 0.718920448816381, 0.094568509671694, 0.189137009856730]]).T,
            decimal=8
        )

    def test_quat2euler(self):
        assert_array_almost_equal(
            quat2euler(
                np.array([[0.662146038842327, 0.718920448816381, 0.094568509671694, 0.189137009856730]]).T
            ),
            np.array([[
                1.622963855375272,
                -0.147244085494920,
                0.401364254133409,
            ]]).T,
            decimal=8
        )

    def test_normalize_quaternions(self):
        assert_array_almost_equal(
            normalize_quaternions(
                np.array([[
                    0.397739773977398,
                    0.113611361136114,
                    0.056805680568057,
                    0.431843184318432,
                ]]).T
            ),
            np.array([[
                0.662146042168,
                0.189137013805,
                0.0945685069025,
                0.718920445079,
            ]]).T,
            decimal=8
        )

    def test_hashable(self):
        non_hashable = np.array([1, 2, 3])
        self.assertRaises(
            TypeError,
            hash,
            non_hashable
        )
        hash(hashable(non_hashable))
        hash(hashable(None))
