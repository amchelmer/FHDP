import numpy as np
import unittest

from ...validation.format_validation import *


class TestFormatValidation(unittest.TestCase):
    @staticmethod
    def _generate_matrix(rows, cols):
        return np.matrix(np.arange(rows * cols)).reshape(rows, cols)

    def test_assert_shape(self):
        self.assertRaises(
            AssertionError,
            assert_shape,
            self._generate_matrix(2, 2),
            (3, 3)
        )
        assert_shape(
            self._generate_matrix(3, 3),
            (3, 3)
        )

    def test_assert_shape_like(self):
        self.assertRaises(
            AssertionError,
            assert_shape_like,
            self._generate_matrix(2, 2),
            np.matrix(np.zeros((3, 3)))
        )
        assert_shape_like(
            self._generate_matrix(3, 3),
            np.matrix(np.zeros((3, 3)))
        )

    def test_assert_same_length(self):
        self.assertRaises(
            AssertionError,
            assert_same_length,
            [1, 2, 3],
            [1, 2]
        )
        assert_same_length(
            [1, 2, 3],
            [1, 2, 5],
            [0, 1, 3],
        )

    def test_assert_length(self):
        self.assertRaises(
            AssertionError,
            assert_length,
            [1, 2, 3],
            2
        )
        assert_length(
            [2, 0, 10],
            3,
        )

    def test_assert_list_of_value(self):
        self.assertRaises(
            AssertionError,
            assert_list_of_value,
            [4, 4, 4, 4, 4, 4, 4, 4, 5],
            4
        )
        self.assertRaises(
            AssertionError,
            assert_list_of_value,
            [4, 4, 4, 4, 4, 4, 4, 4, 5],
        )
        self.assertRaises(
            AssertionError,
            assert_list_of_value,
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            5
        )
        assert_list_of_value(
            10 * [5],
            5,
        )
