import unittest

from ...validation.object_validation import assert_has_attribute, assert_in, assert_not_in, assert_unique


class TestObjectValidation(unittest.TestCase):
    def test_assert_has_attribute(self):
        self.assertRaises(
            AssertionError,
            assert_has_attribute,
            [1, 2, 3],
            "some_attribute"
        )
        assert_has_attribute(
            [2, 3, 4],
            "count"
        )

    def test_assert_in(self):
        l = list(range(5))
        self.assertRaises(
            AssertionError,
            assert_in,
            6,
            l
        )
        assert_in(3, l)

    def test_assert_not_in(self):
        l = list(range(5))
        self.assertRaises(
            AssertionError,
            assert_not_in,
            3,
            l
        )
        assert_not_in(6, l)

    def test_assert_unique(self):
        self.assertRaises(
            AssertionError,
            assert_unique,
            [1., 2., 1.]
        )
        assert_unique([1., 2., 3., 4])
