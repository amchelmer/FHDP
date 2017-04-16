import unittest

from ...validation.type_validation import (assert_is_subclass,
                                           assert_type_in,
                                           assert_is_type,
                                           assert_list_of_type,
                                           assert_list_of_types,
                                           assert_list_of_subclass)


class TypeValidationTest(unittest.TestCase):
    def test_assert_is_subclass(self):
        class TestCls(list):
            pass

        self.assertRaises(
            TypeError,
            assert_is_subclass,
            TestCls,
            dict
        )

    def test_assert_list_of_subclass(self):
        class TestCls(list):
            pass

        self.assertRaises(
            TypeError,
            assert_list_of_subclass,
            [TestCls, TestCls, TestCls, int],
            list
        )

    def test_assert_is_type(self):
        self.assertRaises(
            TypeError,
            assert_is_type,
            [1, 2, 3],
            dict
        )
        self.assertRaises(
            TypeError,
            assert_is_type,
            (3, 2, 1),
            list
        )

    def test_assert_type_in(self):
        self.assertRaises(
            TypeError,
            assert_type_in,
            int(8),
            [list, dict]
        )
        assert_type_in(float(3), [float, dict, list, int])

    def test_assert_list_of_type(self):
        self.assertRaises(
            TypeError,
            assert_list_of_type,
            [float(i) for i in range(3)] + [int(9)],
            float
        )
        assert_list_of_type([float(i) for i in range(3)], float)

    def test_assert_list_of_types(self):
        self.assertRaises(
            TypeError,
            assert_list_of_types,
            [float(i) for i in range(3)] + [int(9)],
            [float, dict]
        )
        assert_list_of_types([float(i) for i in range(3)], [float, int])
