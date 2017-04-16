import numpy as np

from ...eq_and_hash_base_test import EqAndHashBaseTest
from ....function_approximators.llr_function_approximator import Key

from numpy.testing import assert_array_equal


class TestKey(EqAndHashBaseTest):
    KEY_CONTENT_TUPLE = (0.5, -0.8, 1.3)
    KEY_CONTENT_TUPLE_OTHER = (2.1, -0.3, 1.1)
    KEY_CONTENT_ARRAY = np.array([0.5, -0.8, 1.3]).T
    KEY_CONTENT_MATRIX = np.matrix([0.5, -0.8, 1.3]).T
    KEY_CONTENT_MATRIX_T = np.matrix([0.5, -0.8, 1.3])
    MOD = [False, 0.3, 1.1]
    SCALE = np.matrix([1.2, 1., np.pi]).T
    SATURATE = np.matrix([
        [-1, 1],
        [-0.5, 0.6],
        [-10, 1.05]
    ])

    def _generate_instance(self):
        return Key(self.KEY_CONTENT_TUPLE)

    def _generate_other_instance(self):
        return Key(self.KEY_CONTENT_TUPLE_OTHER)

    def test_init(self):
        self.assertEqual(
            Key(self.KEY_CONTENT_MATRIX),
            Key(self.KEY_CONTENT_MATRIX_T)
        )
        self.assertEqual(
            Key(self.KEY_CONTENT_MATRIX),
            Key(self.KEY_CONTENT_ARRAY)
        )
        self.assertEqual(
            Key(self.KEY_CONTENT_MATRIX),
            Key(self.KEY_CONTENT_TUPLE)
        )

    def test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_instance,
            self._generate_other_instance
        )

    def test__len__(self):
        self.assertEqual(
            len(self._generate_instance()),
            3
        )

    def test__iter__(self):
        key = self._generate_instance()
        self.assertEqual(
            [i for i in key],
            list(self.KEY_CONTENT_TUPLE)
        )

    def test_get_array(self):
        key = self._generate_instance()
        assert_array_equal(
            key._array,
            key.get_array()
        )

    def test_aggregate(self):
        key = self._generate_instance()
        assert_array_equal(
            key.aggregate(),
            key._array
        )
