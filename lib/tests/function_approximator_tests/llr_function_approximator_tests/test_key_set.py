import numpy as np

from ...eq_and_hash_base_test import EqAndHashBaseTest
from ....function_approximators.llr_function_approximator import Key, KeySet

from numpy.testing import assert_array_almost_equal


class TestKeySet(EqAndHashBaseTest):
    KEY_1 = Key((0.5, -0.8, 1.3))
    KEY_2 = Key((2.1, -0.3, 1.1))
    KEY_3 = Key((1.2, -2.2, -4.1))
    KEY_4 = Key((-2.1, -1.3, 2.9))
    KEY_5 = Key((-2.1, -1.3, 2.9, 0.4))
    KEY_LIST = [KEY_1, KEY_2, KEY_3, KEY_4]
    OTHER_KEY_LIST = [KEY_1, KEY_4, KEY_3]
    MOD = [False, 0.3, 1.1]
    SCALE = np.matrix([1.2, 1., np.pi]).T
    SATURATE = np.matrix([
        [-1, 1],
        [-0.5, 0.6],
        [-10, 1.05]
    ])

    def _generate_instance(self):
        return KeySet(self.KEY_LIST)

    def _generate_other_instance(self):
        return KeySet(self.OTHER_KEY_LIST)

    def test_init(self):
        self.assertRaises(
            AssertionError,
            KeySet,
            self.KEY_LIST + [self.KEY_5]
        )

    def test__len__(self):
        key_set = self._generate_instance()
        self.assertEqual(
            len(key_set),
            4
        )

    def test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_instance,
            self._generate_other_instance
        )

    def test__getitem__(self):
        key_set = self._generate_instance()
        self.assertEqual(
            key_set[1:3],
            KeySet([self.KEY_2, self.KEY_3])
        )

    def test_aggregate(self):
        key_set = self._generate_instance()
        assert_array_almost_equal(
            key_set.aggregate(),
            np.matrix((
                (0.5, -0.8, 1.3),
                (2.1, -0.3, 1.1),
                (1.2, -2.2, -4.1),
                (-2.1, -1.3, 2.9)
            )).T,
            decimal=10
        )
