import numpy as np

from ..set_base_test import SetBaseTest
from ....sets import FeatureSet
from ....features import Feature

from numpy.testing import assert_array_equal, assert_array_almost_equal


class TestFeatureSet(SetBaseTest):
    FEATURE_1 = Feature(
        "alpha",
        scale=0.3,
        bounds=np.array([-3, 4])
    )
    FEATURE_2 = Feature(
        "beta",
        scale=0.8
    )
    FEATURE_3 = Feature(
        "gamma",
        scale=2.3,
        feature_type="action",
        bounds=np.array([-3, -1])
    )
    FEATURE_4 = Feature(
        "kappa",
        scale=2.0
    )
    FEATURE_5 = Feature(
        "iota",
        scale=8.3,
        feature_type="action"
    )
    ARRAY = np.array([[1.1, -0.2, 2.1]]).T
    STATE = np.array([[1.1, -0.2, 2.1, 0.2]]).T

    OBJECT_IN_SET = FEATURE_1
    OBJECT_NOT_IN_SET = FEATURE_5

    def _get_set_cls(self):
        return FeatureSet

    def _get_set_kwargs(self):
        return {"feature_list": [self.FEATURE_1, self.FEATURE_2, self.FEATURE_3]}

    def _get_other_set_kwargs(self):
        return {"feature_list": [self.FEATURE_2, self.FEATURE_4, self.FEATURE_3]}

    def set_base_test(self):
        self._set_base_test()

    def test_init(self):
        feature_set = self._generate_set_instance()
        assert_array_equal(
            feature_set._scales,
            np.array([[
                self.FEATURE_1._scale,
                self.FEATURE_2._scale,
                self.FEATURE_3._scale,
            ]]).T
        )

    def test_validate_list(self):
        self.assertRaises(
            AssertionError,
            FeatureSet._validate_list,
            [self.FEATURE_3, self.FEATURE_2]
        )
        self.assertRaises(
            TypeError,
            FeatureSet._validate_list,
            [self.FEATURE_3, np.array([1, 2, 3])]
        )
        self.assertRaises(
            AssertionError,
            FeatureSet._validate_list,
            [self.FEATURE_3, self.FEATURE_2]
        )

    def test_get_index(self):
        feature_set = self._generate_set_instance()
        self.assertEqual(
            feature_set.get_index(self.FEATURE_2),
            1
        )

    def test_get_scales(self):
        feature_set = self._generate_set_instance()
        assert_array_equal(
            feature_set.get_scales(),
            feature_set._scales
        )

    def test_get_names(self):
        feature_set = self._generate_set_instance()
        self.assertEqual(
            feature_set.get_names(),
            ["alpha", "beta", "gamma"]
        )

    def test_get_bounds(self):
        feature_set = self._generate_set_instance()
        assert_array_equal(
            feature_set.get_bounds(),
            np.array([
                [-3, 4],
                [-np.inf, np.inf],
                [-3, -1]
            ])
        )

    def test_get_state_set(self):
        feature_set = self._generate_set_instance()
        self.assertEqual(
            feature_set.get_state_set(),
            FeatureSet([self.FEATURE_1, self.FEATURE_2])
        )

    def test_get_action_set(self):
        feature_set = self._generate_set_instance()
        self.assertEqual(
            feature_set.get_action_set(),
            FeatureSet([self.FEATURE_3])
        )

    def test_normalize(self):
        feature_set = self._generate_set_instance()
        assert_array_almost_equal(
            feature_set.normalize(self.ARRAY),
            self.ARRAY / feature_set.get_scales()
        )

    def test_unnormalize(self):
        feature_set = self._generate_set_instance()
        assert_array_almost_equal(
            feature_set.unnormalize(self.ARRAY),
            self.ARRAY * feature_set.get_scales()
        )

    def test_get_insert_position(self):
        feature_set = self._generate_set_instance()
        self.assertEqual(
            feature_set.get_insert_position(self.FEATURE_4),
            2
        )
        self.assertEqual(
            feature_set.get_insert_position(self.FEATURE_5),
            3
        )

    def test_like_me(self):
        feature_set = self._generate_set_instance()
        other_feature_set = self._generate_other_set_instance()
        assert_array_equal(
            feature_set.like_me(self.ARRAY, other_feature_set),
            np.array([[
                0, self.ARRAY.flatten()[0], self.ARRAY.flatten()[2]
            ]]).T
        )

    def test_copy(self):
        feature_set = self._generate_set_instance()
        self.assertEqual(
            feature_set,
            feature_set.copy()
        )
