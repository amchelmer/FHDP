import numpy as np

from ..eq_and_hash_base_test import EqAndHashBaseTest
from ...features import Feature

from numpy.testing import assert_array_equal


class TestFeature(EqAndHashBaseTest):
    NAME = "alpha"
    SCALE = 0.3
    BOUNDS = np.array([-3, 3])
    TYPE = "state"

    OTHER_NAME = "beta"
    OTHER_SCALE = 0.8
    OTHER_BOUNDS = np.array([-2, 4])
    OTHER_TYPE = "action"

    def _generate_feature(self):
        return Feature(**self._get_feature_kwargs())

    def _generate_other_feature(self):
        return Feature(**self._get_other_feature_kwargs())

    def _get_feature_kwargs(self):
        return {
            "name": self.NAME,
            "scale": self.SCALE,
            "bounds": self.BOUNDS,
            "feature_type": self.TYPE,
        }

    def _get_other_feature_kwargs(self):
        return {
            "name": self.OTHER_NAME,
            "scale": self.OTHER_SCALE,
            "bounds": self.OTHER_BOUNDS,
            "feature_type": self.OTHER_TYPE,
        }

    def test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_feature,
            self._generate_other_feature
        )

    def test_get_name(self):
        feature = self._generate_feature()
        self.assertEqual(
            feature.get_name(),
            feature._name
        )

    def test_get_scale(self):
        feature = self._generate_feature()
        self.assertEqual(
            feature.get_scale(),
            feature._scale
        )

    def test_get_type(self):
        feature = self._generate_feature()
        self.assertEqual(
            feature.get_type(),
            feature._type
        )

    def test_get_bounds(self):
        feature = self._generate_feature()
        assert_array_equal(
            feature.get_bounds(),
            feature._bounds
        )

    def test_is_state(self):
        feature = self._generate_feature()
        self.assertEqual(
            feature.is_state(),
            True
        )
        other_feature = self._generate_other_feature()
        self.assertEqual(
            other_feature.is_state(),
            False
        )

    def test_is_action(self):
        feature = self._generate_feature()
        self.assertEqual(
            feature.is_action(),
            False
        )
        other_feature = self._generate_other_feature()
        self.assertEqual(
            other_feature.is_action(),
            True
        )
