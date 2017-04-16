import numpy as np

from ..eq_and_hash_base_test import EqAndHashBaseTest
from ...features import Feature, FeatureChange
from ...sets import FeatureSet


class TestFeatureChange(EqAndHashBaseTest):
    FEATURE = Feature(
        "zeta",
        scale=0.8,
        feature_type="action"
    )
    METHOD = "zero"
    OTHER_METHOD = "threshold"
    OTHER_METHOD_ARGS = 0.3

    FEATURE_SET = FeatureSet([
        Feature(
            "alpha",
            scale=0.3,
            bounds=np.array([-3, 4])
        ),
        Feature(
            "beta",
            scale=0.8
        ),
        Feature(
            "gamma",
            scale=2.3,
            feature_type="action",
            bounds=np.array([-3, -1])
        )
    ])

    def _generate_feature_change(self):
        return FeatureChange(
            self.FEATURE,
            self.METHOD,
        )

    def _generate_other_feature_change(self):
        return FeatureChange(
            self.FEATURE,
            self.OTHER_METHOD,
            spread=self.OTHER_METHOD_ARGS,
        )

    def test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_feature_change,
            self._generate_other_feature_change
        )

    def test_get_feature(self):
        feature_change = self._generate_feature_change()
        self.assertEqual(
            feature_change.get_feature(),
            feature_change._feature
        )

    def test_get_spread(self):
        feature_change = self._generate_feature_change()
        self.assertEqual(
            feature_change.get_spread(),
            feature_change._spread
        )

    def test_get_method(self):
        feature_change = self._generate_feature_change()
        self.assertEqual(
            feature_change.get_method(),
            feature_change._method
        )

    def test_is_add(self):
        feature_change = self._generate_feature_change()
        self.assertTrue(
            feature_change.is_add()
        )
        other_feature_change = self._generate_other_feature_change()
        self.assertFalse(
            other_feature_change.is_add()
        )

    def test_is_remove(self):
        feature_change = self._generate_feature_change()
        self.assertFalse(
            feature_change.is_remove()
        )
        other_feature_change = self._generate_other_feature_change()
        self.assertTrue(
            other_feature_change.is_remove()
        )

    def test_apply(self):
        feature_change = self._generate_feature_change()
        self.assertEqual(
            feature_change.apply(self.FEATURE_SET),
            FeatureSet([
                self.FEATURE_SET[0],
                self.FEATURE_SET[1],
                self.FEATURE_SET[2],
                self.FEATURE

            ])
        )
