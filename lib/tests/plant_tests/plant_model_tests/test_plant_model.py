import numpy as np

from ....features import Feature, FeatureChange
from ....function_approximators.llr_function_approximator import Key, KeySet
from ....plants.plant_models import PlantModel
from ....sets import FeatureSet
from ...function_approximator_tests.llr_function_approximator_tests.llr_function_approximator_base_test import (
    LLRFunctionApproximatorBaseTest
)

from numpy.testing import assert_array_almost_equal


class TestPlantModel(LLRFunctionApproximatorBaseTest):
    PLANT_FEATURE_SET = FeatureSet([
        Feature("x1"),
        Feature("x1_dot", scale=4., derivative=True, bounds=np.array([-2, 2])),
        Feature("x2", scale=2.),
        Feature("x2_dot", scale=0.4, derivative=True),
        Feature("u1", scale=0.5, feature_type="action", bounds=np.array([-2, 2])),
        Feature("u2", scale=2., feature_type="action", bounds=np.array([-3, 3])),
        Feature("u3", scale=1.1, feature_type="action", bounds=np.array([-3, 3]))
    ])
    FEATURE_SET = FeatureSet([
        PLANT_FEATURE_SET[0],
        PLANT_FEATURE_SET[1],
        PLANT_FEATURE_SET[4],
        PLANT_FEATURE_SET[5],
    ])
    KNN = 3
    MAX_MEMORY = 5
    PREDICTION_EPSILON = 1e-3

    QUERY_POINT = np.array([[-0.3, 0.5, 1.1, 0.13, 2.2, -2.4, 0.01]]).T
    QUERY_FEATURES = np.array([[0.5, 2.2, -2.4]]).T
    BETA = np.array([
        [0.51955136, -0.71386679, -0.04463633, -0.26854562],
        [-0.2759945, -0.24647305, 1.02514225, 0.03240115]
    ])
    PREDICTION = np.array([[-1.47214969, -2.]]).T
    NEXT_STATE = QUERY_POINT[:4, :] + np.array([[-1.47214969, -2., 0, 0]]).T
    TARGET = NEXT_STATE + 0.05

    STATES = [
        np.array([[-4.23, 0.42, 0.1, 2.3, -2.86, -1.85, 0.23]]).T,
        np.array([[0.5, 6.15, -0.1, 2.1, 0.82, 2.27, -0.22]]).T,
        np.array([[-0.71, 5.46, 1.97, 9.6, 4.19, -4.87, 0.49]]).T,
        np.array([[-6.39, -1.18, 1.23, -0.98, 1.24, 3.88, 0.13]]).T,
        np.array([[1.82, 0.46, 0.1, 0.02, 3.23, -2.36, -1.09]]).T,
    ]

    FEATURES = [
        np.array([[0.42, 6.15, 5.46, -1.18, 0.46]]).T,
        np.array([[-2.86, 0.82, 4.19, 1.24, 3.23]]).T,
        np.array([[-1.85, 2.27, -4.87, 3.88, -2.36]]).T,
    ]

    KEYS = KeySet([
        Key([0.105, -5.72, -0.925]),
        Key([1.5375, 1.64, 1.135]),
        Key([1.365, 8.38, -2.435]),
        Key([-0.295, 2.48, 1.94]),
        Key([0.115, 6.46, -1.18]),
    ])
    SORTED_KEYS = KeySet([
        Key([-0.295, 2.48, 1.94]),
        Key([0.105, -5.72, -0.925]),
        Key([0.115, 6.46, -1.18]),
        Key([1.365, 8.38, -2.435]),
        Key([1.5375, 1.64, 1.135]),
    ])
    VALUES = [
        np.array([[-1.83, -2.29]]).T,
        np.array([[2.24, 0.46]]).T,
        np.array([[2.56, -2.12]]).T,
        np.array([[-1.94, 4.03]]).T,
        np.array([[-2.23, -3.31]]).T,
    ]
    SORTED_VALUES = [
        np.array([[-1.94, 4.03]]).T,
        np.array([[-1.83, -2.29]]).T,
        np.array([[-2.23, -3.31]]).T,
        np.array([[2.56, -2.12]]).T,
        np.array([[2.24, 0.46]]).T,
    ]
    DISTANCES = np.array([10.123755, 3.881363, 4.347784, 3.704376, 2.060121])
    NEAREST_NEIGHBOR_INDICES = [4, 3, 1]

    NUMBER_OF_SAMPLES_IN_MEMORY = 5
    CONTAINED_KEY = KEYS[0]
    KEYS_SORTED_BY_AGE = KEYS[::-1]
    CUT_OFF_AGE = 0.35
    KEYS_AFTER_CUTOFF_AGE = KEYS[-3:]
    MEMORY_AFTER_CUTOFF_AGE = {k: v for (k, v) in zip(KEYS_AFTER_CUTOFF_AGE, VALUES[-3:])}

    def _get_function_approximator_cls(self):
        return PlantModel

    def _get_function_approximator_kwargs(self):
        return {
            "plant_feature_set": self.PLANT_FEATURE_SET,
            "feature_set": self.FEATURE_SET,
            "knn": self.KNN,
            "max_memory": self.MAX_MEMORY,
            "prediction_epsilon": self.PREDICTION_EPSILON,
        }

    def lwlr_function_approximator_base_test(self):
        self._lwlr_function_approximator_base_test()

    def _test_increment(self):
        pass

    def _test_set_epsilon(self):
        plant_model = self._generate_function_approximator()
        e = 0.0013
        plant_model._set_epsilon(e)
        self.assertEqual(plant_model._epsilon, e ** 0.5)

    def _test_map_zero_initialization(self):
        pass

    def _test_map_perturb(self):
        pass

    def _test_map_sample_cloning_uniform(self):
        pass

    def _test_map_sample_cloning_gauss(self):
        pass

    def _test_map_project(self):
        pass

    def _test_map_threshold(self):
        pass

    def test_merge_state_action(self):
        plant_model_cls = self._get_function_approximator_cls()
        assert_array_almost_equal(
            plant_model_cls.merge_state_action(
                self.QUERY_POINT[:3],
                self.QUERY_POINT[3:],
            ),
            self.QUERY_POINT
        )

    def test_get_next_state(self):
        plant_model = self._generate_function_approximator()
        next_state, knn_keys, dxprime_du = plant_model.get_next_state(
            self.QUERY_POINT[:4],
            self.QUERY_POINT[4:],
        )
        assert_array_almost_equal(
            next_state,
            self.NEXT_STATE,
            decimal=8
        )
        self.assertEqual(
            knn_keys,
            KeySet([
                Key([0.11500, 6.46000, -1.18000]),
                Key([-0.29500, 2.48000, 1.94000]),
                Key([1.53750, 1.64000, 1.13500])
            ])
        )
        assert_array_almost_equal(dxprime_du, self.BETA[:, 1:3])

    def test_update(self):
        plant_model = self._generate_empty_function_approximator()

        query_state = self.QUERY_POINT[:4]
        query_action = self.QUERY_POINT[4:]

        query_features = plant_model.like_me(self.QUERY_POINT)

        plant_model.update(query_state, query_action, self.NEXT_STATE, self.TARGET)
        assert_array_almost_equal(
            plant_model.tree._tree.query(plant_model.make_keys(query_features).get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )
        self.assertTrue(plant_model.make_keys(query_features) in plant_model.value_memory)

        plant_model = self._generate_function_approximator()
        plant_model.update(query_state, query_action, self.NEXT_STATE, self.NEXT_STATE)
        self.assertFalse(plant_model.make_keys(query_features) in plant_model.value_memory)

        plant_model.update(query_state, query_action, self.NEXT_STATE - 0.1, self.TARGET)
        assert_array_almost_equal(
            plant_model.tree._tree.query(plant_model.make_keys(query_features).get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )
        self.assertTrue(plant_model.make_keys(query_features) in plant_model.value_memory)

    def test_compute_predicion_error(self):
        plant_model_cls = self._get_function_approximator_cls()
        prediction_values = np.array([[1.2, -0.3, 0.8]]).T
        target_values = np.array([[-1, 0.04, -1.1]]).T
        error, norm = plant_model_cls._compute_prediction_error(
            prediction_values,
            target_values,
        )
        assert_array_almost_equal(
            error,
            np.array([[2.2, -0.34, 1.9]]).T,
            decimal=8
        )
        self.assertAlmostEqual(
            norm,
            2.9267046314925595
        )

    def _test_change_input_feature(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.change_input_feature(
            FeatureChange(self.PLANT_FEATURE_SET[6], "zero")
        )
        self.assertEqual(
            function_approximator.value_memory,
            {}
        )

        function_approximator = self._generate_function_approximator()
        function_approximator.change_input_feature(
            FeatureChange(self.PLANT_FEATURE_SET[2], "zero")
        )
        self.assertEqual(
            function_approximator.value_memory,
            {}
        )

        function_approximator = self._generate_function_approximator()
        function_approximator.change_input_feature(
            FeatureChange(self.PLANT_FEATURE_SET[4], "project")
        )
        self.assertEqual(
            function_approximator.value_memory,
            {}
        )

        function_approximator = self._generate_function_approximator()
        function_approximator.change_input_feature(
            FeatureChange(self.PLANT_FEATURE_SET[0], "project")
        )
        self.assertEqual(
            function_approximator.value_memory,
            {}
        )

    def test_one_step_ahead_simulation(self):
        pass

    def test_one_step_ahead_errors(self):
        pass

    def test_all_step_ahead_simulation(self):
        pass

    def test_all_step_ahead_errors(self):
        pass
