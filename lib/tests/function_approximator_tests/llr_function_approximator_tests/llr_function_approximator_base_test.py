import numpy as np
import tempfile

from ..function_approximator_base_test import FunctionApproximatorBaseTest
from ....features import FeatureChange
from ....function_approximators.llr_function_approximator import KeySet

from numpy.testing import assert_array_equal, assert_array_almost_equal


class LLRFunctionApproximatorBaseTest(FunctionApproximatorBaseTest):
    SEED = 1803

    KNN = None
    INPUT_FEATURE_SET = None
    OUTPUT_FEATURE_SET = None
    PLANT_FEATURE_SET = None
    MAX_MEMORY = None
    TRACE = None
    SATURATION = None
    DISCOUNT = None
    EPSILON_P_FEATURE = 1e-1
    DT = 0.1

    QUERY_POINT = None
    QUERY_FEATURES = None
    BETA = None
    PREDICTION = None

    STATES = None
    KEYS = None
    SORTED_KEYS = None
    VALUES = None
    SORTED_VALUES = None
    DISTANCES = []
    NEAREST_NEIGHBOR_INDICES = []
    NUMBER_OF_SAMPLES_IN_MEMORY = None
    CONTAINED_KEY = None
    KEYS_SORTED_BY_AGE = None
    CUT_OFF_AGE = None
    KEYS_AFTER_CUTOFF_AGE = None
    MEMORY_AFTER_CUTOFF_AGE = None

    METRIC = "mean"

    NEW_INPUT_FEATURE = None
    INSERT_INDEX = None
    POST_LEARNING_FEATURE_SET = None
    SPREAD = None
    N_CLONES = None
    MAPPED_KEYS_ZERO_INITIALIZATION = None
    MAPPED_KEYS_PERTURB = None
    MAPPED_KEYS_CLONE_UNIFORM = None
    MAPPED_KEYS_CLONE_GAUSSIAN = None

    FORGET_INPUT_FEATURE = None
    REMOVE_INDEX = None
    POST_FORGETTING_FEATURE_SET = None
    THRESHOLD = None
    MAPPED_KEYS_PROJECT = None
    MAPPED_KEYS_THRESHOLDED_PROJECTION = None
    MAPPED_VALUES_THRESHOLDED_PROJECTION = None
    MAPPED_AGES_THRESHOLDED_PROJECTION = None

    def _get_function_approximator_cls(self):
        raise NotImplementedError

    def _get_function_approximator_kwargs(self):
        raise NotImplementedError

    def _generate_empty_function_approximator(self):
        return self._get_function_approximator_cls()(**self._get_function_approximator_kwargs())

    @staticmethod
    def _add_datapoints(function_approximator, states, values, dt):
        for state, value in zip(states, values):
            feature = function_approximator.like_me(state)
            function_approximator.add_sample(feature, value)
            function_approximator.age_samples(dt)

    def _generate_function_approximator(self):
        function_approximator = self._generate_empty_function_approximator()
        self._add_datapoints(function_approximator, self.STATES, self.VALUES, self.DT)
        return function_approximator

    def _generate_other_function_approximator(self):
        function_approximator = self._generate_empty_function_approximator()
        self._add_datapoints(function_approximator, self.STATES[:4], self.VALUES[:4], self.DT)
        return function_approximator

    def assert_equal_memory(self, m1, m2):
        self.assertEqual(len(m1), len(m2))
        for k, v in m1.items():
            self.assertIn(k, m2)
            assert_array_almost_equal(v, m2[k], decimal=12)
        for k, v in m2.items():
            self.assertIn(k, m1)
            assert_array_almost_equal(v, m1[k], decimal=12)

    def _lwlr_function_approximator_base_test(self):
        self._function_approximator_base_test()
        self._test__len__()
        self._test__iter__()
        self._test__contains__()
        self._test_set_rng()
        self._test_training()
        self._test_evaluation()
        self._test_get_feature_set()
        self._test_get_output_feature_set()
        self._test_get_zero_value()
        self._test_set_max_memory_size()
        self._test_set_epsilon()
        self._test_append_bias()
        self._test_get_closest_points()
        self._test_compute_distances()
        self._test_local_fit()
        self._test_predict()
        self._test_add_sample()
        self._test_increment()
        self._test_lookup_values()
        self._test_age_samples()
        self._test_reset_age()
        self._test_get_keys_sorted_by_age()
        self._test_purge_keys()
        self._test_purge_by_age()
        self._test_purge_randomly()
        self._test_purge_by_weighted_age()
        self._test_purge_older_than()
        self._test_assert_consistent_memory()
        self._test_make_keys()
        self._test_unmake_keys()
        self._test_dump_and_load()
        self._test_rebuild_tree()
        self._test_like_me()
        self._test_change_input_feature()
        self._test_map_zero_initialization()
        self._test_map_perturb()
        self._test_map_sample_cloning_uniform()
        self._test_map_sample_cloning_gauss()
        self._test_map_project()
        self._test_map_threshold()
        self._test_assert_empty_memory()

    def _test__len__(self):
        function_approximator = self._generate_function_approximator()
        self.assertEqual(
            len(function_approximator),
            self.NUMBER_OF_SAMPLES_IN_MEMORY
        )

    def _test__iter__(self):
        function_approximator = self._generate_function_approximator()
        sorted_items = sorted(function_approximator.value_memory.items(), key=lambda t: tuple(t[0]))
        for (key_1, value_1), (key_2, value_2) in zip(function_approximator, sorted_items):
            self.assertEqual(key_1, key_2)
            assert_array_equal(value_1, value_2)

    def _test__contains__(self):
        function_approximator = self._generate_function_approximator()
        self.assertTrue(
            self.CONTAINED_KEY in function_approximator
        )

    def _test_set_rng(self):
        function_approximator = self._generate_function_approximator()

        function_approximator.set_rng(101)
        random_1 = function_approximator._rng.randn(10)
        function_approximator.set_rng(101)
        random_2 = function_approximator._rng.randn(10)
        assert_array_equal(random_1, random_2)

    def _test_training(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.training()
        self.assertTrue(function_approximator._training_mode_flag)

    def _test_evaluation(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.evaluation()
        self.assertFalse(function_approximator._training_mode_flag)

    def _test_get_feature_set(self):
        function_approximator = self._generate_function_approximator()
        self.assertEqual(
            function_approximator.get_feature_set(),
            function_approximator._input_feature_set
        )

    def _test_append_bias(self):
        function_approximator_cls = self._get_function_approximator_cls()
        assert_array_equal(
            function_approximator_cls.append_bias(np.arange(8).reshape(2, 4)),
            np.array([
                [0, 1, 2, 3],
                [4, 5, 6, 7],
                [1, 1, 1, 1],
            ])
        )

    def _test_get_output_feature_set(self):
        function_approximator = self._generate_function_approximator()
        self.assertEqual(
            function_approximator.get_output_feature_set(),
            function_approximator._output_feature_set
        )

    def _test_get_zero_value(self):
        function_approximator = self._generate_function_approximator()
        assert_array_almost_equal(
            function_approximator.get_zero_value(),
            np.zeros_like(self.VALUES[0]),
            decimal=8
        )

    def _test_set_max_memory_size(self):
        function_approximator = self._generate_function_approximator()
        new_size = 1010123
        function_approximator.set_max_memory_size(new_size)
        self.assertEqual(function_approximator._max_memory, new_size)

    def _test_set_epsilon(self):
        function_approximator = self._generate_function_approximator()
        function_approximator._set_epsilon(0.0031)
        self.assertEqual(
            function_approximator._epsilon,
            np.sqrt(0.0031 ** 2 * len(self.INPUT_FEATURE_SET))
        )

    def _test_get_closest_points(self):
        function_approximator = self._generate_function_approximator()
        keys1, values1 = function_approximator.get_closest_points(self.QUERY_FEATURES)
        keys2, values2 = (
            KeySet([self.KEYS[k] for k in self.NEAREST_NEIGHBOR_INDICES]),
            np.hstack([self.VALUES[k] for k in self.NEAREST_NEIGHBOR_INDICES]),
        )
        self.assertEqual(keys1, keys2)
        assert_array_almost_equal(
            values1,
            values2,
            decimal=8
        )

    def _test_compute_distances(self):
        function_approximator = self._generate_function_approximator()
        assert_array_almost_equal(
            function_approximator.compute_distances(self.QUERY_FEATURES, self.KEYS),
            self.DISTANCES
        )
        assert_array_almost_equal(
            function_approximator.compute_distances(self.QUERY_FEATURES),
            [self.DISTANCES[k] for k in self.NEAREST_NEIGHBOR_INDICES]
        )

    def _test_local_fit(self):
        function_approximator = self._generate_function_approximator()
        knn_keys, knn_targets = function_approximator.get_closest_points(self.QUERY_FEATURES)
        assert_array_almost_equal(
            function_approximator.local_fit(
                function_approximator.unmake_keys(knn_keys),
                knn_targets
            ),
            self.BETA,
            decimal=8,
        )

    def _test_predict(self):
        function_approximator = self._generate_function_approximator()
        assert_array_almost_equal(
            function_approximator.predict(
                self.BETA,
                self.QUERY_FEATURES
            ),
            self.PREDICTION,
            decimal=8
        )

    def _test_add_sample(self):
        function_approximator = self._generate_empty_function_approximator()
        self.assert_equal_memory(
            function_approximator.value_memory,
            {}
        )
        self.assert_equal_memory(
            function_approximator.age_memory,
            {}
        )
        feature = function_approximator.like_me(self.STATES[0])
        function_approximator.add_sample(feature, self.VALUES[0])
        self.assert_equal_memory(
            function_approximator.value_memory,
            {self.KEYS[0]: self.VALUES[0]}
        )
        self.assert_equal_memory(
            function_approximator.age_memory,
            {self.KEYS[0]: 0.}
        )
        assert_array_almost_equal(
            function_approximator.tree._tree.query(self.KEYS[0].get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )

    def _test_increment(self):
        raise NotImplementedError

    def _test_lookup_values(self):
        function_approximator = self._generate_function_approximator()
        assert_array_almost_equal(
            function_approximator.lookup_values(
                KeySet([
                    self.KEYS[0],
                    self.KEYS[2],
                    self.KEYS[1]
                ])
            ),
            np.concatenate([self.VALUES[0], self.VALUES[2], self.VALUES[1]], axis=1),
            decimal=8
        )

    def _test_age_samples(self):
        function_approximator = self._generate_empty_function_approximator()
        features = function_approximator.like_me(self.STATES[0])
        key = self.KEYS[0]
        function_approximator.add_sample(features, self.VALUES[0])
        self.assertEqual(function_approximator.age_memory[key], 0)

        function_approximator.age_samples(self.DT)
        function_approximator.age_samples(self.DT)
        self.assertEqual(function_approximator.age_memory[key], 2 * self.DT)

    def _test_reset_age(self):
        function_approximator = self._generate_function_approximator()
        key = self.KEYS[1]
        self.assertEqual(function_approximator.age_memory[key], (5 - 1) * self.DT)
        function_approximator.reset_age(key)
        self.assertEqual(function_approximator.age_memory[key], 0)

    def _test_get_keys_sorted_by_age(self):
        function_approximator = self._generate_function_approximator()
        sorted_keys, ages = function_approximator._get_keys_sorted_by_age()
        self.assertEqual(
            KeySet(sorted_keys),
            self.KEYS_SORTED_BY_AGE
        )
        assert_array_almost_equal(
            ages,
            [0.1, 0.2, 0.3, 0.4, 0.5],
            decimal=12
        )

    def _test_purge_keys(self):
        function_approximator = self._generate_function_approximator()
        function_approximator._purge_keys(self.KEYS[:3])
        self.assert_equal_memory(
            function_approximator.value_memory,
            {self.KEYS[3]: self.VALUES[3], self.KEYS[4]: self.VALUES[4]}
        )

    def _test_purge_by_age(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.add_sample(self.QUERY_FEATURES, self.PREDICTION)
        value_memory = function_approximator.value_memory.copy()

        function_approximator.purge_by_age()
        function_approximator.rebuild_tree()
        del value_memory[self.KEYS[0]]

        self.assert_equal_memory(
            function_approximator.value_memory,
            value_memory
        )
        function_approximator.assert_consistent_memory()

    def _test_purge_randomly(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.add_sample(self.QUERY_FEATURES, self.PREDICTION)
        function_approximator.add_sample(2 * self.QUERY_FEATURES, self.PREDICTION)
        function_approximator.add_sample(3 * self.QUERY_FEATURES, self.PREDICTION)
        #
        # value_memory = function_approximator.value_memory.copy()
        # keys = sorted(function_approximator.age_memory.keys())
        #
        # np.random.seed(self.SEED)
        function_approximator.purge_randomly()
        function_approximator.rebuild_tree()
        # del value_memory[keys[0]], value_memory[keys[1]], value_memory[keys[6]]
        #
        # self.assert_equal_memory(
        #     function_approximator.value_memory,
        #     value_memory
        # )
        function_approximator.assert_consistent_memory()
        pass

    def _test_purge_by_weighted_age(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.add_sample(self.QUERY_FEATURES, self.PREDICTION)
        function_approximator.add_sample(2 * self.QUERY_FEATURES, self.PREDICTION)
        function_approximator.add_sample(3 * self.QUERY_FEATURES, self.PREDICTION)
        #
        # value_memory = function_approximator.value_memory.copy()
        # keys = sorted(function_approximator.age_memory.keys())
        #
        # np.random.seed(self.SEED)
        function_approximator.purge_by_weighted_age()
        function_approximator.rebuild_tree()
        #
        # del value_memory[keys[3]], value_memory[keys[7]], value_memory[keys[4]]
        #
        # self.assert_equal_memory(
        #     function_approximator.value_memory,
        #     value_memory
        # )
        function_approximator.assert_consistent_memory()

    def _test_purge_older_than(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.purge_older_than(self.CUT_OFF_AGE)

        self.assertDictEqual(
            function_approximator.value_memory,
            self.MEMORY_AFTER_CUTOFF_AGE
        )

    def _test_assert_consistent_memory(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.assert_consistent_memory()

        del function_approximator.value_memory[self.KEYS[0]]
        self.assertRaises(
            AssertionError,
            function_approximator.assert_consistent_memory
        )

    def _test_make_keys(self):
        function_approximator = self._generate_empty_function_approximator()
        features = function_approximator.like_me(self.STATES[0])
        assert_array_almost_equal(
            function_approximator.make_keys(features).get_array(),
            self.KEYS[0].get_array(),
            decimal=8
        )
        for state, key in zip(self.STATES, self.KEYS):
            feature = function_approximator.like_me(state)
            assert_array_almost_equal(
                function_approximator.make_keys(feature).get_array(),
                key.get_array(),
            )

    def _test_unmake_keys(self):
        function_approximator = self._generate_empty_function_approximator()
        feature = function_approximator.like_me(self.STATES[0])
        assert_array_almost_equal(
            feature,
            function_approximator.unmake_keys(self.KEYS[0]),
            decimal=8
        )
        for state, k2 in zip(self.STATES, self.KEYS):
            feature = function_approximator.like_me(state)
            assert_array_almost_equal(
                feature,
                function_approximator.unmake_keys(k2),
                decimal=8
            )

    def _test_dump_and_load(self):
        file_handle = tempfile.TemporaryFile()
        function_approximator = self._generate_function_approximator()
        function_approximator.dump(file_handle)

        file_handle.seek(0)
        function_approximator_loaded = self._get_function_approximator_cls().load(file_handle)
        self.assertEqual(
            function_approximator,
            function_approximator_loaded
        )

    def _test_rebuild_tree(self):
        function_approximator = self._generate_function_approximator()
        del function_approximator.value_memory[self.KEYS[4]]
        function_approximator.rebuild_tree()

        other_function_approximator = self._generate_empty_function_approximator()
        self._add_datapoints(other_function_approximator, self.STATES[:4], self.VALUES[:4], self.DT)
        other_function_approximator.rebuild_tree()

    def _test_like_me(self):
        function_approximator = self._generate_function_approximator()
        assert_array_almost_equal(
            function_approximator.like_me(self.QUERY_POINT),
            self.QUERY_FEATURES,
            decimal=8
        )

    def _test_change_input_feature(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.change_input_feature(
            FeatureChange(self.NEW_INPUT_FEATURE, "zero")
        )
        self.assert_equal_memory(
            function_approximator.value_memory,
            {k: v for (k, v) in zip(self.MAPPED_KEYS_ZERO_INITIALIZATION, self.SORTED_VALUES)}
        )
        self.assertEqual(
            function_approximator._input_feature_set,
            self.POST_LEARNING_FEATURE_SET
        )
        function_approximator.assert_consistent_memory()

        function_approximator = self._generate_function_approximator()
        function_approximator.set_rng(self.SEED)
        function_approximator.change_input_feature(
            FeatureChange(self.NEW_INPUT_FEATURE, "perturb-gauss", self.SPREAD)
        )
        self.assert_equal_memory(
            function_approximator.value_memory,
            {k: v for (k, v) in zip(self.MAPPED_KEYS_PERTURB, self.SORTED_VALUES)}
        )
        self.assertEqual(
            function_approximator._input_feature_set,
            self.POST_LEARNING_FEATURE_SET
        )
        function_approximator.assert_consistent_memory()

        function_approximator = self._generate_function_approximator()
        function_approximator.set_max_memory_size(10)
        function_approximator.set_rng(self.SEED)
        function_approximator.change_input_feature(
            FeatureChange(self.NEW_INPUT_FEATURE, "clone-uniform", self.SPREAD)
        )
        self.assert_equal_memory(
            function_approximator.value_memory,
            {k: v for (k, v) in zip(self.MAPPED_KEYS_CLONE_UNIFORM, 2 * self.SORTED_VALUES)}
        )
        self.assertEqual(
            function_approximator._input_feature_set,
            self.POST_LEARNING_FEATURE_SET
        )
        function_approximator.assert_consistent_memory()

        function_approximator = self._generate_function_approximator()
        function_approximator.set_max_memory_size(10)
        function_approximator.set_rng(self.SEED)
        function_approximator.change_input_feature(
            FeatureChange(self.NEW_INPUT_FEATURE, "clone-gauss", self.SPREAD)
        )
        self.assert_equal_memory(
            function_approximator.value_memory,
            {k: v for (k, v) in zip(self.MAPPED_KEYS_CLONE_GAUSSIAN, 2 * self.SORTED_VALUES)}
        )
        self.assertEqual(
            function_approximator._input_feature_set,
            self.POST_LEARNING_FEATURE_SET
        )
        function_approximator.assert_consistent_memory()

        function_approximator = self._generate_function_approximator()
        function_approximator.change_input_feature(
            FeatureChange(self.FORGET_INPUT_FEATURE, "project")
        )
        self.assert_equal_memory(
            function_approximator.value_memory,
            {k: v for (k, v) in zip(self.MAPPED_KEYS_PROJECT, self.SORTED_VALUES)}
        )
        self.assertEqual(
            function_approximator._input_feature_set,
            self.POST_FORGETTING_FEATURE_SET
        )
        function_approximator.assert_consistent_memory()

        function_approximator = self._generate_function_approximator()
        function_approximator.change_input_feature(
            FeatureChange(self.FORGET_INPUT_FEATURE, "threshold", self.THRESHOLD)
        )
        self.assert_equal_memory(
            function_approximator.value_memory,
            {k: v for (k, v) in zip(self.MAPPED_KEYS_THRESHOLDED_PROJECTION, self.MAPPED_VALUES_THRESHOLDED_PROJECTION)}
        )
        self.assertEqual(
            function_approximator._input_feature_set,
            self.POST_FORGETTING_FEATURE_SET
        )
        function_approximator.assert_consistent_memory()

    def _test_map_zero_initialization(self):
        function_approximator = self._generate_function_approximator()
        self.assertEqual(
            function_approximator._map_zero_initialization(
                self.SORTED_KEYS.aggregate().T,
                self.INSERT_INDEX
            ),
            KeySet(self.MAPPED_KEYS_ZERO_INITIALIZATION)
        )

    def _test_map_perturb(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.set_rng(self.SEED)
        self.assertEqual(
            function_approximator._map_perturb(
                self.SORTED_KEYS.aggregate().T,
                self.INSERT_INDEX,
                self.SPREAD,
            ),
            KeySet(self.MAPPED_KEYS_PERTURB)
        )

    def _test_map_sample_cloning_uniform(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.set_rng(self.SEED)
        self.assertEqual(
            function_approximator._map_sample_cloning_uniform(
                self.SORTED_KEYS.aggregate().T,
                self.INSERT_INDEX,
                self.N_CLONES,
                self.SPREAD,
            ),
            KeySet(self.MAPPED_KEYS_CLONE_UNIFORM)
        )

    def _test_map_sample_cloning_gauss(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.set_rng(self.SEED)
        self.assertEqual(
            function_approximator._map_sample_cloning_gauss(
                self.SORTED_KEYS.aggregate().T,
                self.INSERT_INDEX,
                self.N_CLONES,
                self.SPREAD,
            ),
            KeySet(self.MAPPED_KEYS_CLONE_GAUSSIAN)
        )

    def _test_map_project(self):
        function_approximator = self._generate_function_approximator()
        self.assertEqual(
            function_approximator._map_project(
                self.SORTED_KEYS.aggregate().T,
                self.REMOVE_INDEX,
            ),
            KeySet(self.MAPPED_KEYS_PROJECT)
        )

    def _test_map_threshold(self):
        function_approximator = self._generate_function_approximator()
        keys1, values1, ages1 = function_approximator._map_threshold(
            self.SORTED_KEYS.aggregate().T,
            self.REMOVE_INDEX,
            self.THRESHOLD,
            self.SORTED_VALUES,
            [function_approximator.age_memory[k] for k in self.SORTED_KEYS],
        )
        keys2, values2, ages2 = (
            KeySet(self.MAPPED_KEYS_THRESHOLDED_PROJECTION),
            self.MAPPED_VALUES_THRESHOLDED_PROJECTION,
            self.MAPPED_AGES_THRESHOLDED_PROJECTION,
        )
        self.assertEqual(keys1, keys2)
        assert_array_equal(values1, values2)
        assert_array_almost_equal(np.array(ages1), np.array(ages2))

    def _test_assert_empty_memory(self):
        function_approximator = self._generate_function_approximator()
        self.assertRaises(
            AssertionError,
            function_approximator.assert_empty_memory,
        )
        empty_function_approximator = self._generate_empty_function_approximator()
        empty_function_approximator.assert_empty_memory()
