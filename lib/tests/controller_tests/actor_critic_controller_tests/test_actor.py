import numpy as np

from ...function_approximator_tests.llr_function_approximator_tests.llr_function_approximator_base_test import \
    LLRFunctionApproximatorBaseTest
from ....controllers.actor_critic_controller import Actor
from ....features import Feature, FeatureChange
from ....function_approximators.llr_function_approximator import Key, KeySet
from ....sets import FeatureSet
from ....tools.math_tools import saturate

from numpy.testing import assert_array_almost_equal


class TestActor(LLRFunctionApproximatorBaseTest):
    SEED = 1803

    PLANT_FEATURE_SET = FeatureSet([
        Feature("x1"),
        Feature("x1_dot", scale=4., derivative=True, bounds=np.array([-2, 2])),
        Feature("x2", scale=2.),
        Feature("x2_dot", scale=0.4, derivative=True),
        Feature("u1", scale=0.5, feature_type="action", bounds=np.array([-2, 2])),
        Feature("u2", scale=2., feature_type="action", bounds=np.array([-3, 3])),
        Feature("u3", scale=1.1, feature_type="action", bounds=np.array([-3, 3])),
    ])
    INPUT_FEATURE_SET = PLANT_FEATURE_SET[:3]
    OUTPUT_FEATURE_SET = PLANT_FEATURE_SET[4:6]
    KNN = 3
    MAX_MEMORY = 5
    ALPHA = 0.03
    METRIC = "min"

    QUERY_POINT = np.array([[-0.8, 5.2, 1.5, 2.2]]).T
    QUERY_FEATURES = np.array([[-0.8, 5.2, 1.5]]).T
    BETA = np.array([[-0.08058936, 0.10336341, -0.07084446, 0.00707762],
                     [-0.44144749, -0.4704881, 0.26161166, 0.17253494]])
    PREDICTION = np.array([[0.50277215, -1.52842769]]).T
    PREDICTION_AS_PLANT = np.array([[0.50277215, -1.52842769, 0]]).T

    STATES = [
        np.array([[-1.14, -10., -9.1, 1.2]]).T,
        np.array([[9.9, 0.17, 13.02, 3.4]]).T,
        np.array([[-13.76, 4.33, 11.24, 3.3]]).T,
        np.array([[-10.16, 13.29, -5.37, -0.1]]).T,
        np.array([[-6., 5.72, 1.72, 1.03]]).T,
    ]
    KEYS = KeySet([
        Key([-1.14, -2.5, -4.55]),
        Key([9.9, 0.0425, 6.51]),
        Key([-13.76, 1.0825, 5.62]),
        Key([-10.16, 3.3225, -2.685]),
        Key([-6., 1.43, 0.86]),
    ])
    SORTED_KEYS = KeySet([
        Key([-13.76, 1.0825, 5.62]),
        Key([-10.16, 3.3225, -2.685]),
        Key([-6., 1.43, 0.86]),
        Key([-1.14, -2.5, -4.55]),
        Key([9.9, 0.0425, 6.51])
    ])
    VALUES = [
        np.array([[-0.29, 3.0]]).T,
        np.array([[1.60, -0.24]]).T,
        np.array([[-2.0, 0.07]]).T,
        np.array([[2.58, -3.0]]).T,
        np.array([[0.96, 0.58]]).T
    ]
    SORTED_VALUES = [
        np.array([[-2.0, 0.07]]).T,
        np.array([[2.58, -3.0]]).T,
        np.array([[0.96, 0.58]]).T,
        np.array([[-0.29, 3.0]]).T,
        np.array([[1.60, -0.24]]).T
    ]

    DISTANCES = np.array([6.53036, 12.216747, 13.846509, 10.173462, 5.202788])
    NEAREST_NEIGHBOR_INDICES = [4, 0, 3]

    NUMBER_OF_SAMPLES_IN_MEMORY = 5
    CONTAINED_KEY = KEYS[0]
    KEYS_SORTED_BY_AGE = KEYS[::-1]
    CUT_OFF_AGE = 0.35
    KEYS_AFTER_CUTOFF_AGE = KEYS[-3:]
    MEMORY_AFTER_CUTOFF_AGE = {k: v for (k, v) in zip(KEYS_AFTER_CUTOFF_AGE, VALUES[-3:])}

    NEW_INPUT_FEATURE = PLANT_FEATURE_SET[3]
    INSERT_INDEX = 3
    POST_LEARNING_FEATURE_SET = FeatureSet([
        Feature("x1"),
        Feature("x1_dot", scale=4., derivative=True, bounds=np.array([-2, 2])),
        Feature("x2", scale=2.),
        Feature("x2_dot", scale=0.4, derivative=True),
    ])
    SPREAD = 0.5
    N_CLONES = 2

    MAPPED_KEYS_ZERO_INITIALIZATION = [
        Key([-13.76, 1.0825, 5.62, 0]),
        Key([-10.16, 3.3225, -2.685, 0]),
        Key([-6., 1.43, 0.86, 0]),
        Key([-1.14, -2.5, -4.55, 0]),
        Key([9.9, 0.0425, 6.51, 0]),
    ]
    RNG = np.random.RandomState(SEED)
    MAPPED_KEYS_PERTURB = [
        Key([-13.76, 1.0825, 5.62, SPREAD * RNG.randn()]),
        Key([-10.16, 3.3225, -2.685, SPREAD * RNG.randn()]),
        Key([-6., 1.43, 0.86, SPREAD * RNG.randn()]),
        Key([-1.14, -2.5, -4.55, SPREAD * RNG.randn()]),
        Key([9.9, 0.0425, 6.51, SPREAD * RNG.randn()]),
    ]

    RNG = np.random.RandomState(SEED)
    MAPPED_KEYS_CLONE_UNIFORM = [
        Key([-13.76, 1.0825, 5.62, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-10.16, 3.3225, -2.685, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-6., 1.43, 0.86, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-1.14, -2.5, -4.55, RNG.uniform(-SPREAD, SPREAD)]),
        Key([9.9, 0.0425, 6.51, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-13.76, 1.0825, 5.62, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-10.16, 3.3225, -2.685, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-6., 1.43, 0.86, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-1.14, -2.5, -4.55, RNG.uniform(-SPREAD, SPREAD)]),
        Key([9.9, 0.0425, 6.51, RNG.uniform(-SPREAD, SPREAD)]),
    ]

    RNG = np.random.RandomState(SEED)
    MAPPED_KEYS_CLONE_GAUSSIAN = [
        Key([-13.76, 1.0825, 5.62, SPREAD * RNG.randn()]),
        Key([-10.16, 3.3225, -2.685, SPREAD * RNG.randn()]),
        Key([-6., 1.43, 0.86, SPREAD * RNG.randn()]),
        Key([-1.14, -2.5, -4.55, SPREAD * RNG.randn()]),
        Key([9.9, 0.0425, 6.51, SPREAD * RNG.randn()]),
        Key([-13.76, 1.0825, 5.62, SPREAD * RNG.randn()]),
        Key([-10.16, 3.3225, -2.685, SPREAD * RNG.randn()]),
        Key([-6., 1.43, 0.86, SPREAD * RNG.randn()]),
        Key([-1.14, -2.5, -4.55, SPREAD * RNG.randn()]),
        Key([9.9, 0.0425, 6.51, SPREAD * RNG.randn()]),
    ]

    FORGET_INPUT_FEATURE = INPUT_FEATURE_SET[1]
    POST_FORGETTING_FEATURE_SET = FeatureSet([
        INPUT_FEATURE_SET[0],
        INPUT_FEATURE_SET[2],
    ])

    REMOVE_INDEX = 1

    MAPPED_KEYS_PROJECT = [
        Key([-13.76, 5.62]),
        Key([-10.16, -2.685]),
        Key([-6., 0.86]),
        Key([-1.14, -4.55]),
        Key([9.9, 6.51])
    ]
    THRESHOLD = 2.4
    MAPPED_KEYS_THRESHOLDED_PROJECTION = [
        Key([-13.76, 5.62]),
        Key([-6., 0.86]),
        Key([9.9, 6.51])
    ]
    MAPPED_VALUES_THRESHOLDED_PROJECTION = [
        np.array([[-2.0, 0.07]]).T,
        np.array([[0.96, 0.58]]).T,
        np.array([[1.60, -0.24]]).T
    ]
    MAPPED_AGES_THRESHOLDED_PROJECTION = np.array([0.3, 0.1, 0.4])

    def _get_function_approximator_cls(self):
        return Actor

    def _get_function_approximator_kwargs(self):
        return {
            "input_feature_set": self.INPUT_FEATURE_SET,
            "output_feature_set": self.OUTPUT_FEATURE_SET,
            "plant_feature_set": self.PLANT_FEATURE_SET,
            "knn": self.KNN,
            "max_memory": self.MAX_MEMORY,
            "alpha": self.ALPHA,
            "epsilon_p_feature": self.EPSILON_P_FEATURE,
        }

    def lwlr_function_approximator_base_test(self):
        self._lwlr_function_approximator_base_test()

    def _test_increment(self):
        function_approximator = self._generate_function_approximator()
        function_approximator.increment(
            self.KEYS[1:4],
            np.array([[0.1, -0.1], [-0.3, 0.21], [0, 0.95]]).T
        )
        assert_array_almost_equal(
            function_approximator.lookup_values(self.KEYS),
            np.concatenate(
                [
                    self.VALUES[0],
                    np.array([[+1.70, -0.340]]).T,
                    np.array([[-2.0, +0.280]]).T,
                    np.array([[+2., -2.050]]).T,
                    self.VALUES[4],
                ],
                axis=1)
        )

    def test_get_action(self):
        actor = self._generate_function_approximator()
        action, knn_keys = actor.get_action(self.QUERY_POINT)

        assert_array_almost_equal(
            action,
            self.PREDICTION_AS_PLANT,
            decimal=8
        )
        self.assertEqual(
            knn_keys,
            KeySet([self.KEYS[k] for k in self.NEAREST_NEIGHBOR_INDICES])
        )

    def test_update(self):
        actor = self._generate_empty_function_approximator()

        delta = np.array([[2.1, -1.32]]).T
        action = np.concatenate([self.VALUES[0], np.zeros((1, 1))], axis=0)
        actor.update(self.STATES[0], action, delta, KeySet([]), self.METRIC)

        assert_array_almost_equal(
            actor.tree._tree.query(self.KEYS[0].get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )
        self.assertTrue(self.KEYS[0] in actor.value_memory)
        assert_array_almost_equal(
            actor.value_memory[self.KEYS[0]],
            self.VALUES[0] + np.array([self.ALPHA]) * delta,
            decimal=8
        )

        actor = self._generate_function_approximator()
        actor.update(
            self.QUERY_POINT,
            self.PREDICTION_AS_PLANT,
            delta,
            KeySet([self.KEYS[k] for k in self.NEAREST_NEIGHBOR_INDICES]), self.METRIC
        )
        assert_array_almost_equal(
            actor.tree._tree.query(actor.make_keys(self.QUERY_FEATURES).get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )
        self.assertTrue(actor.make_keys(self.QUERY_FEATURES) in actor.value_memory)
        values = [
            np.array([[-0.227, +2.9604]]).T,
            np.array([[1.60, -0.24]]).T,
            np.array([[-2.0, 0.07]]).T,
            np.array([[+2, -3.0]]).T,
            np.array([[+1.023, +0.5404]]).T,
        ]
        for key, value in zip(self.KEYS, values):
            assert_array_almost_equal(
                actor.value_memory[key],
                value,
                decimal=8
            )

        close_features = [self.QUERY_FEATURES + 0.001 * k for k in range(1, 4)]
        for feature in close_features:
            actor.add_sample(feature, self.VALUES[0])
        actor.update(
            self.QUERY_POINT - 0.001,
            self.PREDICTION_AS_PLANT,
            delta,
            KeySet([actor.make_keys(feature) for feature in close_features]),
            self.METRIC
        )
        self.assertTrue(actor.tree._tree.query(actor.make_keys(self.QUERY_FEATURES - 0.001).get_array().T)[0] > 0)
        self.assertTrue(actor.make_keys(self.QUERY_FEATURES - 0.001) not in actor.value_memory)

    def test_perturb_action(self):
        actor = self._generate_function_approximator()
        actor.set_rng(self.SEED)
        frac = 12.5
        assert_array_almost_equal(
            actor.perturb_action(self.VALUES[0], frac=frac),
            saturate(
                self.VALUES[0] +
                actor._output_feature_set.get_bounds()[:, 1:] * np.random.RandomState(self.SEED).randn(2, 1) / frac,
                actor._output_feature_set.get_bounds()
            ),
            decimal=8
        )

    def test_action_like_plant(self):
        actor = self._generate_function_approximator()
        assert_array_almost_equal(
            actor.action_like_plant(self.PREDICTION),
            self.PREDICTION_AS_PLANT
        )

    def test_action_like_me(self):
        actor = self._generate_function_approximator()
        assert_array_almost_equal(
            actor.action_like_me(self.PREDICTION_AS_PLANT),
            self.PREDICTION
        )

    def test_change_output_feature(self):
        actor = self._generate_function_approximator()
        new_feature = Feature("u3", scale=1.1, feature_type="action", bounds=np.array([-3, 3]))
        actor.set_rng(self.SEED)
        actor.change_output_feature(FeatureChange(new_feature, "zero"))
        rng = np.random.RandomState(self.SEED)

        new_values = [
            np.array([[-2.0, 0.07, rng.randn() / actor.INITIAL_PERTURBATION_FRACTION]]).T,
            np.array([[2.58, -3.0, rng.randn() / actor.INITIAL_PERTURBATION_FRACTION]]).T,
            np.array([[0.96, 0.58, rng.randn() / actor.INITIAL_PERTURBATION_FRACTION]]).T,
            np.array([[-0.29, 3.0, rng.randn() / actor.INITIAL_PERTURBATION_FRACTION]]).T,
            np.array([[1.60, -0.24, rng.randn() / actor.INITIAL_PERTURBATION_FRACTION]]).T,
        ]

        self.assert_equal_memory(
            actor.value_memory,
            {k: v for (k, v) in zip(self.SORTED_KEYS, new_values)}
        )
        self.assertEqual(
            actor._output_feature_set,
            FeatureSet([
                Feature("u1", scale=0.5, feature_type="action", bounds=np.array([-2, 2])),
                Feature("u2", scale=2., feature_type="action", bounds=np.array([-3, 3])),
                Feature("u3", scale=1.1, feature_type="action", bounds=np.array([-3, 3])),
            ])
        )
        actor.assert_consistent_memory()
