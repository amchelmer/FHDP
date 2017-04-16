import numpy as np

from ...function_approximator_tests.llr_function_approximator_tests.llr_function_approximator_base_test import \
    LLRFunctionApproximatorBaseTest
from ....controllers.actor_critic_controller import Critic
from ....features import Feature
from ....function_approximators.llr_function_approximator import Key, KeySet
from ....reward_functions.quadratic_error_reward_function import QuadraticErrorRewardFunction
from ....sets import FeatureSet

from numpy.testing import assert_array_almost_equal


class TestCritic(LLRFunctionApproximatorBaseTest):
    SEED = 1803
    PLANT_FEATURE_SET = FeatureSet([
        Feature("x1"),
        Feature("x1_dot", scale=4., derivative=True, bounds=np.array([-2, 2])),
        Feature("x2", scale=2.),
        Feature("x2_dot", scale=0.4, derivative=True),
        Feature("u1", scale=0.5, feature_type="action", bounds=np.array([-2, 2])),
        Feature("u2", scale=2., feature_type="action", bounds=np.array([-3, 3])),
        Feature("u3", scale=1.1, feature_type="action", bounds=np.array([-3, 3]))
    ])
    INPUT_FEATURE_SET = FeatureSet([
        Feature("x1"),
        Feature("x1_dot", scale=4., derivative=True, bounds=np.array([-2, 2])),
        Feature("x2_dot", scale=0.4, derivative=True),
    ])
    OUTPUT_FEATURE_SET = FeatureSet([Feature("value")])
    KNN = 3
    MAX_MEMORY = 5
    TRACE = 0.75
    ALPHA = 0.03
    DISCOUNT = 0.95
    REWARD_FUNCTION = QuadraticErrorRewardFunction([1, 3, 0.5, 0.2], [0.1])

    QUERY_POINT = np.array([[-0.3, 0.5, 2.2, 1.2]]).T
    QUERY_FEATURES = np.array([[-0.3, 0.5, 1.2]]).T
    BETA = np.array([[-4.33752667, -1.03218972, -8.69026691, -1.9743365]])
    PREDICTION = np.array([[-11.61749365]]).T

    STATES = [
        np.array([[0.53, -0.98, 0.25, 1.2]]).T,
        np.array([[0.18, 0.06, -2.93, 3.4]]).T,
        np.array([[-0.84, -1.76, 2.63, 3.3]]).T,
        np.array([[-3.25, -0.57, -0.51, -0.1]]).T,
        np.array([[-4.82, 0.67, 1.03, 1.03]]).T,
    ]

    KEYS = KeySet([
        Key([0.53, -0.245, 3.]),
        Key([0.18, 0.015, 8.5]),
        Key([-0.84, -0.44, 8.25]),
        Key([-3.25, -0.1425, -0.25]),
        Key([-4.82, 0.1675, 2.575]),
    ])
    SORTED_KEYS = KEYS[::-1]

    VALUES = [
        np.array([[-13.69]]).T,
        np.array([[2.2]]).T,
        np.array([[-1.23]]).T,
        np.array([[13.58]]).T,
        np.array([[9.29]]).T,
    ]
    SORTED_VALUES = [
        np.array([[9.29]]).T,
        np.array([[13.58]]).T,
        np.array([[-1.23]]).T,
        np.array([[2.2]]).T,
        np.array([[-13.69]]).T,
    ]
    DISTANCES = np.array([0.908735, 5.522001, 5.307855, 4.397335, 4.540136])
    NEAREST_NEIGHBOR_INDICES = [0, 3, 4]

    NUMBER_OF_SAMPLES_IN_MEMORY = 5
    CONTAINED_KEY = KEYS[0]
    KEYS_SORTED_BY_AGE = KEYS[::-1]
    CUT_OFF_AGE = 0.35
    KEYS_AFTER_CUTOFF_AGE = KEYS[-3:]
    MEMORY_AFTER_CUTOFF_AGE = {k: v for (k, v) in zip(KEYS_AFTER_CUTOFF_AGE, VALUES[-3:])}

    NEW_INPUT_FEATURE = PLANT_FEATURE_SET[2]
    INSERT_INDEX = 3
    POST_LEARNING_FEATURE_SET = FeatureSet([
        Feature("x1"),
        Feature("x1_dot", scale=4., derivative=True, bounds=np.array([-2, 2])),
        Feature("x2_dot", scale=0.4, derivative=True),
        Feature("x2", scale=2.),
    ])
    SPREAD = 0.5
    N_CLONES = 2

    MAPPED_KEYS_ZERO_INITIALIZATION = [
        Key([-4.82, 0.1675, 2.575, 0]),
        Key([-3.25, -0.1425, -0.25, 0]),
        Key([-0.84, -0.44, 8.25, 0]),
        Key([0.18, 0.015, 8.5, 0]),
        Key([0.53, -0.245, 3., 0]),
    ]

    RNG = np.random.RandomState(SEED)
    MAPPED_KEYS_PERTURB = [
        Key([-4.82, 0.1675, 2.575, SPREAD * RNG.randn()]),
        Key([-3.25, -0.1425, -0.25, SPREAD * RNG.randn()]),
        Key([-0.84, -0.44, 8.25, SPREAD * RNG.randn()]),
        Key([0.18, 0.015, 8.5, SPREAD * RNG.randn()]),
        Key([0.53, -0.245, 3., SPREAD * RNG.randn()]),
    ]

    RNG = np.random.RandomState(SEED)
    MAPPED_KEYS_CLONE_UNIFORM = [
        Key([-4.82, 0.1675, 2.575, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-3.25, -0.1425, -0.25, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-0.84, -0.44, 8.25, RNG.uniform(-SPREAD, SPREAD)]),
        Key([0.18, 0.015, 8.5, RNG.uniform(-SPREAD, SPREAD)]),
        Key([0.53, -0.245, 3., RNG.uniform(-SPREAD, SPREAD)]),
        Key([-4.82, 0.1675, 2.575, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-3.25, -0.1425, -0.25, RNG.uniform(-SPREAD, SPREAD)]),
        Key([-0.84, -0.44, 8.25, RNG.uniform(-SPREAD, SPREAD)]),
        Key([0.18, 0.015, 8.5, RNG.uniform(-SPREAD, SPREAD)]),
        Key([0.53, -0.245, 3., RNG.uniform(-SPREAD, SPREAD)]),
    ]

    RNG = np.random.RandomState(SEED)
    MAPPED_KEYS_CLONE_GAUSSIAN = [
        Key([-4.82, 0.1675, 2.575, SPREAD * RNG.randn()]),
        Key([-3.25, -0.1425, -0.25, SPREAD * RNG.randn()]),
        Key([-0.84, -0.44, 8.25, SPREAD * RNG.randn()]),
        Key([0.18, 0.015, 8.5, SPREAD * RNG.randn()]),
        Key([0.53, -0.245, 3., SPREAD * RNG.randn()]),
        Key([-4.82, 0.1675, 2.575, SPREAD * RNG.randn()]),
        Key([-3.25, -0.1425, -0.25, SPREAD * RNG.randn()]),
        Key([-0.84, -0.44, 8.25, SPREAD * RNG.randn()]),
        Key([0.18, 0.015, 8.5, SPREAD * RNG.randn()]),
        Key([0.53, -0.245, 3., SPREAD * RNG.randn()]),
    ]

    FORGET_INPUT_FEATURE = INPUT_FEATURE_SET[1]
    REMOVE_INDEX = 1
    POST_FORGETTING_FEATURE_SET = FeatureSet([
        INPUT_FEATURE_SET[0],
        INPUT_FEATURE_SET[2],
    ])
    THRESHOLD = 0.35
    MAPPED_KEYS_PROJECT = [
        Key([-4.82, 2.575]),
        Key([-3.25, -0.25]),
        Key([-0.84, 8.25]),
        Key([0.18, 8.5]),
        Key([0.53, 3.]),
    ]

    MAPPED_KEYS_THRESHOLDED_PROJECTION = [
        Key([-4.82, 2.575]),
        Key([-3.25, -0.25]),
        Key([0.18, 8.5]),
        Key([0.53, 3.]),
    ]
    MAPPED_VALUES_THRESHOLDED_PROJECTION = [
        np.array([[9.29]]).T,
        np.array([[13.58]]).T,
        np.array([[2.2]]).T,
        np.array([[-13.69]]).T,
    ]
    MAPPED_AGES_THRESHOLDED_PROJECTION = (0.1, 0.2, 0.4, 0.5)

    def _get_function_approximator_cls(self):
        return Critic

    def _get_function_approximator_kwargs(self):
        return {
            "input_feature_set": self.INPUT_FEATURE_SET,
            "plant_feature_set": self.PLANT_FEATURE_SET,
            "knn": self.KNN,
            "max_memory": self.MAX_MEMORY,
            "trace": self.TRACE,
            "alpha": self.ALPHA,
            "discount": self.DISCOUNT,
            "reward_function": self.REWARD_FUNCTION,
            "epsilon_p_feature": self.EPSILON_P_FEATURE,
        }

    @staticmethod
    def _add_datapoints(function_approximator, states, values, dt):
        for state, value in zip(states, values):
            feature = function_approximator.like_me(state)
            function_approximator.add_sample(feature, value)
            function_approximator.age_samples(dt)
            function_approximator.update_traces()

    def lwlr_function_approximator_base_test(self):
        self._lwlr_function_approximator_base_test()

    def _test_increment(self):
        critic = self._generate_function_approximator()
        critic.increment(
            self.KEYS[1:4],
            np.array([[0.1, -0.3, 0]])
        )
        assert_array_almost_equal(
            critic.lookup_values(self.KEYS),
            np.concatenate(
                [
                    self.VALUES[0],
                    self.VALUES[1] + 0.1,
                    self.VALUES[2] - 0.3,
                    self.VALUES[3] + 0,
                    self.VALUES[4],
                ],
                axis=1
            )
        )

    def _test_change_output(self):
        pass

    def test_get_value(self):
        critic = self._generate_function_approximator()
        value, knn_keys, dv_dx = critic.get_value(self.QUERY_POINT)

        assert_array_almost_equal(
            value,
            self.PREDICTION
        )
        self.assertEqual(
            knn_keys,
            KeySet([self.KEYS[k] for k in self.NEAREST_NEIGHBOR_INDICES])
        )
        self.assertEqual(
            [critic.trace_memory[k] for k in knn_keys],
            [0.18362184356689445, 0.5076562499999999, 0.7124999999999999]
        )
        assert_array_almost_equal(
            dv_dx,
            self.BETA[:, :-1]
        )

    def test_update(self):
        critic = self._generate_empty_function_approximator()
        td_error = -2.3
        critic.update(self.STATES[0], self.VALUES[0], td_error, KeySet([]), self.METRIC)

        assert_array_almost_equal(
            critic.tree._tree.query(self.KEYS[0].get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )
        self.assertTrue(self.KEYS[0] in critic.value_memory)
        self.assertEqual(
            critic.value_memory[self.KEYS[0]],
            self.VALUES[0]
        )

        critic = self._generate_function_approximator()
        critic.update(
            self.QUERY_POINT,
            self.PREDICTION,
            td_error,
            KeySet([self.KEYS[k] for k in self.NEAREST_NEIGHBOR_INDICES]),
            self.METRIC,
        )
        assert_array_almost_equal(
            critic.tree._tree.query(critic.make_keys(self.QUERY_FEATURES).get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )
        self.assertTrue(critic.make_keys(self.QUERY_FEATURES) in critic.value_memory)
        values = [
            np.array([[-13.70266991]]).T,
            np.array([[2.18221767]]).T,
            np.array([[-1.25495765]]).T,
            np.array([[13.54497172]]).T,
            np.array([[9.2408375]]).T,
        ]
        for key, value in zip(self.KEYS, values):
            assert_array_almost_equal(
                critic.value_memory[key],
                value,
                decimal=8
            )

        close_features = [self.QUERY_FEATURES + 0.001 * k for k in range(1, 4)]
        for feature in close_features:
            critic.add_sample(feature, self.VALUES[0])
        critic.update(
            self.QUERY_POINT - 0.001,
            self.PREDICTION,
            td_error,
            KeySet([critic.make_keys(feature) for feature in close_features]),
            self.METRIC
        )
        self.assertTrue(critic.tree._tree.query(critic.make_keys(self.QUERY_FEATURES - 0.001).get_array().T)[0] > 0)
        self.assertTrue(critic.make_keys(self.QUERY_FEATURES - 0.001) not in critic.value_memory)

    def _test_add_sample(self):
        critic = self._generate_empty_function_approximator()
        self.assertEqual(
            critic.value_memory,
            {}
        )
        self.assertEqual(
            critic.trace_memory,
            {}
        )
        self.assertEqual(
            critic.age_memory,
            {}
        )
        critic.add_sample(critic.like_me(self.STATES[0]), self.VALUES[0])
        self.assertEqual(
            critic.value_memory,
            {self.KEYS[0]: self.VALUES[0]}
        )
        self.assertEqual(
            critic.trace_memory,
            {self.KEYS[0]: 1.}
        )
        self.assertEqual(
            critic.age_memory,
            {self.KEYS[0]: 0.}
        )
        assert_array_almost_equal(
            critic.tree._tree.query(self.KEYS[0].get_array().T)[0],
            np.array([[0]]),
            decimal=8
        )

    def test_compute_td_error(self):
        critic = self._generate_empty_function_approximator()
        reward = 2
        value = np.array([[10]])
        next_value = np.array([[11]])
        self.assertAlmostEqual(
            critic.compute_td_error(reward, value, next_value),
            2.45,
            places=8
        )

    def _test_update_traces(self):
        critic = self._generate_empty_function_approximator()
        critic.add_sample(self.KEYS[0], self.VALUES[0])
        critic.update_traces()
        critic.add_sample(self.KEYS[1], self.VALUES[1])
        critic.update_traces()

        self.assertAlmostEqual(
            critic.trace_memory[self.KEYS[0]],
            (self.TRACE * self.DISCOUNT) ** 2,
            places=8
        )
        self.assertAlmostEqual(
            critic.trace_memory[self.KEYS[1]],
            (self.TRACE * self.DISCOUNT) ** 1,
            places=8
        )

    def _test_set_trace(self):
        critic = self._generate_function_approximator()
        traces = [critic.trace_memory[k] for k in self.KEYS]

        critic.set_trace(self.KEYS[0])
        critic.set_trace(self.KEYS[3])
        traces[0], traces[3] = 1, 1

        for key, trace in zip(self.KEYS, traces):
            self.assertAlmostEqual(
                critic.trace_memory[key],
                trace,
                places=8
            )

    def _test_clear_traces(self):
        critic = self._generate_function_approximator()
        critic.clear_traces()
        self.assertEqual(
            critic.trace_memory,
            {}
        )

    def test_get_discount(self):
        critic = self._generate_empty_function_approximator()
        self.assertEqual(
            critic.get_discount(),
            critic._discount
        )

    def test_get_reward_function(self):
        critic = self._generate_empty_function_approximator()
        self.assertEqual(
            critic.get_reward_function(),
            critic._reward_function
        )
