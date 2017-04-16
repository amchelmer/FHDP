import numpy as np

from ...features import Feature
from ...function_approximators.llr_function_approximator import Key, KeySet, LLRFunctionApproximator
from ...sets import FeatureSet
from ...tools.math_tools import hashable
from ...validation.object_validation import assert_not_in, assert_in, assert_true
from ...validation.type_validation import assert_is_type, assert_type_in


class Critic(LLRFunctionApproximator):
    """
    Critic class for approximating the discounted sum of future rewards.
    """
    OUTPUT_FEATURE_SET = FeatureSet([
        Feature("value", bounds=1e6 * np.array([-1, 1]))
    ])

    def __init__(self,
                 input_feature_set,
                 plant_feature_set,
                 knn,
                 max_memory,
                 trace,
                 alpha,
                 discount,
                 reward_function,
                 epsilon_p_feature):
        super(Critic, self).__init__(input_feature_set,
                                     self.OUTPUT_FEATURE_SET,
                                     plant_feature_set,
                                     knn,
                                     max_memory,
                                     epsilon_p_feature)
        for feature in self._input_feature_set:
            assert_in(feature, self._plant_state_feature_set)
        assert_type_in(trace, [int, float])
        self._alpha = alpha
        self._discount = discount
        self._trace = trace
        self._reward_function = reward_function
        self.trace_memory = {}

    def _key(self):
        sorted_keys, sorted_values = zip(*self)
        return (
            self._input_feature_set,
            self._output_feature_set,
            self._plant_state_feature_set,
            self._knn,
            self._alpha,
            self._discount,
            self._trace,
            self._epsilon,
            tuple(sorted_keys),
            tuple(hashable(v) for v in sorted_values)
        )

    def get_value(self, query_point):
        """
        Computes the value for a given query point
        :param query_point: state as numpy.array of shape(nstates, 1)
        :return: value as a numpy.array of shape(1, 1)
        """
        self.logger.debug("Query point {:s}".format(query_point.flatten()))

        query_features = self.like_me(query_point)
        knn_keys, knn_targets = self.get_closest_points(query_features)

        if len(knn_keys) < self._knn:
            value = self.get_zero_value()
            betas = np.zeros((1, len(self._input_feature_set) + 1))
            self.logger.warning(
                "Not enough data points in memory ({}). Resorting to zero value {}".format(len(self), value)
            )
        else:
            knn_features = self.unmake_keys(knn_keys)
            betas = self.local_fit(knn_features, knn_targets)
            self.logger.debug("Betas are {}".format(betas.flatten()))
            value = self.predict(betas, query_features)
            self.logger.debug("Predicting a value of {} for features {}".format(
                value.flatten(),
                query_features.flatten())
            )

        dv_dx = betas[:, :self._input_feature_set.n_states]
        self.logger.debug("dv/dx is {} for features {:s}".format(dv_dx.flatten(), query_features.flatten()))

        return value, knn_keys, dv_dx

    def update(self, query_point, prediction, td_error, knn_keys, metric):
        """
        Increments the trace memory with a quantity td_error and adds a new sample to the memory 
        :param query_point: states as numpy.array of shape(nstates, 1)
        :param prediction: predicted value as numpy.array of shape (1, 1)
        :param td_error: temporal difference error as float 
        :param knn_keys: KeySet of nearest neigbors
        :param metric: string, mean or min
        """
        if self._training_mode_flag:
            self.logger.debug("Updating Critic")
            assert_in(metric, self.UPDATE_METRICS)
            assert_is_type(knn_keys, KeySet)

            query_features = self.like_me(query_point)
            query_key = self.make_keys(query_features)

            dists = self.compute_distances(query_features, knn_key_set=knn_keys)
            if len(dists) == 0:
                min_dist, mean_dist = np.inf, np.inf
            else:
                min_dist, mean_dist = dists[0], dists.mean()

            decision_dist = min_dist if metric == "min" else mean_dist

            if min_dist > 0 and (decision_dist >= self._epsilon or len(self) < self._knn):
                self.add_sample(
                    query_features,
                    prediction,
                )
            else:
                self.logger.debug("Not adding key {}, {}_dist = {}".format(query_key, metric, decision_dist))

            if len(self) >= self._knn:
                keys, traces = zip(*self.trace_memory.items())
                self.logger.debug("Incrementing traces by {}".format(self._alpha * td_error))
                self.increment(
                    keys,
                    self._alpha * td_error * np.array(traces).reshape(1, -1),
                )

    def compute_td_error(self, reward, value, next_value):
        """
        Compute TD Error
        :param reward: reward at t+1
        :param value: value at t
        :param next_value: value at t+1
        :return: float of td error
        """
        td_error, = (reward + self._discount * next_value - value).flatten()
        self.logger.debug("Temporal difference error of {}".format(td_error))
        return td_error

    def add_sample(self, query_features, value):
        """
        Adds single sample to memory
        :param query_features: numpy array
        :param value: array of shape (n_outputs, 1)
        :param tree: add to tree
        """
        key = self.make_keys(query_features)
        try:
            assert_not_in(key, self.value_memory)
        except AssertionError:
            self.logger.warning("Trying to add point {}, but its already in the memory".format(key))
        self.value_memory[key] = value
        self.reset_trace(key)
        self.reset_age(key)
        self.tree.append(key)
        self.logger.debug("Added sample <{:s}: {:s}> to memory".format(key, value.flatten()))

    def reset_trace(self, keys):
        """
        Set trace for feature_set in feature_sets
        :param keys: Key or KeySet
        :return:
        """
        if self._training_mode_flag:
            assert_type_in(keys, [Key, KeySet])
            if isinstance(keys, Key):
                self.trace_memory[keys] = 1.
            else:
                for key in keys:
                    self.trace_memory[key] = 1.

    def update_traces(self):
        """
        Factor trace of all samples with trace
        :return:
        """
        if self._training_mode_flag:
            for key in self.trace_memory.keys():
                self.trace_memory[key] *= self._trace * self._discount
            self.logger.debug("Updated trace for all samples")

    def clear_traces(self):
        """
        Resets entries in trace memory
        :return:
        """
        if self._training_mode_flag:
            self.trace_memory.clear()
            self.logger.debug("Emptied trace memory")

    def get_discount(self):
        """
        :return: Returns discount factor 
        """
        return self._discount

    def get_reward_function(self):
        """
        :return: Returns reward function 
        """
        return self._reward_function

    def assert_empty_memory(self):
        """
        Ensures all memories are empty 
        """
        assert_true(
            len(self.value_memory) == len(self.age_memory) == len(self.trace_memory) == 0,
            "Non-empty memory: Value memory: {:d}, age memory: {:d}, trace memory: {:d}".format(
                len(self.value_memory),
                len(self.age_memory),
                len(self.trace_memory),
            )
        )
