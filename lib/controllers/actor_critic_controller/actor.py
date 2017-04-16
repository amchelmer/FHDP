import numpy as np

from ...function_approximators.llr_function_approximator import LLRFunctionApproximator
from ...tools.math_tools import saturate, hashable
from ...validation.object_validation import assert_in, assert_true


class Actor(LLRFunctionApproximator):
    """
    Actor class for learning the optimal action given some state. It is used in the ActorCriticController class.
    """
    PERTURBATION_FRACTION = 3.
    INITIAL_PERTURBATION_FRACTION = 30.

    def __init__(self,
                 input_feature_set,
                 output_feature_set,
                 plant_feature_set,
                 knn,
                 max_memory,
                 alpha,
                 epsilon_p_feature):
        super(Actor, self).__init__(input_feature_set,
                                    output_feature_set,
                                    plant_feature_set,
                                    knn,
                                    max_memory,
                                    epsilon_p_feature)
        self._alpha = alpha
        self._plant_action_set = plant_feature_set.get_action_set()

        for feature in self._input_feature_set:
            assert_in(feature, self._plant_state_feature_set)
        for feature in self._output_feature_set:
            assert_in(feature, self._plant_action_set)

    def _key(self):
        sorted_keys, sorted_values = zip(*self)
        return (
            self._knn,
            self._input_feature_set,
            self._output_feature_set,
            self._plant_state_feature_set,
            self._max_memory,
            self._epsilon,
            self._alpha,
            tuple(sorted_keys),
            tuple(hashable(v) for v in sorted_values)
        )

    def get_action(self, query_point):
        """
        Compute the action given a state
        :param query_point: numpy.array of shape(nfeatures, 1)
        :return: numpy.array of shape(naction, 1)
        """
        self.logger.debug("Query point {:s}".format(query_point.flatten()))
        query_features = self.like_me(query_point)
        knn_keys, knn_targets = self.get_closest_points(query_features)

        if len(knn_keys) < self._knn:
            self.logger.warning(
                "Not enough data points in memory ({}). Resorting to random action.".format(len(self)))
            action = saturate(
                self.perturb_action(
                    self.get_zero_value(),
                    frac=self.INITIAL_PERTURBATION_FRACTION
                ),
                self._output_feature_set.get_bounds()
            )
        else:
            knn_features = self.unmake_keys(knn_keys)
            betas = self.local_fit(knn_features, knn_targets)
            action = self.predict(betas, query_features)
            self.logger.debug("Computed action {} for key {}".format(
                action.flatten(),
                self.make_keys(query_features),
            ))

        return self.action_like_plant(action), knn_keys

    def update(self, query_point, action, delta, knn_keys, metric):
        """
        Updates actor memory
        :param query_point: state as numpy.array of shape(nstates, 1)
        :param action: action to be stored in memory as numpy.array
        :param delta: numpy.array of same shape as action with increment values
        :param knn_keys: KeySet of keys to be updated with delta
        :param metric: string, mean or min
        """
        if self._training_mode_flag:
            self.logger.debug("Updating actor")
            assert_in(metric, self.UPDATE_METRICS)

            query_features = self.like_me(query_point)
            query_key = self.make_keys(query_features)
            action_features = self.action_like_me(action)

            self.logger.debug("Incrementing nearest neighbors by {}".format(self._alpha * delta.flatten()))
            self.increment(
                knn_keys,
                self._alpha * delta.repeat(len(knn_keys), axis=1),
            )

            dists = self.compute_distances(query_features, knn_key_set=knn_keys)
            if len(dists) == 0:
                min_dist, mean_dist = np.inf, np.inf
            else:
                min_dist, mean_dist = dists[0], dists.mean()

            decision_dist = min_dist if metric == "min" else mean_dist
            if min_dist > 0 and (decision_dist >= self._epsilon or len(self) < self._knn):
                self.add_sample(
                    query_features,
                    saturate(
                        action_features + self._alpha * delta,
                        self._output_feature_set.get_bounds()
                    ),
                )
            else:
                self.logger.debug("Not adding key {}, {}_dist = {:.3f}".format(query_key, metric, decision_dist))

    def perturb_action(self, action, frac=None):
        """
        Perturbs an action using a Gaussian distribution with mean=0, sigma=1/frac. If no value for frac is given, 
        it defaults to the self.PERTURBATION_FRACTION
        :param action: action as numpy.array of shape(nactions, 1)
        :param frac: float representing 1/sigma
        :return: perturbed action as numpy.array of shape(nactions, 1)
        """
        perturbation = np.divide(
            self._output_feature_set.get_bounds()[:, 1:] * self._rng.randn(*action.shape),
            frac if frac is not None else self.PERTURBATION_FRACTION
        )
        perturbed_action = saturate(
            action + perturbation,
            self._output_feature_set.get_bounds()
        )
        self.logger.debug("Perturbed action {:s} by {:s} to {:s}".format(
            action.flatten(),
            perturbation.flatten(),
            perturbed_action.flatten(),
        ))
        return perturbed_action

    def action_like_plant(self, action):
        """
        Convenience method for describing an action with the FeatureSet of the plant. 
        :param action: Actor action as numpy.array of shape(nactions_actor, 1)
        :return: plant action as numpy.array of shape(nactions_plant, 1)
        """
        return self._plant_action_set.get_action_set().like_me(action, self._output_feature_set)

    def action_like_me(self, action):
        """
        Convenience method for describing a plant action with the FeatureSet of the Actor. 
        :param action: plant action as numpy.array of shape(nactions_plant, 1)
        :return: Actor action as numpy.array of shape(nactions_actor, 1)
        """
        return self._output_feature_set.like_me(action, self._plant_action_set)

    def change_output_feature(self, feature_change):
        """
        Change the output features by adding an new action feature 
        :param feature_change: FeatureChange object with an action feature
        :return: 
        """
        assert_true(feature_change.is_action(), "Cannot add states as Actor outputs")
        feature = feature_change.get_feature()

        keys, old_values = zip(*self)
        old_values_array = np.hstack(old_values).T
        ages = [self.age_memory[k] for k in keys]

        self.value_memory.clear()
        self.age_memory.clear()
        self.assert_empty_memory()

        new_values_as_keys = self._map_perturb(
            old_values_array,
            self._output_feature_set.get_insert_position(feature),
            1. / self.INITIAL_PERTURBATION_FRACTION,
        )
        new_values = map(lambda x: x.get_array(), new_values_as_keys)

        for key, age, new_value in zip(keys, ages, new_values):
            self.value_memory[key] = new_value
            self.age_memory[key] = age

        self.logger.info(
            "Performing output change {} on output set {}".format(feature_change, self._output_feature_set)
        )
        self._output_feature_set = feature_change.apply(self._output_feature_set)
        self.rebuild_tree()
