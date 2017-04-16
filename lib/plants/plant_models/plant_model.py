import matplotlib.pyplot as plt
import numpy as np

from ...function_approximators.llr_function_approximator import LLRFunctionApproximator
from ...sets import FeatureSet
from ...tools.math_tools import hashable
from ...validation.object_validation import assert_in


class PlantModel(LLRFunctionApproximator):
    """
    PlantModel class for approximating plant dynamics. Aims to predict the next state given a state and an action.
    """

    def __init__(self, plant_feature_set, feature_set, knn, max_memory, prediction_epsilon):
        super(PlantModel, self).__init__(
            FeatureSet([f for f in feature_set if f.is_action() or f.is_derivative()]),
            FeatureSet([f for f in feature_set if f.is_state()]),
            plant_feature_set,
            knn,
            max_memory,
            prediction_epsilon,
        )

        for feature in self._input_feature_set:
            assert_in(feature, self._plant_feature_set)
        for feature in self._output_feature_set:
            assert_in(feature, self._plant_state_feature_set)

    def _key(self):
        sorted_keys, sorted_values = zip(*self)
        return (
            self._input_feature_set,
            self._output_feature_set,
            self._knn,
            self._epsilon,
            self._max_memory,
            tuple(sorted_keys),
            tuple(hashable(v) for v in sorted_values),
        )

    def _set_epsilon(self, e):
        """
        Set tolerance for deciding whether a sample should be added to memory
        :param e: tolerance as float
        """
        self._epsilon = e ** 0.5

    def like_me(self, state):
        """
        Convenience method for describing a plant state by the FeatureSet of the LLR
        :param state: plant state as numpy.array of shape(nplantstates, 1) 
        :return: state as features of the LLR
        """
        features = self._input_feature_set.like_me(state, self._plant_feature_set)
        self.logger.debug("Input features are {}".format(features.flatten()))
        return features

    @staticmethod
    def merge_state_action(state, action):
        """
        Stack a state and action into one column vector
        :param state: numpy.array of shape(nstates, 1)
        :param action: numpy.array of shape(nactions, 1)
        :return: numpy.array of shape(nstates + nactions, 1)
        """
        return np.vstack([state, action])

    def get_next_state(self, query_state, query_action):
        """
        Estimates next state from a state and saturated action.
        :param query_state: numpy.array of shape(nstates, 1)
        :param query_action: numpy.array of shape(nactions, 1)
        :return: next_state and dxprime_du
        """
        self.logger.debug("Query state is {:s}, action is {:s}".format(
            query_state.flatten(),
            query_action.flatten(),
        ))
        query_features = self.like_me(
            self.merge_state_action(query_state, query_action)
        )
        knn_keys, knn_targets = self.get_closest_points(query_features)

        if len(knn_keys) < self._knn:
            betas = np.zeros((len(self._output_feature_set), len(self._input_feature_set)))
            state_diff = np.zeros((len(self._output_feature_set), 1))
            self.logger.warning(
                "Not enough data points in memory ({}). Resorting to zero diff".format(len(self))
            )
        else:
            knn_features = self.unmake_keys(knn_keys)
            betas = self.local_fit(knn_features, knn_targets)
            state_diff = self.predict(betas, query_features)
            self.logger.debug("Predicted state diff to be {:s}".format(state_diff.flatten()))

        next_state = query_state + self._plant_state_feature_set.like_me(state_diff, self._output_feature_set)
        dxprime_du = betas[:, -(self._input_feature_set.n_actions + 1):-1]
        self.logger.debug("dx/du = {:s}".format(np.array2string(dxprime_du.T, separator=" ").replace('\n', '')))

        return next_state, knn_keys, dxprime_du

    def update(self, query_state, query_action, predicted_state_prime, truth_state_prime):
        """
        Updates plant model by adding a sample if the tolerance of the prediction is violated.
        :param query_state: state as numpy.array of shape(nstates, 1)
        :param query_action: action as numpy.array of shape(nactions, 1)
        :param predicted_state_prime: predicted state as numpy.array of shape(nstates, 1)
        :param truth_state_prime: truth state as numpy.array of shape(nstates, 1)
        """
        if self._training_mode_flag:
            self.logger.debug("Updating plant model")
            query_features = self.like_me(
                self.merge_state_action(query_state, query_action)
            )

            truth_features = self._output_feature_set.like_me(
                truth_state_prime,
                self._plant_state_feature_set,
            )
            prediction_features = self._output_feature_set.like_me(
                predicted_state_prime,
                self._plant_state_feature_set,
            )

            error, norm = self._compute_prediction_error(prediction_features, truth_features)
            self.logger.debug("Prediction error is {:s}, 1-norm is {:.5f}".format(error.flatten(), norm))

            query_key = self.make_keys(query_features)
            if query_key not in self and (norm > self._epsilon or len(self) < self._knn):
                self.add_sample(
                    query_features,
                    truth_features - self._output_feature_set.like_me(
                        query_state,
                        self._plant_state_feature_set,
                    ),
                )
                self.logger.debug("Error is {:.5f}, so added point to memory".format(norm))
            return norm
        else:
            return 0.

    @staticmethod
    def _compute_prediction_error(prediction, target):
        """
        Compute state prediction error with respect to target state.
        :param prediction: numpy.array of shape (nstates, 1)
        :param target: numpy.array of shape (nstates, 1)
        :return: error as numpy.array of shape (nstates, 1), magnitude of error
        """
        error = prediction - target
        return error, np.linalg.norm(error, ord=2)

    def change_input_feature(self, feature_change):
        """
        Change input feature of plant model
        :param feature_change: FeatureChange object
        """
        feature = feature_change.get_feature()

        self.logger.info(
            "Performing feature change {:s} on {:s}".format(feature_change, self.__class__.__name__)
        )
        self.value_memory.clear()
        self.age_memory.clear()

        if feature.is_action():
            self._input_feature_set = feature_change.apply(self._input_feature_set)
        elif feature.is_state() and feature.is_derivative():
            self._input_feature_set = feature_change.apply(self._input_feature_set)
            self._output_feature_set = feature_change.apply(self._output_feature_set)
        else:
            self._output_feature_set = feature_change.apply(self._output_feature_set)

        self.rebuild_tree()
        self.assert_consistent_memory()

    def one_step_ahead_simulation(self, plant, length, controller, initial_state=None):
        """
        Plot state errors in one-step-ahead simulation for assessing plant model quality
        :param plant: Plant object to simulate on
        :param length: episode length in float or integer
        :param controller: Controller with 'get_action' method
        :param initial_state: numpy.array of shape(nstates, 1) or None
        """
        plant_output = plant.get_feature_set().get_state_set()
        sim = plant.simulate(length, controller, initial_state=initial_state)
        predictions = np.hstack(
            [self._output_feature_set.like_me(
                self.get_next_state(s, a)[0],
                plant_output,
            ) for (s, a, _, _) in sim]
        )[:, :-1]
        truth = np.hstack([self._output_feature_set.like_me(s, plant_output) for (s, _, _, _) in sim])[:, 1:]

        time = sim.get_time_vector()[1:]

        fig, axes = plt.subplots(nrows=len(self._output_feature_set))
        for i in range(len(axes)):
            axes[i].plot(time, truth[i])
            axes[i].plot(time, predictions[i], color="red")
            axes[i].set_ylabel(self._output_feature_set.get_names()[i])
            axes[i].legend(["Plant", "Model"])
        axes[-1].set_xlabel("Time [s]")
        return axes

    def one_step_ahead_errors(self, plant, length, controller, initial_state=None):
        """
        Plot error magnitudes in one-step-ahead simulation for assessing plant model quality
        :param plant: Plant object to simulate on
        :param length: episode length in float or integer
        :param controller: Controller with 'get_action' method
        :param initial_state: numpy.array of shape(nstates, 1) or None
        """
        plant_output = plant.get_feature_set().get_state_set()

        sim = plant.simulate(length, controller, initial_state=initial_state)
        predictions = np.hstack(
            [self._output_feature_set.like_me(
                self.get_next_state(s, a)[0],
                plant_output,
            ) for (s, a, _, _) in sim]
        )[:, :-1]
        truth = np.hstack([self._output_feature_set.like_me(s, plant_output) for (s, _, _, _) in sim])[:, 1:]
        errors = predictions - truth
        time = sim.get_time_vector()[1:]

        fig, axes = plt.subplots(nrows=len(self._output_feature_set))
        for i in range(len(axes)):
            axes[i].plot(time, errors[i])
            axes[i].set_ylabel(self._output_feature_set.get_names()[i])
        axes[-1].set_xlabel("Time [s]")
        return axes

    def all_step_ahead_simulation(self, plant, length, controller, initial_state=None):
        """
        Plot state errors in bootstrap simulation for assessing plant model quality
        :param plant: Plant object to simulate on
        :param length: episode length in float or integer
        :param controller: Controller with 'get_action' method
        :param initial_state: numpy.array of shape(nstates, 1) or None
        """
        plant_output = plant.get_feature_set().get_state_set()
        sim = plant.simulate(length, controller, initial_state=initial_state)

        time = sim.get_time_vector()

        states = [sim.get_states()[:, 0:1]]
        for _ in time:
            states.append(
                self.get_next_state(states[-1], controller.get_action(states[-1]))[0],
            )
        predictions = np.hstack(
            [self._output_feature_set.like_me(s, plant_output) for s in states]
        )[:, 1:-1]
        truth = np.hstack([self._output_feature_set.like_me(s, plant_output) for (s, _, _, _) in sim])[:, 1:]

        fig, axes = plt.subplots(nrows=len(self._output_feature_set))
        for i in range(len(axes)):
            axes[i].plot(time[1:], truth[i])
            axes[i].plot(time[1:], predictions[i], color="red")
            axes[i].set_ylabel(self._output_feature_set.get_names()[i])
            axes[i].legend(["Plant", "Model"])
        axes[-1].set_xlabel("Time [s]")
        return axes

    def all_step_ahead_errors(self, plant, length, controller, initial_state=None):
        """
        Plot error magnitudes in bootstrap simulation for assessing plant model quality
        :param plant: Plant object to simulate on
        :param length: episode length in float or integer
        :param controller: Controller with 'get_action' method
        :param initial_state: numpy.array of shape(nstates, 1) or None
        """
        plant_output = plant.get_feature_set().get_state_set()
        sim = plant.simulate(length, controller, initial_state=initial_state)

        time = sim.get_time_vector()
        states = [sim.get_states()[:, 0:1]]
        for _ in time:
            states.append(
                self.get_next_state(states[-1], controller.get_action(states[-1]))[0],
            )
        predictions = np.hstack(
            [self._output_feature_set.like_me(s, plant_output) for s in states]
        )[:, 1:-1]
        truth = np.hstack([self._output_feature_set.like_me(s, plant_output) for (s, _, _, _) in sim])[:, 1:]
        errors = predictions - truth

        fig, axes = plt.subplots(nrows=len(self._output_feature_set))
        for i in range(len(axes)):
            axes[i].plot(time[1:], errors[i])
            axes[i].set_ylabel(self._output_feature_set.get_names()[i])
            axes[i].legend(["Plant", "Model"])
        axes[-1].set_xlabel("Time [s]")
        return axes
