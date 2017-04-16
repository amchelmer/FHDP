import numpy as np

from .. import AbstractSet
from ...features.feature import Feature
from ...validation.format_validation import assert_length, assert_same_length
from ...validation.object_validation import assert_true
from ...validation.type_validation import assert_is_type, assert_list_of_type

from copy import deepcopy


class FeatureSet(AbstractSet):
    """
    Wrapper class built around a list of Features. Plants and function approximators can use FeatureSets select features 
    from state and action vectors.   
    """

    def __init__(self, feature_list):
        """
        Instantiates a FeatureSet
        :param feature_list: python list of Feature objects
        """
        self._validate_list(feature_list)
        super(FeatureSet, self).__init__(feature_list)

        self._scales = np.array([f.get_scale() for f in self]).reshape(-1, 1)

        self.n_states = len([f for f in self if f.is_state()])
        self.n_actions = len([f for f in self if f.is_action()])

    def __repr__(self):
        return "<{}: {:s}>".format(self.__class__.__name__, " ".join([str(f) for f in self]))

    def _key(self):
        return tuple(self._iterable)

    @classmethod
    def _validate_list(cls, feature_list):
        """
        Validates feature list. Checks for consistency in order, and for duplicates.
        :param feature_list: list of Feature objects
        """
        assert_list_of_type(feature_list, Feature)
        name_list = [x.get_name() for x in feature_list]
        assert_same_length(name_list, set(name_list))
        type_list = [f.get_type() for f in feature_list]
        assert_true(type_list == sorted(type_list, reverse=True), "Set not sorted properly")

    def get_index(self, feature):
        """
        :param feature: Feature object 
        :return: index of Feature in FeatureSet
        """
        assert_is_type(feature, Feature)
        return self._iterable.index(feature)

    def get_scales(self):
        """
        :return: Returns a 1D np.array with scales 
        """
        return self._scales

    def get_names(self):
        """
        :return: Returns a list with names 
        """
        return [f.get_name() for f in self]

    def get_bounds(self):
        """
        :return: Returns an np.array of shape (n_features, 2) with bounds 
        """
        return np.array([f.get_bounds() for f in self])

    def get_state_set(self):
        """
        :return: Returns a FeatureSet with only the state Features 
        """
        return FeatureSet([f for f in self if f.is_state()])

    def get_action_set(self):
        """
        :return: Returns a FeatureSet with only the action Features 
        """
        return FeatureSet([f for f in self if f.is_action()])

    def normalize(self, array):
        """
        Normalizes an array using the scales.
        :param array: array of shape (len(FeatureSet), 1)
        :return: array of same shape
        """
        return np.divide(array, self._scales)

    def unnormalize(self, array):
        """
        Rescales an array using the scales.
        :param array: array of shape (len(FeatureSet), 1)
        :return: array of same shape
        """
        return np.multiply(array, self._scales)

    def get_insert_position(self, feature):
        """
        Finds insert position for a new feature. State features are added after the last state, 
        action features at the end.
        :param feature: Feature object
        :return: integer representing index
        """
        assert_is_type(feature, Feature)
        return len(self) if feature.is_action() else self.n_states

    def like_me(self, state, other_feature_set):
        """
        Reformats a state that represent another FeatureSet as the current FeatureSet.
        :param state: np.array of shape (n, 1)
        :param other_feature_set: FeatureSet of length n
        :return: np.array of shape(len(self), 1)
        """
        assert_length(other_feature_set, state.shape[0])
        state = state.flatten()
        return np.array(
            [state[other_feature_set.get_index(feature)] if feature in other_feature_set else 0. for feature in self]
        ).reshape(-1, 1)

    def copy(self):
        """
        :return: Returns a deepcopy of FeatureSet 
        """
        return deepcopy(self)
