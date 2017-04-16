from copy import deepcopy

from ..abstract_object import AbstractObject
from ..validation.object_validation import assert_in, assert_not_in


class FeatureChange(AbstractObject):
    """
    The FeatureChange describes a change in a FeatureSet. In contains the feature, the method, and method parameters. 
    Its main method is apply, which applies the FeatureChange to a given FeatureSet.
    """
    LEARNING_METHODS = ["zero", "perturb-gauss", "clone-uniform", "clone-gauss"]
    FORGET_METHODS = ["project", "threshold"]

    def __init__(self, feature, method, spread=0):
        super(FeatureChange, self).__init__()
        self._feature = feature

        assert_in(method, self.LEARNING_METHODS + self.FORGET_METHODS)

        self._kind = "add" if method in self.LEARNING_METHODS else "remove"

        self._method = method if self.is_state() else None
        self._spread = spread if self.is_state() else None

    def __repr__(self):
        return "<FeatureChange {:s} {:s} using {:s}-method with spread {}>".format(
            self._kind.capitalize(),
            self._feature,
            self._method,
            self._spread
        )

    def _key(self):
        return self._kind, self._feature, self._method, self._spread

    def get_feature(self):
        """
        :return: Returns Feature 
        """
        return self._feature

    def get_spread(self):
        """
        :return: Returns spread parameter 
        """
        return self._spread

    def get_method(self):
        """
        :return: Returns method 
        """
        return self._method

    def is_add(self):
        """
        :return: Returns boolean, which is True when adding a Feature, and False when removing one. 
        """
        return self._kind == "add"

    def is_remove(self):
        """
        :return: Returns boolean, which is True when removing a Feature, and False when adding one. 
        """
        return not self.is_add()

    def is_state(self):
        """
        :return: Returns boolean, which is True when Feature is a state, and False when it is an action. 
        """
        return self._feature.is_state()

    def is_action(self):
        """
        :return: Returns boolean, which is True when Feature is an action, and False when it is a state. 
        """
        return self._feature.is_action()

    def apply(self, feature_set):
        """
        Applies itself to a FeatureSet.
        :param feature_set: FeatureSet object.
        :return: FeatureSet with FeatureChange incorporated.
        """
        new_list = deepcopy(feature_set.get_iterable())

        if self.is_add():
            assert_not_in(self._feature, feature_set)
            new_list.insert(feature_set.get_insert_position(self._feature), self._feature)
        elif self.is_remove():
            assert_in(self._feature, feature_set)
            new_list.pop(feature_set.get_index(self._feature))
        else:
            raise ValueError("Logic error")
        return feature_set.__class__(new_list)
