import numpy as np

from ..abstract_object import AbstractObject
from ..tools.math_tools import hashable
from ..validation.object_validation import assert_in
from ..validation.format_validation import assert_shape
from ..validation.type_validation import assert_is_type, assert_type_in


class Feature(AbstractObject):
    """
    The Feature object describes a state or action variable for a state-space system. Together with other features it 
    can form a FeatureSet. Plants and function approximators can use FeatureSets select features from state and 
    action vectors.   
    """
    FEATURE_TYPES = ["state", "action"]

    def __init__(self, name, feature_type="state", scale=1., derivative=False, bounds=np.array([-np.inf, np.inf])):
        assert_is_type(name, str)
        assert_is_type(bounds, np.ndarray)
        assert_type_in(scale, [int, float])
        assert_is_type(derivative, bool)
        assert_in(feature_type, self.FEATURE_TYPES)
        assert_shape(bounds, (2,))

        super(Feature, self).__init__()
        self._name = name
        self._scale = float(scale)
        self._derivative = derivative
        self._type = feature_type
        self._bounds = bounds

        if self.is_action():
            assert derivative is False, "Action features cannot be derivatives"

    def __repr__(self):
        return "<{}: {}>".format(self.__class__.__name__, self._name)

    def __str__(self):
        return "Feature {:s}".format(self._name)

    def _key(self):
        return self._name, self._scale, self._type, self._derivative, hashable(self._bounds)

    def get_name(self):
        """
        :return: Returns Feature name 
        """
        return self._name

    def get_scale(self):
        """
        :return: Returns scale as a float 
        """
        return self._scale

    def get_bounds(self):
        """
        :return: Return np.array of shape (2,) with Feature bounds 
        """
        return self._bounds

    def get_type(self):
        """
        :return: Returns Feature type as a string. Either 'add' or 'remove' 
        """
        return self._type

    def is_state(self):
        """
        :return: Boolean, True if Feature is a state, False if it is an action 
        """
        return self._type == "state"

    def is_action(self):
        """
        :return: Boolean, False if Feature is a state, True if it is an action 
        """
        return not self.is_state()

    def is_derivative(self):
        """
        :return: Boolean, True if Feature is a time-derivative 
        """
        return self._derivative
