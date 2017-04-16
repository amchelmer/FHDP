import numpy as np

from .key import Key
from ...sets import AbstractSet
from ...validation.format_validation import assert_same_length
from ...validation.type_validation import assert_list_of_types, assert_type_in


class KeySet(AbstractSet):
    """
    Set wrapper object for a list of Key objects. Allows aggregation in numpy.array
    """

    def __init__(self, key_list):
        assert_type_in(key_list, [list, tuple])
        assert_list_of_types(key_list, [Key, tuple, np.ndarray])
        if len(key_list) > 0:
            assert_same_length(*key_list)
        super(KeySet, self).__init__([t if isinstance(t, Key) else Key(t) for t in key_list])

    def __repr__(self):
        return "<{} of {} keys>".format(self.__class__.__name__, len(self._iterable))

    def _key(self):
        return tuple([tuple(t) for t in self])

    def aggregate(self):
        """
        Aggregates Keys in KeySet in a numpy.array. Useful for fitting the linear regression.
        :return: numpy.array of shape(key_length, number of keys)
        """
        try:
            return np.hstack([t.get_array() for t in self])
        except IndexError:
            return np.array([[]])
