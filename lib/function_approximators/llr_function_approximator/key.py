import numpy as np

from ...abstract_object import AbstractObject
from ...validation.type_validation import assert_type_in


class Key(AbstractObject):
    """
    The Key object is a hashable, scaled version of a state or state/action combination, that can be used for 
    key/value store.
    """
    FORM = "%.5f"
    PRECISION = 14

    def __init__(self, key):
        super(Key, self).__init__()
        assert_type_in(key, [np.ndarray, tuple, list])
        array = np.round(
            np.matrix(key, dtype=np.float64),
            self.PRECISION
        )
        self._array = array.reshape(array.size, 1)

    def __repr__(self):
        return "<{}: [{}]>".format(self.__class__.__name__, ", ".join([self.FORM % x for x in self._array]))

    def __str__(self):
        return "[{}]".format(", ".join([self.FORM % x for x in self._array]))

    def __len__(self):
        return self._array.shape[0]

    def __iter__(self):
        for x in self._array.flatten():
            yield x

    def _key(self):
        return tuple(self._array.flatten())

    def get_array(self):
        """
        :return: Returns underlying array 
        """
        return self._array

    def aggregate(self):
        """
        Convenience method so same calls can be made on Key and KeySet
        :return: 
        """
        return self.get_array()
