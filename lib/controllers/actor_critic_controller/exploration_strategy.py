import numpy as np
import pandas as pd

from ...abstract_object import AbstractObject
from ...validation.object_validation import assert_in, assert_unique
from ...validation.type_validation import assert_is_type

from copy import deepcopy


class ExplorationStrategy(AbstractObject):
    """
    Zero order hold exploration strategy
    """

    def __init__(self, exploration_dict):
        super(ExplorationStrategy, self).__init__()
        assert_unique(exploration_dict.keys())
        assert_in(1, exploration_dict.keys())
        self._range = max(exploration_dict.keys())
        self._series = pd.Series(
            pd.Series(exploration_dict).reindex(range(1, self._range + 1)).ffill().astype(np.int)
        )

    def __repr__(self):
        s = self._series.drop_duplicates()
        return "<{:s} [{:s}]>".format(
            self.__class__.__name__,
            "".join(["{}: {}, ".format(i, v) for i, v in zip(s.index, s)])[:-2]
        )

    def __getitem__(self, item):
        assert_is_type(item, int)
        return self._series.__getitem__(min(item, self._range))

    def _key(self):
        return tuple(self._series.iteritems())

    def update(self, d):
        new_dict = dict(self._series.drop_duplicates())
        new_dict.update(d)

        assert_is_type(d, dict)
        assert min(d.keys()) >= self._range, "Cannot update exploration in the past"
        assert_unique(new_dict.keys())

        return ExplorationStrategy(new_dict)

    def copy(self):
        return deepcopy(self)
