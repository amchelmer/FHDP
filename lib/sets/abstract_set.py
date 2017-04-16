import pickle

from ..abstract_object import AbstractObject
from ..env import DUMP_PATH
from ..validation.type_validation import assert_type_in


class AbstractSet(AbstractObject):
    """
    Abstract wrapper object for sets. Adds support for iteration, item getting, hashing, and dumping and loading. Child
     classes are the SimulationSet, RewardSet, FeatureSet and ControllerSet
    """

    def __init__(self, iterable):
        super(AbstractSet, self).__init__()
        assert_type_in(iterable, [list, tuple])
        self._iterable = iterable

    def __len__(self):
        return self._iterable.__len__()

    def __iter__(self):
        for s in self._iterable:
            yield s

    def __getitem__(self, item):
        items = self._iterable.__getitem__(item)
        if isinstance(items, list) or isinstance(items, tuple):
            return self.__class__(items)
        else:
            return items

    def __contains__(self, item):
        return self._iterable.__contains__(item)

    def _key(self):
        return tuple(self._iterable)

    def get_iterable(self):
        """
        :return: returns underlying iterable 
        """
        return self._iterable

    def dump(self, file_handle=None):
        """
        Dumps object to file in DUMP_PATH, requires hashable object
        :param file_handle: python file handle object
        :return: 
        """
        if not isinstance(file_handle, file):
            file_handle = open(
                "{:s}{:s}-{:d}.pckl".format(
                    DUMP_PATH,
                    self.__class__.__name__,
                    self.get_id()
                ),
                "wb")
        pickle.dump(self, file_handle)

    @classmethod
    def load(cls, file_id):
        """
        Load object from DUMP_PATH
        :param file_id: python file handle object
        :return: Collection object
        """
        if not isinstance(file_id, file):
            handle = open(
                "{:s}{:s}-{:d}.pckl".format(
                    DUMP_PATH,
                    cls.__name__,
                    file_id
                ),
                "rb")
        else:
            handle = file_id
        return pickle.load(handle)
