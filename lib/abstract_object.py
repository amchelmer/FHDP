import logging
import time

from abc import ABCMeta


class AbstractObject(object):
    """
    Abstract generic object for all other classes. Defines logger, hashing, and unique object ID, which is also unique 
    throughout time.
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        """
        
        """
        self.logger = self.set_logger()
        self._id = int(
            str(int(time.time())) + str(id(self))
        )

    def __repr__(self):
        return "<{}>".format(self.__class__.__name__)

    def __hash__(self):
        return hash(self._key())

    def __eq__(self, other):
        return isinstance(other, self.__class__) and hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        """
        Method for removing unpicklable logger before loading 
        """
        dictionary = self.__dict__.copy()
        try:
            del dictionary["logger"]
        except KeyError:
            pass
        return dictionary

    def __setstate__(self, state):
        """
        Method for restoring logger after loading 
        """
        self.__dict__ = state.copy()
        self.logger = self.set_logger()

    def _key(self):
        """
        Returns a tuple of (hashable) elements that describe current object.
        :return: Tuple with elements that make object unique 
        """
        raise NotImplementedError

    def set_logger(self):
        return logging.getLogger(self.__class__.__name__)

    def get_id(self):
        return self._id
