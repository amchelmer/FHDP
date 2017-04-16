import numpy as np
from sklearn.neighbors import KDTree

from .key_set import KeySet
from ...abstract_object import AbstractObject
from ...validation.object_validation import assert_true


class KDTreeWrapper(AbstractObject):
    """
    Wrapper object around the KDTree class of sklearn. Integrates use of keys, adding samples, rebuilding and 
    dealing with empty trees.
    """

    def __init__(self, key_list=None, n_features=None):
        super(KDTreeWrapper, self).__init__()

        if key_list is None or key_list == []:
            assert_true(n_features is not None, "Supply data or n_features")
            self._key_list = []
            self._tree = None
        else:
            self._key_list = key_list
            self._rebuild()

    def __len__(self):
        return self._key_list.__len__()

    def _key(self):
        raise NotImplementedError

    def get_knn(self, query_key, k):
        """
        Get k nearest neighbors for a query key 
        :param query_key: Key object with query point
        :param k: number of nearest neighbors
        :return: KeySet and numpy.array with distances to query key
        """
        if self.is_empty():
            dists = np.array([])
            keys = KeySet([])
        else:
            dists, indices = self._tree.query(
                query_key.get_array().T,
                k=min(k, len(self)),
            )
            keys = KeySet([self._key_list[i] for i in indices.flatten()])
        return keys, dists

    def _rebuild(self):
        """
        Rebuilds tree from samples in tree. 
        """
        self._tree = KDTree(np.hstack([key.get_array() for key in self._key_list]).T)

    def is_empty(self):
        """
        :return: Boolean representing empty tree. 
        """
        return self._key_list == []

    def append(self, key):
        """
        Add sample to KD-Tree
        :param key: KeySet object
        """
        self._key_list.append(key)
        self._rebuild()
