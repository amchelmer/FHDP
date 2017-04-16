import numpy as np
import pickle

from .kdtree_wrapper import KDTreeWrapper
from .key import Key
from .key_set import KeySet
from ..abstract_function_approximator import AbstractFunctionApproximator
from ...env import DUMP_PATH
from ...sets import FeatureSet
from ...tools.math_tools import saturate
from ...validation.format_validation import assert_same_length, assert_length
from ...validation.object_validation import assert_in, assert_not_in, assert_true
from ...validation.type_validation import assert_is_type, assert_type_in, assert_list_of_type


class LLRFunctionApproximator(AbstractFunctionApproximator):
    """
    Local function approximator y_hat_local = beta1*x1 + beta2*x2 + beta3*x3 + ... + betaN*xN. Stores values in a 
    memory. Several key-value stores are used as the memory. One of storing the value (value_memory) and one for 
    storing the ages (age_memory). The function approximator uses KDtrees for KNN search
    """
    UPDATE_METRICS = ["mean", "min"]

    def __init__(self, input_feature_set, output_feature_set, plant_feature_set, knn, max_memory, epsilon_p_feature):
        """
        Instantiates an LLRFunctionApproximator
        :param input_feature_set: FeatureSet representing the features of the function input
        :param output_feature_set: FeatureSet representing the features of the function output
        :param plant_feature_set: FeatureSet represeting the plant features
        :param knn: number of nearest neighbors
        :param max_memory: Maximum size of memory
        :param epsilon_p_feature: Tolerance for adding samples (per input dimension)
        """
        super(LLRFunctionApproximator, self).__init__()

        assert_is_type(knn, int)
        assert_is_type(max_memory, int)
        assert_is_type(input_feature_set, FeatureSet)
        assert_is_type(output_feature_set, FeatureSet)
        assert_is_type(plant_feature_set, FeatureSet)

        self._plant_feature_set = plant_feature_set
        self._plant_state_feature_set = plant_feature_set.get_state_set()
        self._input_feature_set = input_feature_set
        self._output_feature_set = output_feature_set
        self._knn = knn

        self._max_memory = max_memory
        self._epsilon_p_feature = epsilon_p_feature
        self._set_epsilon(self._epsilon_p_feature)

        self.tree = KDTreeWrapper(n_features=len(input_feature_set))

        self.age_memory = {}
        self.value_memory = {}

        self._rng = np.random.RandomState()

        self._training_mode_flag = True

        self.logger.info(
            "Instantiated {:s} (id:{:d}) with {:d} features and {:d} outputs".format(
                self.__class__.__name__,
                self.get_id(),
                len(self._input_feature_set),
                len(self._output_feature_set),
            )
        )

    def __len__(self):
        return len(self.value_memory)

    def __iter__(self):
        for key, value in sorted(self.value_memory.items(), key=lambda x: tuple(x[0])):
            yield key, value

    def _key(self):
        raise NotImplementedError

    def __getstate__(self):
        dictionary = self.__dict__.copy()
        try:
            del dictionary["logger"]
            del dictionary["tree"]
        except KeyError:
            pass
        return dictionary

    def __setstate__(self, dictionary):
        self.__dict__ = dictionary.copy()
        self.logger = self.set_logger()
        self.rebuild_tree()
        self.assert_consistent_memory()

    def __contains__(self, item):
        return self.value_memory.__contains__(item)

    def set_rng(self, i):
        """
        Set state for random number generator
        :param i: integer
        """
        self._rng = np.random.RandomState(i)

    def training(self):
        """
        Set mode to training. Function approximator can be trained.
        """
        self._training_mode_flag = True
        self.logger.info("Set to training mode")

    def evaluation(self):
        """
        Set mode to evaluation. Function approximator cannot be trained.
        """
        self._training_mode_flag = False
        self.logger.info("Set to evaluation mode")

    def get_feature_set(self):
        """
        :return: FeatureSet of the input features 
        """
        return self._input_feature_set

    def get_output_feature_set(self):
        """
        :return: FeatureSet of the output features 
        """
        return self._output_feature_set

    def get_zero_value(self):
        """
        :return: numpy.array with zeros of shape(noutputs, 1) 
        """
        return np.zeros((len(self._output_feature_set), 1))

    def set_max_memory_size(self, n):
        """
        Sets maximum memory size for the sample memory
        :param n: integer representing max number of samples
        """
        assert_type_in(n, [int, type(None)])
        self._max_memory = n if isinstance(n, int) else self._max_memory

    def _set_epsilon(self, e):
        """
        Set tolerance for deciding whether a sample should be added to memory
        :param e: tolerance as float 
        """
        self._epsilon = np.sqrt(e ** 2 * len(self._input_feature_set))

    @staticmethod
    def append_bias(array):
        """
        Adds row of ones for affine fit
        :param array: numpy array
        :return: array with row of ones on bottom
        """
        return np.vstack([
            array,
            np.ones((1, array.shape[1]), dtype=np.float64)
        ])

    def get_closest_points(self, query_point):
        """
        Get closest points to query point, including values and distances
        :param query_point: array of query point
        :return:
        """
        if self.tree.is_empty():
            return KeySet([]), np.array([[]])

        query_key = self.make_keys(query_point)
        self.logger.debug("Query key is {}".format(query_key))

        knn_key_set, dists = self.tree.get_knn(query_key, self._knn)
        self.logger.debug("Found {:d} nearest neighbors".format(len(knn_key_set)))

        knn_targets = self.lookup_values(knn_key_set)
        knn_features = self.unmake_keys(knn_key_set)

        for feature, target, dist in zip(knn_features.T, knn_targets.T, dists.flatten()):
            self.logger.debug("Neighbor: <{:s}: {:s}> at d={:.4f}".format(feature, target, dist))
        return knn_key_set, knn_targets

    def compute_distances(self, query_point, knn_key_set=None):
        """
        Compute distances of knn_key_set to query_point
        :param query_point: array with query_point
        :param knn_key_set: iterable with keys (Keys) of nearest neighbors
        :return:
        """
        knn_key_set = self.get_closest_points(query_point)[0] if knn_key_set is None else knn_key_set
        return np.linalg.norm(
            self.make_keys(query_point).get_array() - knn_key_set.aggregate(),
            ord=2,
            axis=0
        )

    def local_fit(self, features, targets):
        """
        Make local fit around features using n_samples.
        :param features: array of shape (n_features, k)
        :param targets: array of shape (n_outputs, k)
        :return: np.array of shape (n_outputs, n_features+1)
        """
        design_array = self.append_bias(features)
        return targets.dot(design_array.T).dot(np.linalg.pinv(design_array.dot(design_array.T)))

    def predict(self, beta, query_point):
        """
        Predicts result y = beta*x from linear regression parameters. Also saturates according to bounds.
        :param beta: fitted parameters of shape(noutputs, nfeatures+1)
        :param query_point: numpy.array of shape(nfeatures, 1)
        :return: numpy.array of shape(noutput, 1)
        """
        return saturate(
            beta.dot(self.append_bias(query_point)),
            self._output_feature_set.get_bounds()
        )

    def add_sample(self, query_features, value):
        """
        Adds single sample to memory
        :param query_features: numpy array
        :param value: array of shape (n_outputs, 1)
        """
        if self._training_mode_flag:
            key = self.make_keys(query_features)
            try:
                assert_not_in(key, self.value_memory)
            except AssertionError:
                self.logger.warning("Trying to add point {}, but its already in the memory".format(key))
                try:
                    self.assert_consistent_memory()
                    self.logger.warning("Memory still consistent")
                except AssertionError:
                    self.logger.warning("Memory has become inconsistent. Rebuilding tree")
                    self.rebuild_tree()
                return
            self.value_memory[key] = value
            self.reset_age(key)
            self.tree.append(key)
            self.logger.debug("Added sample <{:s}: {:s}> to memory".format(key, value.flatten()))

    def increment(self, key_set, deltas):
        """
        Increment samples
        :param key_set: key_set of values to be updated
        :param deltas: KeySet of same size as key_set, sorted in same order
        """
        if self._training_mode_flag:
            assert_list_of_type(key_set, Key)
            assert_length(key_set, deltas.shape[1])
            for key, delta in zip(key_set, deltas.T):
                self.value_memory[key] = saturate(
                    self.value_memory[key] + delta.reshape(-1, 1),
                    self._output_feature_set.get_bounds(),
                )

    def lookup_values(self, key_set):
        """
        Looks up value of k query points
        :param key_set: KeySet of length k
        :return: array of shape (n_outputs, k)
        """
        assert_is_type(key_set, KeySet)
        try:
            return np.hstack([self.value_memory[key] for key in key_set])
        except ValueError:
            return np.array([[]])

    def age_samples(self, dt):
        """
        Ages all samples in memory
        """
        if self._training_mode_flag:
            for key in self.age_memory.keys():
                self.age_memory[key] += dt
            self.logger.debug("Aged all samples in memory")

    def reset_age(self, keys):
        """
        Reset ages of keys to zero
        :param keys: key or keyset
        :return:
        """
        if self._training_mode_flag:
            assert_type_in(keys, [Key, KeySet])
            if isinstance(keys, Key):
                self.age_memory[keys] = 0.
            else:
                for key in keys:
                    self.age_memory[key] = 0.

    def _get_keys_sorted_by_age(self):
        """
        Return samples in memory, sorted by increasing age.
        :return: sorted keys, sorted ages
        """
        keys_sorted_by_age, sorted_ages = zip(
            *sorted(
                self.age_memory.items(),
                key=lambda x: x[1]
            )
        )
        return keys_sorted_by_age, sorted_ages

    def _purge_keys(self, keys_to_purge):
        """
        Purges a set of Keys from the memories
        :param keys_to_purge: iterable with Key objects
        :return: 
        """
        for key in keys_to_purge:
            self.logger.debug("Purged {} with age {:.1f} seconds".format(key, self.age_memory[key]))
            del self.age_memory[key], self.value_memory[key]

    def purge_by_age(self):
        """
        Purges old samples from memory
        """
        has_purged_flag = False
        if len(self.age_memory) > self._max_memory:
            keys_sorted_by_age, _ = self._get_keys_sorted_by_age()
            keys_to_purge = keys_sorted_by_age[self._max_memory:]
            self._purge_keys(keys_to_purge)
            self.logger.debug("Purged {:d} oldest items".format(len(keys_to_purge)))
            has_purged_flag = True
        return has_purged_flag

    def purge_randomly(self):
        """
        Purges random samples from memory
        """
        has_purged_flag = False
        if len(self.age_memory) > self._max_memory:
            sorted_keys, _ = zip(*self)
            keys_to_purge = [sorted_keys[r] for r in self._rng.choice(
                len(self.age_memory),
                len(self.age_memory) - self._max_memory,
                replace=False
            )]
            self._purge_keys(keys_to_purge)
            self.logger.debug("Purged {:d} samples randomly".format(len(keys_to_purge)))
            has_purged_flag = True
        return has_purged_flag

    def purge_by_weighted_age(self):
        """
       Purges samples from memory using a distribution weighted by age
       """
        has_purged_flag = False
        if len(self.age_memory) > self._max_memory:
            keys_sorted_by_age, sorted_ages = self._get_keys_sorted_by_age()
            total_age = float(sum(sorted_ages))
            keys_to_purge = [keys_sorted_by_age[r] for r in self._rng.choice(
                len(keys_sorted_by_age),
                replace=False,
                p=[a / total_age for a in sorted_ages],
                size=len(self.age_memory) - self._max_memory
            )]
            self._purge_keys(keys_to_purge)
            self.logger.debug("Purged {:d} samples weighted by age".format(len(keys_to_purge)))
            has_purged_flag = True
        return has_purged_flag

    def purge_older_than(self, age_threshold):
        """
        Purges all samples in memory with age >= age_threshold
        :param age_threshold: threshold in seconds as float
        """
        has_purged_flag = False
        assert_type_in(age_threshold, [float, int])
        keys_to_purge = [key for (key, age) in self.age_memory.items() if age >= age_threshold]
        if len(keys_to_purge) > 0:
            has_purged_flag = True
            self._purge_keys(keys_to_purge)
            self.logger.debug("Purged {:d} samples with age > {}".format(len(keys_to_purge), age_threshold))
        return has_purged_flag

    def assert_consistent_memory(self):
        """
        Ensures all memories are consistent, meaning all contain the same keys. 
        """
        assert_same_length(self, self.value_memory, self.age_memory, self.tree)
        for key, _ in self:
            assert_in(key, self.value_memory)
            assert_in(key, self.age_memory)
        self.logger.debug("Memory consistent: {0:d} samples remain".format(len(self)))

    def make_keys(self, array):
        """
        Make a hashable key from a query point of array of query points
        :param array: numpy.array
        :return: Key of KeySet (depending on number of columns in array)
        """
        assert_is_type(array, np.ndarray)
        key_array = self._input_feature_set.normalize(array)
        if key_array.shape[1] == 1:
            return Key(key_array)
        else:
            return KeySet([Key(key) for key in key_array.T])

    def unmake_keys(self, keys):
        """
        Make features from a KeySet or a Key
        :param keys: Key or KeySet object
        :return: numpy.array of shape(key length, number of keys)
        """
        assert_type_in(keys, [Key, KeySet])
        return self._input_feature_set.unnormalize(keys.aggregate())

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
        :return: LLRFunctionApproximator object
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

    def rebuild_tree(self):
        """
        Rebuild tree given the samples in the value memory
        """
        self.tree = KDTreeWrapper(
            n_features=len(self._input_feature_set)
        ) if len(self) == 0 else KDTreeWrapper(
            key_list=self.value_memory.keys()
        )
        self.logger.debug("Rebuilt tree ({:d} nodes)".format(len(self.tree)))

    def like_me(self, state):
        """
        Convenience method for describing a plant state by the FeatureSet of the LLR
        :param state: plant state as numpy.array of shape(nplantstates, 1) 
        :return: state as features of the LLR
        """
        features = self._input_feature_set.like_me(state, self._plant_state_feature_set)
        self.logger.debug("Input features are {}".format(features.flatten()))
        return features

    def change_input_feature(self, feature_change):
        """
        Change a feature of the input_feature_set
        :param feature_change: FeatureChange object
        """
        feature = feature_change.get_feature()
        method = feature_change.get_method()

        old_key_set, values = zip(*self)
        key_set_array = np.hstack([k.get_array() for k in old_key_set]).T
        ages = [self.age_memory[k] for k in old_key_set]

        if feature_change.is_add():
            index = self._input_feature_set.get_insert_position(feature)
            self.logger.info("Learning feature {:s}, method is {:s}".format(feature, method))

            if method == "zero":
                new_key_set = self._map_zero_initialization(key_set_array, index)

            elif method == "perturb-gauss":
                new_key_set = self._map_perturb(key_set_array, index, feature_change.get_spread())

            elif method == "clone-uniform":
                n_clones = int(self._max_memory / len(self))
                assert n_clones > 0, "At least one clone is required"
                self.logger.info("Creating {:d} clones w/ {} spread".format(
                    n_clones,
                    feature_change.get_spread(),
                ))
                new_key_set = self._map_sample_cloning_uniform(key_set_array, index, n_clones,
                                                               feature_change.get_spread())
                ages = [age for _ in range(n_clones) for age in ages]
                values = [value.copy() for _ in range(n_clones) for value in values]

            elif method == "clone-gauss":
                n_clones = int(self._max_memory / len(self))
                assert n_clones > 0, "At least one clone is required"
                self.logger.info("Creating {:d} clones w/ {} spread".format(
                    n_clones,
                    feature_change.get_spread(),
                ))
                new_key_set = self._map_sample_cloning_gauss(
                    key_set_array,
                    index,
                    n_clones,
                    feature_change.get_spread()
                )
                ages = [age for _ in range(n_clones) for age in ages]
                values = [value.copy() for _ in range(n_clones) for value in values]

            else:
                raise ValueError("Unknown learning method {}".format(method))

        elif feature_change.is_remove():
            index = self._input_feature_set.get_index(feature)
            self.logger.info("Forgetting feature {:s}, method is {:s}".format(feature, method))

            if method == "project":
                new_key_set = self._map_project(key_set_array, index)

            elif method == "threshold":
                self.logger.info("Applying a threshold of {:.2f}".format(feature_change.get_spread()))
                new_key_set, values, ages = self._map_threshold(
                    key_set_array,
                    index,
                    feature_change.get_spread(),
                    values,
                    ages
                )

            else:
                raise ValueError("Unknown forgetting method {}".format(method))

        else:
            raise ValueError("Logic error")

        self.value_memory.clear()
        self.age_memory.clear()
        if hasattr(self, "trace_memory"):
            self.trace_memory.clear()
        self.assert_empty_memory()
        assert_same_length(new_key_set, ages, values)

        for new_key, age, value in zip(new_key_set, ages, values):
            self.value_memory[new_key] = value
            self.age_memory[new_key] = age

        self.logger.info(
            "Performing feature change {} on feature set {}".format(feature_change, self._input_feature_set)
        )
        self._input_feature_set = feature_change.apply(self._input_feature_set)
        self._set_epsilon(self._epsilon_p_feature)
        self.rebuild_tree()
        self.assert_consistent_memory()

    @staticmethod
    def _map_zero_initialization(key_set_array, idx):
        """
        Map samples in memory to higher dimension by using 0 as value
        :param key_set_array: numpy.array of keys of shape(number of keys, length of keys)
        :param idx: position where new value should be added
        :return: KeySet of mapped keys
        """
        new_key_set_array = np.hstack(
            [key_set_array[:, :idx], np.zeros((key_set_array.shape[0], 1), dtype=np.float64), key_set_array[:, idx:]]
        )
        return KeySet([Key(arr) for arr in new_key_set_array])

    def _map_perturb(self, key_set_array, idx, s):
        """
        Map samples in memory to higher dimension by using a Gaussian random variable as value
        :param key_set_array: numpy.array of keys of shape(number of keys, length of keys)
        :param idx: position where new value should be added
        :param s: standard deviation of Gaussian distribution
        :return: KeySet of mapped keys
        """
        return self._map_sample_cloning_gauss(key_set_array, idx, 1, s)

    def _map_sample_cloning_uniform(self, key_set_array, idx, n_clones, s):
        """
        Map samples in memory to higher dimension by using a uniform random variable as value. Clones each sample a 
        few times as well.
        :param key_set_array: numpy.array of keys of shape(number of keys, length of keys)
        :param idx: position where new value should be added
        :param n_clones: number of clones to be made per sample
        :param s: bound of uniform distribution [-s, +s]
        :return: KeySet of cloned and mapped keys
        """
        key_set_array_cloned = np.tile(key_set_array, (n_clones, 1))
        new_key_set_array = np.hstack([
            key_set_array_cloned[:, :idx],
            self._rng.uniform(-s, s, (n_clones * key_set_array.shape[0], 1)).astype(np.float64),
            key_set_array_cloned[:, idx:]
        ])
        return KeySet([Key(arr) for arr in new_key_set_array])

    def _map_sample_cloning_gauss(self, key_set_array, idx, n_clones, s):
        """
        Map samples in memory to higher dimension by using a Gaussian random variable as value. Clones each sample a 
        few times as well.
        :param key_set_array: numpy.array of keys of shape(number of keys, length of keys)
        :param idx: position where new value should be added
        :param n_clones: number of clones to be made per sample
        :param s: standard deviation of Gaussian distribution
        :return: KeySet of cloned and mapped keys
        """
        key_set_array_cloned = np.tile(key_set_array, (n_clones, 1))
        new_key_set_array = np.hstack([
            key_set_array_cloned[:, :idx],
            s * self._rng.randn(n_clones * key_set_array.shape[0], 1).astype(np.float64),
            key_set_array_cloned[:, idx:]
        ])
        return KeySet([Key(arr) for arr in new_key_set_array])

    @staticmethod
    def _map_project(key_set_array, idx):
        """
        Map all samples in memory to a lower dimension by removing variable idx
        :param key_set_array: numpy.array of keys of shape(number of keys, length of keys)
        :param idx: position of the variable to be removed
        :return: KeySet of mapped keys
        """
        new_key_set_array = np.hstack([key_set_array[:, :idx], key_set_array[:, idx + 1:]])
        return KeySet([Key(arr) for arr in new_key_set_array])

    @staticmethod
    def _map_threshold(key_set_array, idx, threshold, values, ages):
        """
        Map all samples in memory to a lower dimension by removing variable idx
        :param key_set_array: numpy.array of keys of shape(number of keys, length of keys)
        :param idx: position of the variable to be removed
        :param threshold: cut-off value for keeping samples
        :param values: list of values with same sorting as ages and key_set_array
        :param ages: list of ages with same sorting as values and key_set_array
        :return: KeySet of mapped keys
        """
        selected_old_key_set, new_values, new_ages = zip(
            *[(k, value, age) for (k, value, age) in zip(key_set_array, values, ages) if abs(k[idx]) <= threshold]
        )
        selected_key_set_array = np.array(selected_old_key_set)
        new_key_set_array = np.hstack([selected_key_set_array[:, :idx], selected_key_set_array[:, idx + 1:]])
        return (
            KeySet([Key(arr) for arr in new_key_set_array]),
            new_values,
            new_ages
        )

    def assert_empty_memory(self):
        """
        Ensures all memories are empty 
        """
        assert_true(
            len(self.value_memory) == len(self.age_memory) == 0,
            "Non-empty memory: Value memory: {:d}, age memory: {:d}".format(
                len(self.value_memory),
                len(self.age_memory)
            )
        )
