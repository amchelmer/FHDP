import numpy as np

from ..abstract_object import AbstractObject
from ..simulations import Simulation
from ..tools.math_tools import saturate, center_mod
from ..validation.object_validation import assert_in


class AbstractPlant(AbstractObject):
    """
    Abstract class for plants. Defines functions for computing next states, and simulating episodes given a controller.
    """
    _VISUALIZER_CLS = None
    _FEATURE_SET = None
    _MOD = None
    INTEGRATORS = ["euler", "rk4"]

    def __init__(self, time_step, integrator, feature_set, init_mean, init_std):
        super(AbstractPlant, self).__init__()
        assert_in(integrator.lower(), self.INTEGRATORS)
        self._integrator_type = integrator
        self._time_step = time_step
        self._feature_set = feature_set
        self._bounds = self._feature_set.get_state_set().get_bounds()
        self._state_modulus = self._MOD
        self._initial_state_mean = init_mean
        self._initial_state_sigma = init_std

        self.rng = np.random.RandomState()

        self.logger.info(
            "{:s} with {:d} states {:s} and {:d} inputs {:s}".format(
                self.__class__.__name__,
                self.get_number_of_state_vars(),
                ", ".join(self.get_state_labels()),
                self.get_number_of_action_vars(),
                ", ".join(self.get_action_labels()),
            )
        )

    def _key(self):
        raise NotImplementedError

    @classmethod
    def get_default_feature_set(cls):
        """
        :return: Returns the default FeatureSet 
        """
        return cls._FEATURE_SET

    def get_feature_set(self):
        """
        :return: Returns FeatureSet 
        """
        return self._feature_set

    def get_visualizer_cls(self):
        """
        Visualizer class for visualizing Simulations with current plant
        :return: Class
        """
        return self._VISUALIZER_CLS

    def set_mod(self, flag):
        """
        Turn on modulation
        :param flag: boolean
        """
        self._state_modulus = self._MOD if flag else None

    def set_rng(self, i):
        """
        Set random number generator for reproducable experiments.
        :param i: integer
        """
        self.rng = np.random.RandomState(i)

    def get_time_step(self):
        """
        :return: Return time step as float 
        """
        return self._time_step

    def get_bounds(self):
        """
        :return: Return state bounds in np.array of shape(nstates, 2) 
        """
        return self._bounds

    def get_state_modulus(self):
        """
        :return: Return modulus for modulating states 
        """
        return self._state_modulus

    def get_number_of_state_vars(self):
        """
        :return: Return number of state variables as int 
        """
        return len(self._feature_set.get_state_set())

    def get_number_of_action_vars(self):
        return len(self._feature_set.get_action_set())

    def get_initial_state(self, s=1.):
        """
        Process for getting an initial state. Can be stochastic or deterministic
        :return: np.array of shape (ns, 1)
        """
        random = s * self.rng.randn(self.get_number_of_state_vars(), 1)
        initial_state = center_mod(
            self._initial_state_mean + self._initial_state_sigma * random,
            self._state_modulus
        )
        self.logger.debug("Initial state: {:s}".format(initial_state.flatten()))
        return initial_state

    def get_state_labels(self):
        """
        :return: Returns a list of state labels 
        """
        return self._feature_set.get_state_set().get_names()

    def get_action_labels(self):
        """
        :return: Returns a list of action labels 
        """
        return self._feature_set.get_action_set().get_names()

    def compute_derivative(self, state, action):
        """
        Undefined method for computing the time-derivative of the state variables. Needs to be implemented in plants.
        :param state: np.array of shape(nstates, 1)
        :param action: np.array of shape(naction, 1)
        :return: derivatives in np.array of shape(nstates, 1)  
        """
        raise NotImplementedError

    def _get_next_state_rk4(self, state, action):
        """
        Worker method for computing the next state using the fourth-order Runge-Kutta integration method.
        :param state: np.array of shape(nstates, 1)
        :param action: np.array of shape(naction, 1)
        :return: np.array of shape(nstates, 1) 
        """
        k1 = self._time_step * self.compute_derivative(state, action)
        k2 = self._time_step * self.compute_derivative(state + 0.5 * k1, action)
        k3 = self._time_step * self.compute_derivative(state + 0.5 * k2, action)
        k4 = self._time_step * self.compute_derivative(state + 1.0 * k3, action)
        return state + (k1 + 2.0 * (k2 + k3) + k4) / 6.0

    def _get_next_state_euler(self, state, action):
        """
        Worker method for computing the next state using Euler integration.
        :param state: np.array of shape(nstates, 1)
        :param action: np.array of shape(naction, 1)
        :return: np.array of shape(nstates, 1) 
        """
        return state + self._time_step * self.compute_derivative(state, action)

    def get_next_state(self, state, action):
        """
        Wrapper method for computing next state. Will delegate to proper integrator function. The method first calls 
        preprocess_state_action and after computing next_state, calls postprocess_next_state on the next state before 
        returning it.
        :param state: np.array of shape(nstates, 1)
        :param action: np.array of shape(naction, 1)
        :return: np.array of shape(nstates, 1)
        """
        func = self._get_next_state_euler if self._integrator_type.lower() == "euler" else self._get_next_state_rk4
        next_state = func(
            *self.preprocess_state_action(state, action)
        )
        post_processed_next_state = self.postprocess_next_state(next_state)
        if np.isnan(post_processed_next_state).any():
            raise ValueError("Next state contains a nan: {}".format(
                post_processed_next_state.flatten())
            )
        return post_processed_next_state

    def preprocess_state_action(self, state, action):
        """
        Function that is called before the get_next_state method.
        :param state: np.array of shape(nstates, 1)
        :param action: np.array of shape(naction, 1)
        :return: np.array of shape(nstates, 1) and np.array of shape(naction, 1)
        """
        return state, action

    def postprocess_next_state(self, next_state):
        """
        Function that is called at the end of the get_next_state method. Can be used e.g. to incorporate ground 
        saturation.
        :param next_state: np.array of shape(nstates, 1)
        :return: np.array of shape(nstates, 1)
        """
        return next_state

    def like_me(self, state, other_feature_set):
        """
        Convenience method for representing states according to Plant FeatureSet.
        :param state: np.array of shape (n, 1)
        :param other_feature_set: FeatureSet of length n
        :return: np.array of shape(len(self), 1) 
        """
        return self._feature_set.like_me(state, other_feature_set)

    def simulate(self, length, controller, initial_state=None):
        """
        Creates a Simulation by having controller interact with plant.
        :param length: length of Simulation in integer or float
        :param controller: Controller object with 'get_action' method.
        :param initial_state: np.array of shape(nstates, 1) or None
        :return: Simulation object
        """
        states, actions = [], []
        state = self.get_initial_state() if initial_state is None else initial_state
        controller.reset()

        for _ in np.arange(0., length, self.get_time_step()):
            action = controller.get_action(state)
            states.append(state)
            actions.append(action)
            if self.is_out_of_bounds(state):
                break
            else:
                state = center_mod(
                    self.get_next_state(state, action),
                    self.get_state_modulus()
                )

        return Simulation(
            self,
            np.hstack(states),
            np.hstack(actions),
            values=controller.get_critic() if hasattr(controller, "get_critic") else None,
            rewards=controller.get_reward_function() if hasattr(controller, "get_reward_function") else None,
        )

    def is_out_of_bounds(self, state):
        """
        Checks whether a state is out of bounds according to FeatureSet bounds.
        :param state: np.array of shape(nstates, 1)
        :return: boolean
        """
        is_out_of_bounds = not np.isclose(
            state,
            saturate(state, self._bounds)
        ).all()
        if is_out_of_bounds:
            self.logger.warning("System out of bounds in state {}".format(state.flatten()))
        return is_out_of_bounds
