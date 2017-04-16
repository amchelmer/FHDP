import numpy as np
import pickle

from ..abstract_controller import AbstractController
from ...sets.simulation_set import SimulationSet
from ...env import DUMP_PATH
from ...tools.math_tools import center_mod
from ...simulations import Simulation, SimulationResult
from ...validation.object_validation import assert_in
from ...validation.type_validation import assert_type_in


class ActorCriticController(AbstractController):
    """
    The ActorCriticController class is a controller that is able to learn from interaction with the environment (plant).
    It requires an actor, critic, plant model and some learning parameters. Is able to train through train() method.
    """
    AGE_THRESHOLD = 45.  # seconds
    PURGE_METHODS = ["random", "age-weighted", "age"]
    ADD_METHODS = ["mean", "min"]

    def __init__(self, actor, critic, plant_model, plant, episode_length, add_method, purge_method, exploration_strategy,
                 seed):
        super(ActorCriticController, self).__init__(actor.get_output_feature_set().get_bounds())

        assert_in(purge_method, self.PURGE_METHODS)
        assert_in(add_method, self.ADD_METHODS)
        assert_type_in(episode_length, [int, float])

        self._actor = actor
        self._critic = critic
        self._plant_model = plant_model
        self._plant = plant
        self._add_method = add_method
        self._purge_method = purge_method
        self._exploration_strategy = exploration_strategy

        self._train_simulation_results = []

        self._episodes = 0
        self._length = episode_length

        self._seed = seed
        self.reset()

    def _key(self):
        raise NotImplementedError

    def get_action(self, state):
        """
        Compute the action given a state
        :param state: numpy.array
        :return: numpy.array of shape(naction, 1)
        """
        return self._actor.get_action(state)[0]

    def reset(self):
        """
        Resets the controller to the original state
        """
        self._actor.set_rng(self._seed + 1)
        self._critic.set_rng(self._seed + 2)
        self._plant_model.set_rng(self._seed + 3)
        self._plant.set_rng(self._seed + 4)

    def get_critic(self):
        """
        :return: Returns critic 
        """
        return self._critic

    def get_reward_function(self):
        """
        :return: Returns reward function 
        """
        return self._critic.get_reward_function()

    def get_plant(self):
        """
        :return: Returns plant 
        """
        return self._plant

    def get_plant_model(self):
        """
        :return: Returns plant model 
        """
        return self._plant_model

    def get_train_results(self):
        """
        Returns a list of training results of the last training episodes
        :return: list object
        """
        return self._train_simulation_results

    def get_last_results(self, lookback, look_back_metric="median"):
        """
        Computes aggregate training result of last training episodes 
        :param lookback: number of episodes to lookback
        :param look_back_metric: metric for aggregating SimulationResult objects
        :return: SimulationResult
        """
        lookback = min(lookback, len(self._train_simulation_results))
        return SimulationResult(self._train_simulation_results[-lookback:], metric=look_back_metric)

    def set_memory_sizes(self, actor_size, critic_size, pm_size):
        """
        Set memory sizes of actor, critic and plant model
        :param actor_size: Integer representing new actor memory size
        :param critic_size: Integer representing new critic memory size
        :param pm_size: Integer representing new plant model memory size
        """
        self._actor.set_max_memory_size(actor_size)
        self._critic.set_max_memory_size(critic_size)
        self._plant_model.set_max_memory_size(pm_size)

    def purge_and_rebuild(self, approximator, method):
        """
        Purges samples in memory and rebuilds KD-Trees
        :param approximator: 
        :param method: 
        :return: 
        """
        has_purged = approximator.purge_older_than(self.AGE_THRESHOLD)

        if method == "age":
            has_purged_two = approximator.purge_by_age()
        elif method == "age-weighted":
            has_purged_two = approximator.purge_by_weighted_age()
        else:
            has_purged_two = approximator.purge_randomly()
        if has_purged or has_purged_two:
            approximator.rebuild_tree()

    def change_feature(self, feature_change):
        """
        Change feature over all components of the ActorCritic; the actor, critic and plant model.
        :param feature_change: FeatureChange object
        """
        assert_in(feature_change.get_feature(), self.get_plant().get_feature_set())
        self.logger.info("Changing feature space with {:s}".format(feature_change))
        self._critic.clear_traces()

        if feature_change.is_state():
            self._actor.change_input_feature(feature_change)
            self._critic.change_input_feature(feature_change)
        else:
            self._actor.change_output_feature(feature_change)
        self._plant_model.change_input_feature(feature_change)

        self.purge_and_rebuild(self._actor, self._purge_method)
        self.purge_and_rebuild(self._critic, self._purge_method)
        self.purge_and_rebuild(self._plant_model, self._purge_method)

        self._actuator_limits = self._actor.get_output_feature_set().get_bounds()

    def train(self, n_episodes, train_hold=0):
        """
        Trains the ActorCriticController
        :param n_episodes: Number of training episodes
        :param train_hold: How many episodes are run without updating the actor and critic, before starting actual 
        training. Plant model is updated in the meantime.
        """
        self._actor.evaluation()
        self._critic.evaluation()

        for episode in range(self._episodes + 1, self._episodes + n_episodes + 1):
            if episode == train_hold + self._episodes + 1:
                self._actor.training()
                self._critic.training()

            self.logger.info(
                "Starting episode {:d}/{:d} with exploration every {:d} steps".format(
                    episode,
                    self._episodes + n_episodes,
                    self._exploration_strategy[episode],
                )
            )

            simulation_output = zip(*self._state_generator(self._exploration_strategy[episode]))
            self._train_simulation_results.append(
                SimulationResult(
                    Simulation(self.get_plant(), *map(np.hstack, simulation_output)),
                    metric=self.get_reward_function()
                )
            )

            self.logger.info(
                "{:d}: Terminated ep. {:d}/{:d} in state {:s} with total reward {:+.1f}".format(
                    self.get_id(),
                    episode,
                    n_episodes,
                    self._train_simulation_results[-1].get_last_state().flatten(),
                    self._train_simulation_results[-1].get_cum_reward(),
                )
            )
            self.logger.debug(
                "{:d}: Memory sizes: Actor: {:d}, Critic: {:d}, Process model: {:d}".format(
                    self.get_id(),
                    len(self._actor),
                    len(self._critic),
                    len(self._plant_model)
                )
            )

            self.purge_and_rebuild(self._actor, self._purge_method)
            self.purge_and_rebuild(self._critic, self._purge_method)
            self.purge_and_rebuild(self._plant_model, self._purge_method)

        self._episodes += n_episodes

    def _state_generator(self, exploration_interval):
        """
        State generator method for training method
        :param exploration_interval: exploration interval in integer
        :return: state, action, value, reward
        """
        plant = self.get_plant()
        dt = plant.get_time_step()
        self._critic.clear_traces()
        next_state = self.get_plant().get_initial_state()

        for t in np.arange(0, self._length, dt):
            state = next_state.copy()
            self.logger.debug("\n{:s}\nTime step: {:+.2f}, state: {}".format("#" * 90, t, state.flatten()))

            # Compute action
            action, action_keys = self._actor.get_action(state)
            self._actor.reset_age(action_keys)

            # Get reward
            next_reward = self.get_reward_function().get_reward(state, action)

            # Observe next state
            state_prime = plant.get_next_state(state, action)
            self.logger.debug("State' is {:s} for action {:s}".format(state_prime.flatten(), action.flatten()))

            # Compute value
            value, critic_keys, _ = self._critic.get_value(state)
            self._critic.reset_trace(critic_keys)
            self._critic.reset_age(critic_keys)

            # Predict state prime and get dfprime du
            predicted_state_prime, plant_keys, dfprime_du = self._plant_model.get_next_state(state, action)
            self._plant_model.reset_age(plant_keys)

            # Compute TD error
            next_value, _, dv_dfprime = self._critic.get_value(predicted_state_prime)
            td_error = self._critic.compute_td_error(next_reward, value, next_value)

            # Update plant model
            self._plant_model.update(state, action, predicted_state_prime, state_prime)

            # Update critic
            self._critic.update(state, value, td_error, critic_keys, self._add_method)

            # Update actor
            dr_du = self.get_reward_function().get_derivative_to_action(action)
            dr_daction_features = self._actor.action_like_me(dr_du)
            dv_du = dv_dfprime.dot(dfprime_du).T
            delta = dr_daction_features + self._critic.get_discount() * dv_du
            self.logger.debug("dv/du: {} for state {:s}".format(dv_du.flatten(), state.flatten()))
            self._actor.update(state, action, delta, action_keys, self._add_method)

            # Next step
            do_explore = int(t / dt) % exploration_interval == 0 and t != 0
            action_features = self._actor.perturb_action(
                self._actor.action_like_me(action)
            ) if do_explore else self._actor.action_like_me(action)

            next_state = center_mod(
                plant.get_next_state(
                    state,
                    self._actor.action_like_plant(action_features),
                ),
                plant.get_state_modulus()
            )

            # Age samples and update traces
            self._actor.age_samples(dt)
            self._plant_model.age_samples(dt)
            self._critic.age_samples(dt)
            self._critic.update_traces()

            if plant.is_out_of_bounds(state):
                self.logger.warning("Forced break at {}".format(state.flatten()))
                yield state, action, value, np.array(next_reward).reshape(1, 1)
                break

            yield state, action, value, np.array(next_reward).reshape(1, 1)

    def test(self, n, exploration=False):
        """
        Runs n simulations in testing mode (without updating).  
        :param n: number of simulations
        :param exploration: boolean representing 
        :return: 
        """
        sims = []
        for _ in range(n):
            simulation_output = zip(*self._simulation_generator(
                self._exploration_strategy[self._episodes] if exploration else int(1e10),
            ))
            sims.append(Simulation(
                self.get_plant(),
                *map(np.hstack, simulation_output),
                values=self._critic,
                rewards=self.get_reward_function()
            ))
        return SimulationSet(sims)

    def _simulation_generator(self, exploration_interval):
        """
        State generator for test method
        :param exploration_interval: exploration interval as integer
        :return: state, action 
        """
        t = 0.
        state = self.get_plant().get_initial_state()
        plant = self.get_plant()
        dt = plant.get_time_step()
        is_out_of_bounds = plant.is_out_of_bounds(state)

        while t <= self._length + dt and not is_out_of_bounds:
            action = self.get_action(state)
            do_explore = int(t / dt) % exploration_interval == 0 and t != 0
            if do_explore:
                action = self._actor.action_like_plant(
                    self._actor.perturb_action(self._actor.action_like_me(action))
                )
            yield state, action
            state = center_mod(
                plant.get_next_state(state, action),
                plant.get_state_modulus()
            )
            is_out_of_bounds = plant.is_out_of_bounds(state)
            t += dt

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
                "wb"
            )
        pickle.dump(self, file_handle)

    @classmethod
    def load(cls, file_id):
        """
        Load object from DUMP_PATH
        :param file_id: python file handle object
        :return: ActorCriticController object
        """
        if not isinstance(file_id, file):
            handle = open(
                "{:s}{:s}-{:d}.pckl".format(
                    DUMP_PATH,
                    cls.__name__,
                    file_id
                ),
                "rb"
            )
        else:
            handle = file_id
        return pickle.load(handle)
