import numpy as np
import tempfile

from .test_actor import TestActor
from .test_critic import TestCritic
from ..controller_base_test import ControllerBaseTest
from ...plant_tests.plant_model_tests.test_plant_model import TestPlantModel
from ....plants import AbstractPlant
from ....plants.plant_models.plant_model import PlantModel
from ....controllers.actor_critic_controller import Actor, ActorCriticController, Critic, ExplorationStrategy

from numpy.testing import assert_array_equal


class _DummyPlant(AbstractPlant):
    DT = 0.1
    INTEGRATE = "euler"

    def __init__(self, feature_set, ):
        super(_DummyPlant, self).__init__(self.DT,
                                          self.INTEGRATE,
                                          feature_set,
                                          np.zeros((4, 1)),
                                          np.ones((4, 1)))

    def _key(self):
        return 0

    def compute_derivative(self, state, action):
        return np.array([[-0.1, 0.2, 0.25, -0.25]]).T


class TestActorCritic(ControllerBaseTest):
    SEED = 1803
    ACTOR = Actor(**{
        "input_feature_set": TestActor.INPUT_FEATURE_SET,
        "output_feature_set": TestActor.OUTPUT_FEATURE_SET,
        "plant_feature_set": TestActor.PLANT_FEATURE_SET,
        "knn": TestActor.KNN,
        "max_memory": TestActor.MAX_MEMORY,
        "alpha": TestActor.ALPHA,
        "epsilon_p_feature": TestActor.EPSILON_P_FEATURE,
    })
    TestActor._add_datapoints(ACTOR, TestActor.STATES, TestActor.VALUES, TestActor.DT)
    CRITIC = Critic(**{
        "input_feature_set": TestCritic.INPUT_FEATURE_SET,
        "plant_feature_set": TestCritic.PLANT_FEATURE_SET,
        "knn": TestCritic.KNN,
        "max_memory": TestCritic.MAX_MEMORY,
        "trace": TestCritic.TRACE,
        "alpha": TestCritic.ALPHA,
        "discount": TestCritic.DISCOUNT,
        "reward_function": TestCritic.REWARD_FUNCTION,
        "epsilon_p_feature": TestCritic.EPSILON_P_FEATURE,
    })
    TestCritic._add_datapoints(CRITIC, TestCritic.STATES, TestCritic.VALUES, TestCritic.DT)
    PLANT_MODEL = PlantModel(**{
        "plant_feature_set": TestPlantModel.PLANT_FEATURE_SET,
        "feature_set": TestPlantModel.FEATURE_SET,
        "knn": TestPlantModel.KNN,
        "max_memory": TestPlantModel.MAX_MEMORY,
        "prediction_epsilon": TestPlantModel.PREDICTION_EPSILON,
    })
    TestPlantModel._add_datapoints(PLANT_MODEL, TestPlantModel.STATES, TestPlantModel.VALUES, TestPlantModel.DT)

    PLANT = _DummyPlant(TestActor.PLANT_FEATURE_SET)
    ADD_METHOD = "mean"
    PURGE_METHOD = "age"
    QUERY_POINT = TestActor.QUERY_POINT
    EPISODE_LENGTH = 0.5
    EXPLORATION_STRATEGY = ExplorationStrategy({1: 3})

    def _get_controller_cls(self):
        return ActorCriticController

    def _get_controller_kwargs(self):
        return {
            "actor": self.ACTOR,
            "critic": self.CRITIC,
            "plant_model": self.PLANT_MODEL,
            "plant": self.PLANT,
            "episode_length": self.EPISODE_LENGTH,
            "add_method": self.ADD_METHOD,
            "purge_method": self.PURGE_METHOD,
            "exploration_strategy": self.EXPLORATION_STRATEGY,
            "seed": self.SEED,
        }

    def _generate_controller(self):
        return self._get_controller_cls()(**self._get_controller_kwargs())

    def controller_base_test(self):
        self._controller_base_test()

    def _test_get_action(self):
        actor_critic = self._generate_controller()
        ac_action = actor_critic.get_action(self.QUERY_POINT)
        actor_action, actor_knn_keys = actor_critic._actor.get_action(self.QUERY_POINT)
        assert_array_equal(ac_action, actor_action)

    def _test_reset(self):
        actor_critic = self._generate_controller()
        actor_critic.reset()

    def test_get_critic(self):
        actor_critic = self._generate_controller()
        self.assertEqual(actor_critic.get_critic(), actor_critic._critic)

    def test_get_reward_function(self):
        actor_critic = self._generate_controller()
        self.assertEqual(actor_critic.get_reward_function(), actor_critic._critic._reward_function)

    def test_get_plant(self):
        actor_critic = self._generate_controller()
        self.assertEqual(actor_critic.get_plant(), actor_critic._plant)

    def test_get_plant_model(self):
        actor_critic = self._generate_controller()
        self.assertEqual(actor_critic.get_plant_model(), actor_critic._plant_model)

    def test_get_train_results(self):
        raise NotImplementedError

    def test_get_last_results(self):
        raise NotImplementedError

    def test_set_memory_sizes(self):
        actor_critic = self._generate_controller()
        actor_critic.set_memory_sizes(200, 300, 400)
        self.assertEqual(actor_critic._actor._max_memory, 200)
        self.assertEqual(actor_critic._critic._max_memory, 300)
        self.assertEqual(actor_critic._plant_model._max_memory, 400)

    def test_purge_and_rebuild(self):
        raise NotImplementedError

    def test_change_feature(self):
        raise NotImplementedError

    def test_train(self):
        raise NotImplementedError

    def test_test(self):
        raise NotImplementedError

    def _test_dump_and_load(self):
        file_handle = tempfile.TemporaryFile()
        actor_critic = self._generate_controller()

        file_handle.seek(0)
        actor_critic_loaded = self._get_controller_cls().load(file_handle)
        self.assertEqual(
            actor_critic,
            actor_critic_loaded
        )
