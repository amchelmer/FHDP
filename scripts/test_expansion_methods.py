import argparse
import logging
import numpy as np
import os

from lib.controllers.actor_critic_controller import Actor, ActorCriticController, Critic, ExplorationStrategy
from lib.env import LOG_DIRECTORY
from lib.features import FeatureChange
from lib.plants.plant_models import PlantModel
from lib.plants.quad_rotor_plant import QuadRotor2DPlant
from lib.reward_functions import QuadraticErrorRewardFunction
from lib.sets import ControllerSet, FeatureSet, RewardSet
from lib.simulations import SimulationResult
from lib.tools.runtime_tools import parallelize
from lib.validation.object_validation import assert_true

from copy import deepcopy
from requests import ConnectionError


parser = argparse.ArgumentParser(description='ID parser')
parser.add_argument('-j', type=int, help="threads", default=1)
parser.add_argument('-p', type=int, help="peers", default=1)
parser.add_argument('-shrink', type=int, help="Number of peers to shrink to", default=16)
parser.add_argument('-debug', help="Full debug logging", default=False, action='store_true')
parser.add_argument("-metric", help="metric (median/mean)", default="median")
parser.add_argument("-v", help="verbose mode", default=True, action="store_false")
parsed_args = parser.parse_args()

assert_true(parsed_args.p >= 1, "Number of peers should be at least 1")
assert_true(parsed_args.j >= 1, "Number of threads should be at least 1")

logging.basicConfig(
    filename="{}{}.log".format(LOG_DIRECTORY, os.path.basename(__file__).split(".")[0]),
    level=logging.DEBUG if parsed_args.debug else logging.INFO,
    filemode="w"
)
SEED = 4124135
np.random.seed(SEED)
np.set_printoptions(
    precision=4,
    linewidth=200,
    suppress=True,
)
np.seterr(divide="raise", invalid="raise")

LOOK_BACK_WINDOW = 5
FREQUENCY = 50.  # Hz
BLADE_FLAPPING = True
DEFAULT_ADD_METHOD = "mean"
DEFAULT_PURGE_METHOD = "age-weighted"
DEFAULT_LENGTH = 3  # seconds
STAGE_ONE_EPISODES = 75
EXPLORATION_DICT = {1: 2, 51: 3, 101: 3}
DEFAULT_INIT_STATE_MEAN = np.array([[0, -9., 0, 0, 0, 0]]).T
DESIRED_STATE = np.array([[0, -10., 0, 0, 0, 0]]).T
AGE_THRESHOLD = 45.

# Actor
ALPHA_ACTOR = 0.15032140063618069
K_ACTOR = 16
STAGE_ONE_AC_MEMORY = 1000
TOLERANCE_ACTOR = 2.0884739914537273e-3

# Critic
ALPHA_CRITIC = 0.26968673558982431
K_CRITIC = 20
TOLERANCE_CRITIC = TOLERANCE_ACTOR
DISCOUNT = 0.94820523852228111
LAMBDA_TRACE = 0.75055692999458412
STATE_REWARDS = np.array([0., 10., 0, 0.4491648864, 7., 0.3])
ACTION_REWARDS = np.array([3.551383408, 4.983189])

# Plant model
K_PLANT_MODEL = 9
STAGE_ONE_PM_MEMORY = 350
PREDICTION_TOLERANCE = 4.9763444056056387e-07

# Phase two
STAGE_TWO_EPISODES = 75
STAGE_TWO_AC_MEMORY = 9500
STAGE_TWO_PM_MEMORY = 6500
STAGE_TWO_INBETWEEN_MEMORY = 3000
STAGE_TWO_INCR_HOLD = 3
STAGE_TWO_ACTOR_KNN = 25
STAGE_TWO_CRITIC_KNN = 30
STAGE_TWO_PM_KNN = 27
PITCH_SPREAD = 0.106011
PITCH_DOT_SPREAD = 0.761986


def train_stage_one(args):
    actor_critic = ActorCriticController(*args)
    actor_critic.AGE_THRESHOLD = AGE_THRESHOLD
    actor_critic.train(STAGE_ONE_EPISODES)
    actor_critic._actor._knn = int(STAGE_TWO_ACTOR_KNN)
    actor_critic._critic._knn = int(STAGE_TWO_CRITIC_KNN)
    actor_critic._plant_model._knn = int(STAGE_TWO_PM_KNN)
    actor_critic.change_feature(FeatureChange(feature_u3, "zero"))
    return actor_critic


def train_zero_expansion(actor_critic):
    actor_critic.set_memory_sizes(STAGE_TWO_AC_MEMORY, STAGE_TWO_AC_MEMORY, STAGE_TWO_PM_MEMORY)

    actor_critic.change_feature(
        FeatureChange(feature_theta, "zero")
    )
    actor_critic.change_feature(
        FeatureChange(feature_thetadot, "zero")
    )
    actor_critic.train(
        STAGE_TWO_EPISODES,
        train_hold=STAGE_TWO_INCR_HOLD,
    )
    return actor_critic


def train_perturb_expansion(actor_critic):
    actor_critic.set_memory_sizes(STAGE_TWO_AC_MEMORY, STAGE_TWO_AC_MEMORY, STAGE_TWO_PM_MEMORY)

    actor_critic.change_feature(
        FeatureChange(feature_theta, "perturb-gauss", spread=PITCH_SPREAD)
    )
    actor_critic.change_feature(
        FeatureChange(feature_thetadot, "perturb-gauss", spread=PITCH_DOT_SPREAD)
    )
    actor_critic.train(
        STAGE_TWO_EPISODES,
        train_hold=STAGE_TWO_INCR_HOLD,
    )
    return actor_critic


def train_uniform_clone_expansion(actor_critic):
    actor_critic.set_memory_sizes(STAGE_TWO_INBETWEEN_MEMORY, STAGE_TWO_INBETWEEN_MEMORY, None)

    actor_critic.change_feature(
        FeatureChange(feature_theta, "clone-uniform", spread=3 * PITCH_SPREAD)
    )

    actor_critic.set_memory_sizes(STAGE_TWO_AC_MEMORY, STAGE_TWO_AC_MEMORY, STAGE_TWO_PM_MEMORY)
    actor_critic.change_feature(
        FeatureChange(feature_thetadot, "clone-uniform", spread=3 * PITCH_DOT_SPREAD)
    )
    actor_critic.train(STAGE_TWO_EPISODES, train_hold=STAGE_TWO_INCR_HOLD)
    return actor_critic


def train_gaussian_clone_expansion(actor_critic):
    actor_critic.set_memory_sizes(STAGE_TWO_INBETWEEN_MEMORY, STAGE_TWO_INBETWEEN_MEMORY, None)

    actor_critic.change_feature(
        FeatureChange(feature_theta, "clone-gauss", spread=PITCH_SPREAD)
    )

    actor_critic.set_memory_sizes(STAGE_TWO_AC_MEMORY, STAGE_TWO_AC_MEMORY, STAGE_TWO_PM_MEMORY)
    actor_critic.change_feature(
        FeatureChange(feature_thetadot, "clone-gauss", spread=PITCH_DOT_SPREAD)
    )
    actor_critic.train(STAGE_TWO_EPISODES, train_hold=STAGE_TWO_INCR_HOLD)
    return actor_critic


def train_separate(actor_critic):
    actor_critic.set_memory_sizes(STAGE_TWO_INBETWEEN_MEMORY, STAGE_TWO_INBETWEEN_MEMORY, 1500)

    actor_critic.change_feature(
        FeatureChange(feature_theta, "clone-gauss", spread=PITCH_SPREAD)
    )
    actor_critic.train(int(np.ceil(STAGE_TWO_EPISODES / 2)), train_hold=STAGE_TWO_INCR_HOLD)

    actor_critic.set_memory_sizes(STAGE_TWO_AC_MEMORY, STAGE_TWO_AC_MEMORY, STAGE_TWO_PM_MEMORY)
    actor_critic.change_feature(
        FeatureChange(feature_thetadot, "clone-gauss", spread=PITCH_DOT_SPREAD)
    )
    actor_critic.train(int(np.floor(STAGE_TWO_EPISODES / 2)), train_hold=STAGE_TWO_INCR_HOLD)
    return actor_critic


if __name__ == "__main__":
    try:
        quad_rotor_plant = QuadRotor2DPlant(
            1. / FREQUENCY,
            blade_flapping=BLADE_FLAPPING,
            init_mean=DEFAULT_INIT_STATE_MEAN,
        )
        (feature_x, feature_z,
         feature_xdot, feature_zdot,
         feature_theta, feature_thetadot,
         feature_u1, feature_u3) = quad_rotor_plant.get_feature_set()

        actor_critic_args = (
            Actor(
                FeatureSet([feature_z, feature_zdot]),
                FeatureSet([feature_u1]),
                quad_rotor_plant.get_feature_set(),
                K_ACTOR,
                STAGE_ONE_AC_MEMORY,
                ALPHA_ACTOR,
                TOLERANCE_ACTOR,
            ),
            Critic(
                FeatureSet([feature_z, feature_zdot]),
                quad_rotor_plant.get_feature_set(),
                K_CRITIC,
                STAGE_ONE_AC_MEMORY,
                LAMBDA_TRACE,
                ALPHA_CRITIC,
                DISCOUNT,
                QuadraticErrorRewardFunction(
                    ACTION_REWARDS,
                    STATE_REWARDS,
                    desired_state=DESIRED_STATE
                ),
                TOLERANCE_CRITIC,
            ),
            PlantModel(
                quad_rotor_plant.get_feature_set(),
                FeatureSet([feature_z, feature_zdot, feature_u1]),
                K_PLANT_MODEL,
                STAGE_ONE_PM_MEMORY,
                PREDICTION_TOLERANCE,
            ),
            quad_rotor_plant,
            DEFAULT_LENGTH,
            DEFAULT_ADD_METHOD,
            DEFAULT_PURGE_METHOD,
            ExplorationStrategy(EXPLORATION_DICT),
        )

        print("Starting training of quad-rotor.")

        # STAGE ONE
        first_stage_cs = ControllerSet(
            parallelize(
                parsed_args.j,
                train_stage_one,
                [actor_critic_args + (SEED + i,) for i in range(parsed_args.p)],
            )
        )
        print("Finished stage one with {:.2f}".format(
            SimulationResult(
                first_stage_cs.lookback_result(LOOK_BACK_WINDOW),
                metric=parsed_args.metric
            ).get_cum_state_error()[1:2].sum()
        ))

        # ZERO EXPANSION
        zero_expansion = ControllerSet(
            parallelize(
                parsed_args.j,
                train_zero_expansion,
                [deepcopy(ac) for ac in first_stage_cs],
            )
        )
        print("Finished zero expansion (id={}) with {:.2f}".format(
            zero_expansion.get_id(),
            SimulationResult(
                zero_expansion.lookback_result(LOOK_BACK_WINDOW),
                metric=parsed_args.metric
            ).get_cum_state_error()[1:2].sum()
        ))
        zero_expansion.notes = "Test with zero expansion method"
        # zero_expansion.dump()
        RewardSet(zero_expansion).dump()

        # PERTURB-GAUSS EXPANSION
        perturb_gauss_expansion = ControllerSet(
            parallelize(
                parsed_args.j,
                train_perturb_expansion,
                [deepcopy(ac) for ac in first_stage_cs],
            )
        )
        print("Finished perturb Gauss expansion (id={}) with {:.2f}".format(
            perturb_gauss_expansion.get_id(),
            SimulationResult(
                perturb_gauss_expansion.lookback_result(LOOK_BACK_WINDOW),
                metric=parsed_args.metric
            ).get_cum_state_error()[1:2].sum()
        ))
        perturb_gauss_expansion.notes = "Test with perturb Gauss expansion method"
        # perturb_gauss_expansion.dump()
        RewardSet(perturb_gauss_expansion).dump()

        # UNIFORM CLONE EXPANSION
        uniform_clone_expansion = ControllerSet(
            parallelize(
                parsed_args.j,
                train_uniform_clone_expansion,
                [deepcopy(ac) for ac in first_stage_cs],
            )
        )
        print("Finished uniform clone expansion (id={}) with {:.2f}".format(
            uniform_clone_expansion.get_id(),
            SimulationResult(
                uniform_clone_expansion.lookback_result(LOOK_BACK_WINDOW),
                metric=parsed_args.metric
            ).get_cum_state_error()[1:2].sum()
        ))
        uniform_clone_expansion.notes = "Test with uniform clone expansion method"
        # uniform_clone_expansion.dump()
        RewardSet(uniform_clone_expansion).dump()

        # GAUSSIAN CLONE EXPANSION
        gaussian_clone_expansion = ControllerSet(
            parallelize(
                parsed_args.j,
                train_gaussian_clone_expansion,
                [deepcopy(ac) for ac in first_stage_cs],
            )
        )
        print("Finished Gaussian clone expansion (id={}) with {:.2f}".format(
            gaussian_clone_expansion.get_id(),
            SimulationResult(
                gaussian_clone_expansion.lookback_result(LOOK_BACK_WINDOW),
                metric=parsed_args.metric
            ).get_cum_state_error()[1:2].sum()
        ))
        gaussian_clone_expansion.notes = "Test with Gaussian clone expansion method"
        gaussian_clone_expansion.dump()
        RewardSet(gaussian_clone_expansion).dump()

        # SEPARATE GAUSSIAN CLONE EXPANSION
        separate_gaussian_clone_expansion = ControllerSet(
            parallelize(
                parsed_args.j,
                train_separate,
                [deepcopy(ac) for ac in first_stage_cs],
            )
        )
        print("Finished separate Gaussian clone expansion (id={}) with {:.2f}".format(
            separate_gaussian_clone_expansion.get_id(),
            SimulationResult(
                separate_gaussian_clone_expansion.lookback_result(LOOK_BACK_WINDOW),
                metric=parsed_args.metric
            ).get_cum_state_error()[1:2].sum()
        ))
        separate_gaussian_clone_expansion.notes = "Test with separate Gaussian clone expansion method"
        # separate_gaussian_clone_expansion.dump()
        RewardSet(separate_gaussian_clone_expansion).dump()

    except KeyboardInterrupt:
        print("Shutdown requested... exiting")
    finally:
        print("Stopped run!")
