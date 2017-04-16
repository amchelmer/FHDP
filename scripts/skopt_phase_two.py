import argparse
import logging
import numpy as np
import os

from lib.sets.controller_set.controller_set import ControllerSet
from lib.controllers.actor_critic_controller import Actor, ActorCriticController, Critic, ExplorationStrategy
from lib.env import LOG_DIRECTORY
from lib.features import Feature, FeatureChange
from lib.plants.plant_models import PlantModel
from lib.plants.quad_rotor_plant import QuadRotor2DPlant
from lib.reward_functions import QuadraticErrorRewardFunction
from lib.sets import FeatureSet
from lib.simulations import SimulationResult
from lib.tools.runtime_tools import parallelize
from lib.validation.object_validation import assert_in, assert_true

from copy import deepcopy
from skopt import space, gp_minimize, gbrt_minimize


parser = argparse.ArgumentParser(description='ID parser')
parser.add_argument('-j', type=int, help="threads", default=1)
parser.add_argument('-p', type=int, help="peers", default=1)
parser.add_argument('-shrink', type=int, help="Number of peers to shrink to", default=16)
parser.add_argument("-v", help="verbose mode", default=True, action="store_false")
parser.add_argument("-f", help="function", default="gauss")
parser.add_argument("-starts", type=int, help="number of random starts", default=10)
parser.add_argument("-metric", help="metric (median/mean)", default="median")
parser.add_argument("-calls", type=int, help="number of function calls", default=100)
parsed_args = parser.parse_args()

FUNCTIONS = {
    "gauss": gp_minimize,
    "boost": gbrt_minimize,
}
METRICS = ["mean", "median"]

assert_true(parsed_args.p >= 1, "Number of peers should be at least 1")
assert_true(parsed_args.j >= 1, "Number of threads should be at least 1")
assert_in(parsed_args.f, FUNCTIONS)
assert_in(parsed_args.metric, METRICS)

np.seterr(divide="raise", invalid="raise")
np.set_printoptions(
    precision=4,
    linewidth=200,
    suppress=True,
)

logging.basicConfig(
    filename="{}{}.log".format(LOG_DIRECTORY, os.path.basename(__file__).split(".")[0]),
    level=logging.WARNING,
    filemode="w"
)

SEED = 12235
np.random.seed(SEED)

LOOK_BACK_WINDOW = 15
FREQUENCY = 50.  # Hz
BLADE_FLAPPING = True
DEFAULT_ADD_METHOD = "mean"
DEFAULT_PURGE_METHOD = "age-weighted"
DEFAULT_LENGTH = 3  # seconds
DEFAULT_FEATURE_CHANGES = ()
DESIRED_STATE = np.array([[0, -10., 0, 0, 0, 0]]).T
DEFAULT_INIT_STATE_MEAN = np.array([[0, -9., 0, 0, 0, 0]]).T
STAGE_ONE_EXPLORATION_DICT = {1: 2, 51: 3, 101: 2}
STAGE_ONE_EPISODES = 75
AGE_THRESHOLD = 30.

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
STATE_REWARDS = np.array([0, 10., 0, 0.4491648864, 0, 0])
ACTION_REWARDS = np.array([3.551383408, 0])

# Plant model
K_PLANT_MODEL = 9
STAGE_ONE_PM_MEMORY = 350
PREDICTION_TOLERANCE = 4.9763444056056387e-07

# STAGE TWO
STAGE_TWO_EPISODES = 75
STAGE_TWO_INCR_HOLD = 3
STAGE_TWO_METHOD = "clone-gauss"

(df_x, df_z,
 df_xdot, df_zdot,
 df_theta, df_thetadot,
 df_a1, df_a2) = QuadRotor2DPlant.get_default_feature_set()


def train_stage_one(args):
    actor_critic = ActorCriticController(*args)
    actor_critic.AGE_THRESHOLD = AGE_THRESHOLD
    actor_critic.train(STAGE_ONE_EPISODES)
    return actor_critic


def train_stage_two(args):
    (
        max_mem_ac,
        max_mem_pm,
        theta_spread,
        thetadot_spread,
        exploration,
        theta_reward,
        thetadot_reward,
        u_3_reward,
        k_a,
        k_c,
        k_pm,
        feature_theta,
        feature_thetadot,
        feature_a2,
        actor_critic
    ) = args

    actor_critic._actor._knn = int(k_a)
    actor_critic._critic._knn = int(k_c)
    actor_critic._plant_model._knn = int(k_pm)

    actor_critic.get_reward_function().add_weights(
        np.array([0, u_3_reward]),
        np.array([0, 0., 0., 0., theta_reward, thetadot_reward]),
    )

    actor_critic.change_feature(FeatureChange(feature_a2, "zero"))
    actor_critic.set_memory_sizes(3000, 3000, None)
    actor_critic.change_feature(
        FeatureChange(
            feature_theta,
            STAGE_TWO_METHOD,
            spread=theta_spread,
        )
    )

    # TRAIN PITCH ANGLE
    actor_critic.set_memory_sizes(int(max_mem_ac), int(max_mem_ac), int(max_mem_pm))
    actor_critic.change_feature(
        FeatureChange(feature_thetadot, STAGE_TWO_METHOD, spread=thetadot_spread)
    )
    actor_critic.train(STAGE_TWO_EPISODES, train_hold=STAGE_TWO_INCR_HOLD)
    return actor_critic


def objective(objective_args):
    (
        a2_scale,
        theta_scale,
        thetadot_scale,
        max_mem_ac,
        max_mem_pm,
        theta_spread,
        thetadot_spread,
        exploration,
        theta_reward,
        thetadot_reward,
        u_3_reward,
        k_a,
        k_c,
        k_pm,
    ) = objective_args

    feature_theta = Feature(r"$\theta$ [rad]", scale=theta_scale),
    feature_thetadot = Feature(r"$\dot{\theta}$ [rad/s]", scale=thetadot_scale, derivative=True)
    feature_a2 = Feature(r"$a_2$ [-]", feature_type="action", scale=0.760859, bounds=0.3 * np.array([-1, 1]))
    quad_rotor_plant = QuadRotor2DPlant(
        1. / FREQUENCY,
        blade_flapping=BLADE_FLAPPING,
        init_mean=DEFAULT_INIT_STATE_MEAN,
        feature_set=FeatureSet([
            df_x, df_z,
            df_xdot, df_zdot,
            feature_theta, feature_thetadot,
            df_a1, feature_a2
        ]),
    )

    stage_one_args = [
        Actor(
            FeatureSet([df_z, df_zdot]),
            FeatureSet([df_a1]),
            quad_rotor_plant.get_feature_set(),
            K_ACTOR,
            STAGE_ONE_AC_MEMORY,
            ALPHA_ACTOR,
            TOLERANCE_ACTOR,
        ),
        Critic(
            FeatureSet([df_z, df_zdot]),
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
            FeatureSet([df_z, df_zdot, df_a1]),
            K_PLANT_MODEL,
            STAGE_ONE_PM_MEMORY,
            PREDICTION_TOLERANCE,
        ),
        quad_rotor_plant,
        DEFAULT_LENGTH,
        DEFAULT_ADD_METHOD,
        DEFAULT_PURGE_METHOD,
        ExplorationStrategy(STAGE_ONE_EXPLORATION_DICT)
    ]

    print("Training basic quad-rotor")

    # STAGE ONE
    cs_stage_one = ControllerSet(
        parallelize(
            parsed_args.j,
            train_stage_one,
            [stage_one_args + [SEED + i] for i in range(parsed_args.p)],
        )
    )

    _, z_error_stage_one, _, _, _, _ = SimulationResult(
        cs_stage_one.lookback_result(LOOK_BACK_WINDOW),
        metric=parsed_args.metric
    ).get_cum_state_error().flatten()
    print(
        "Finished stage one with {:s} cumulative z-error of {:.2f}".format(parsed_args.metric, z_error_stage_one)
    )

    stage_two_args = [
        max_mem_ac,
        max_mem_pm,
        theta_spread,
        thetadot_spread,
        exploration,
        theta_reward,
        thetadot_reward,
        u_3_reward,
        k_a,
        k_c,
        k_pm,
        feature_theta,
        feature_thetadot,
        feature_a2
    ]
    cs_stage_two = ControllerSet(
        parallelize(
            parsed_args.j,
            train_stage_two,
            [stage_two_args + [deepcopy(ac)] for ac in cs_stage_one],
        )
    )
    x_error, z_error, _, _, theta_error, _ = SimulationResult(
        cs_stage_two.lookback_result(LOOK_BACK_WINDOW),
        metric=parsed_args.metric
    ).get_cum_state_error().flatten()
    return z_error


def callback(r):
    if r["func_vals"][-1] == min(r["func_vals"]):
        print(
            "Found new optimum of {:.2f} in iteration {:d} by using parameters {}!".format(
                r["fun"],
                len(r["func_vals"]),
                np.array2string(np.array(r["x"]), precision=6),
            )
        )


if __name__ == "__main__":
    try:
        optimization_space = [
            space.Real(0.3, 1.),  # a2-scale
            space.Real(1, 5.),  # theta_scale
            space.Real(3., 8.),  # theta_dot_scale +
            space.Integer(65, 95),  # ac memory +
            space.Integer(50, 70),  # pm memory +
            space.Real(0.03, 0.3),  # theta-spread +
            space.Real(0.3, 1.),  # thetadot-spread +
            space.Integer(1, 3),  # exploration
            space.Real(5., 10.),  # theta-reward +
            space.Real(0.1, 2.),  # thetadot_reward +
            space.Real(3., 6.),  # a2-reward +
            space.Integer(16, 30),  # k_a +
            space.Integer(18, 35),  # k_c +
            space.Integer(9, 30),  # k_pm +
        ]

        res = FUNCTIONS[parsed_args.f](
            objective,
            optimization_space,
            n_calls=parsed_args.calls,
            n_random_starts=parsed_args.starts,
            callback=callback,
            verbose=parsed_args.v,
        )
    except KeyboardInterrupt:
        print("Shutdown requested... exiting")
    finally:
        try:
            message = "" if len(res["func_vals"]) >= parsed_args.calls else "(error)"
        except NameError:
            message = "prematurely"
        print("Stopped optimization {}".format(message))
