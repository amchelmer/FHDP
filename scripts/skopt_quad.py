import argparse
import logging
import numpy as np
import os

from lib.controllers.actor_critic_controller import Actor, ActorCriticController, Critic, ExplorationStrategy
from lib.env import LOG_DIRECTORY
from lib.features import Feature
from lib.plants.plant_models import PlantModel
from lib.plants.quad_rotor_plant import QuadRotor2DPlant
from lib.reward_functions import QuadraticErrorRewardFunction
from lib.sets import ControllerSet, FeatureSet
from lib.simulations import SimulationResult
from lib.tools.runtime_tools import parallelize
from lib.validation.object_validation import assert_in, assert_true

from skopt import gp_minimize, space, gbrt_minimize

parser = argparse.ArgumentParser(description='ID parser')
parser.add_argument('-j', type=int, help="threads", default=1)
parser.add_argument('-p', type=int, help="peers", default=1)
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

logging.basicConfig(
    filename="{}{}.log".format(LOG_DIRECTORY, os.path.basename(__file__).split(".")[0]),
    level=logging.WARNING,
    filemode="w"
)

SEED = 1429
np.random.seed(SEED)
np.set_printoptions(
    precision=4,
    linewidth=200,
    suppress=True,
)
np.seterr(divide="raise", invalid="raise")

LOOK_BACK_WINDOW = 15
FREQUENCY = 50.  # Hz
BLADE_FLAPPING = True
DEFAULT_ADD_METHOD = "mean"
DEFAULT_PURGE_METHOD = "age-weighted"
DEFAULT_LENGTH = 3  # seconds
DEFAULT_EPISODES = 75
DESIRED_STATE = np.array([[0, -10., 0, 0, 0, 0]]).T
DEFAULT_INIT_STATE_MEAN = np.array([[0, -9., 0, 0, 0, 0]]).T


def train(args):
    actor_critic = ActorCriticController(*args)
    actor_critic.train(DEFAULT_EPISODES)
    return actor_critic


def callback(r):
    if r["func_vals"][-1] == min(r["func_vals"]):
        print(
            "Found new optimum of {:.2f} in iteration {:d} by using parameters {}!".format(
                r["fun"],
                len(r["func_vals"]),
                np.array2string(np.array(r["x"]), precision=3),
            )
        )


def objective(objective_args):
    (z_scale,
     zdot_reward,
     action_reward,
     exploration,
     tolerance,
     max_mem_a,
     k_a,
     alpha_a,
     k_c,
     alpha_c,
     max_mem_pm,
     k_pm,
     pred_tol,
     lambda_trace,
     gamma) = objective_args

    (df_x,
     df_z,
     df_xdot,
     df_zdot,
     df_theta,
     df_thetadot,
     df_u1,
     df_u3) = QuadRotor2DPlant.get_default_feature_set()

    feature_z = Feature(r"$z$ [m]", scale=z_scale, bounds=np.array([-25, 0]))
    quad_rotor_plant = QuadRotor2DPlant(
        1. / FREQUENCY,
        blade_flapping=BLADE_FLAPPING,
        init_mean=DEFAULT_INIT_STATE_MEAN,
        feature_set=FeatureSet([
            df_x, feature_z,
            df_xdot, df_zdot,
            df_theta, df_thetadot,
            df_u1, df_u3,
        ])
    )

    train_args = (
        Actor(
            FeatureSet([feature_z, df_zdot]),
            FeatureSet([df_u1]),
            quad_rotor_plant.get_feature_set(),
            k_a,
            max_mem_a * 50,
            alpha_a,
            tolerance,
        ),
        Critic(
            FeatureSet([feature_z, df_zdot]),
            quad_rotor_plant.get_feature_set(),
            k_c,
            max_mem_a * 50,
            lambda_trace,
            alpha_c,
            gamma,
            QuadraticErrorRewardFunction(
                [action_reward, 0],
                [0, 10., 0, zdot_reward, 0, 0],
                desired_state=DESIRED_STATE
            ),
            tolerance
        ),
        PlantModel(
            quad_rotor_plant.get_feature_set(),
            FeatureSet([feature_z, df_zdot, df_u1]),
            k_pm,
            max_mem_pm * 50,
            pred_tol,
        ),
        quad_rotor_plant,
        DEFAULT_LENGTH,
        DEFAULT_ADD_METHOD,
        DEFAULT_PURGE_METHOD,
        ExplorationStrategy({1: exploration}),
    )

    cs = ControllerSet(
        parallelize(
            parsed_args.j,
            train,
            [train_args + (SEED + i,) for i in range(parsed_args.p)],
        )
    )

    result = SimulationResult(
        cs.lookback_result(
            LOOK_BACK_WINDOW,
            look_back_metric="median",
        ),
        metric=parsed_args.metric,
    )
    training_message = "Finished training with cumulative z-error {:.2f}".format(
        result.get_cum_state_error().flatten()[1]
    )
    print(training_message)
    return result.get_cum_state_error().flatten()[1]


if __name__ == "__main__":
    try:
        optimization_space = [
            space.Real(2., 35.0),  # z-scale
            space.Real(0.3, 1.0),  # zdot_reward
            space.Real(2., 5.),  # action reward
            space.Integer(1, 3),  # exploration
            space.Real(1e-5, 3e-2, prior="log-uniform"),  # tolerance
            space.Integer(16, 32),  # Maxmem * 50
            space.Integer(13, 18),  # ka
            space.Real(1e-3, 1., prior="log-uniform"),  # alpha a
            space.Integer(18, 24),  # kc
            space.Real(1e-3, 1., prior="log-uniform"),  # alpha c
            space.Integer(10, 20),  # Maxmem * 50
            space.Integer(9, 12),  # kpm
            space.Real(3e-8, 3e-4, prior="log-uniform"),  # pred_tol pm
            space.Real(0.75, 0.95),  # lambda
            space.Real(0.90, 0.97),  # gamma
        ]

        print("Starting new optimization run.")

        res = FUNCTIONS[parsed_args.f](
            objective,
            optimization_space,
            n_calls=parsed_args.calls,
            n_random_starts=parsed_args.starts,
            callback=callback,
            verbose=parsed_args.v,
            random_state=np.random.RandomState(405),
        )
    except KeyboardInterrupt:
        print("Shutdown requested... exiting")
    finally:
        try:
            message = "" if len(res["func_vals"]) >= parsed_args.calls else "(error)"
        except NameError:
            message = "prematurely"
        print("Stopped optimization {}".format(message))
