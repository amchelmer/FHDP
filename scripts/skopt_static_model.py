import logging
import numpy as np
import os

from lib.controllers.actor_critic_controller import Actor, ActorCriticController, Critic, ExplorationStrategy
from lib.env import LOG_DIRECTORY
from lib.plants.plant_models import PlantModel
from lib.plants.quad_rotor_plant import QuadRotor2DPlant
from lib.reward_functions import QuadraticErrorRewardFunction
from lib.sets import ControllerSet, FeatureSet, RewardSet
from lib.simulations import SimulationResult
from lib.tools.runtime_tools import parallelize
from lib.validation.object_validation import assert_in, assert_true

from argparse import ArgumentParser
from copy import deepcopy
from skopt import space, gp_minimize, gbrt_minimize

parser = ArgumentParser(description='ID parser')
parser.add_argument('-j', type=int, help="threads", default=1)
parser.add_argument('-p', type=int, help="peers", default=1)
parser.add_argument('-shrink', type=int, help="Number of peers to shrink to", default=16)
parser.add_argument('-e', type=int, help="Number of episodes", default=150)
parser.add_argument("-v", help="verbose mode", default=True, action="store_false")
parser.add_argument("-f", help="function", default="gauss")
parser.add_argument("-starts", type=int, help="number of random starts", default=0)
parser.add_argument("-metric", help="metric (median/mean)", default="median")
parser.add_argument("-calls", type=int, help="number of function calls", default=1)
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

SEED = 4124135
np.random.seed(SEED)

LOOK_BACK_WINDOW = 20
FREQUENCY = 50.  # Hz
BLADE_FLAPPING = True
DEFAULT_ADD_METHOD = "mean"
DEFAULT_PURGE_METHOD = "age-weighted"
DEFAULT_LENGTH = 3  # seconds
DESIRED_STATE = np.array([[0, -10., 0, 0, 0, 0]]).T
DEFAULT_INIT_STATE_MEAN = np.array([[0, -9., 0, 0, 0, 0]]).T
TRAINING_EPISODES = int(parsed_args.e)
AGE_THRESHOLD = 45.

DISCOUNT = 0.94820523852228111
LAMBDA_TRACE = 0.75055692999458412
STATE_REWARDS = np.array([0., 10., 0, 0.4491648864, 7., 0.3])
ACTION_REWARDS = np.array([3.551383408, 4.983189])

(df_x, df_z,
 df_xdot, df_zdot,
 df_theta, df_thetadot,
 df_u1, df_u3) = QuadRotor2DPlant.get_default_feature_set()


def train(args):
    actor_critic = ActorCriticController(*args)
    actor_critic.AGE_THRESHOLD = AGE_THRESHOLD
    actor_critic.train(TRAINING_EPISODES)
    return actor_critic


def objective(objective_args):
    (
        ac_memory,
        pm_memory,
        exploration_phase_one,
        extra_exploration_phase_two,
        k_a,
        k_c,
        k_pm,
        tolerance_ac,
        prediction_tolerance,
        alpha_a,
        alpha_c
    ) = objective_args
    print(objective_args)

    quad_rotor_plant = QuadRotor2DPlant(
        1. / FREQUENCY,
        blade_flapping=BLADE_FLAPPING,
        init_mean=DEFAULT_INIT_STATE_MEAN,
    )

    actor_critic_args = (
        Actor(
            FeatureSet([df_z, df_zdot, df_theta, df_thetadot]),
            FeatureSet([df_u1, df_u3]),
            quad_rotor_plant.get_feature_set(),
            k_a,
            ac_memory * 100,
            alpha_a,
            tolerance_ac,
        ),
        Critic(
            FeatureSet([df_z, df_zdot, df_theta, df_thetadot]),
            quad_rotor_plant.get_feature_set(),
            k_c,
            ac_memory * 100,
            LAMBDA_TRACE,
            alpha_c,
            DISCOUNT,
            QuadraticErrorRewardFunction(
                ACTION_REWARDS,
                STATE_REWARDS,
                desired_state=DESIRED_STATE
            ),
            tolerance_ac,
        ),
        PlantModel(
            quad_rotor_plant.get_feature_set(),
            FeatureSet([df_z, df_zdot, df_theta, df_thetadot, df_u1, df_u3]),
            k_pm,
            pm_memory * 100,
            prediction_tolerance,
        ),
        quad_rotor_plant,
        DEFAULT_LENGTH,
        DEFAULT_ADD_METHOD,
        DEFAULT_PURGE_METHOD,
        ExplorationStrategy({
            1: exploration_phase_one,
            76: exploration_phase_one + extra_exploration_phase_two,
        })
    )

    trained_cs = ControllerSet(
        parallelize(
            parsed_args.j,
            train,
            [deepcopy(actor_critic_args) + (SEED + i,) for i in range(parsed_args.p)],
        )
    )
    trained_cs.dump()
    RewardSet(trained_cs).dump()
    return SimulationResult(
        trained_cs.lookback_result(LOOK_BACK_WINDOW),
        metric=parsed_args.metric
    ).get_cum_state_error().flatten()[1]


def callback(r):
    if r["func_vals"][-1] == min(r["func_vals"]):
        print(
            "^^^ Found new optimum of {:.2f} in iteration {:d} by using parameters {}!".format(
                r["fun"],
                len(r["func_vals"]),
                np.array2string(np.array(r["x"]), precision=6, separator=", "),
            )
        )


if __name__ == "__main__":
    try:
        print("Starting optimization process")

        # OPTIMIZATION
        optimization_space = [
            space.Integer(70, 95),  # ac memory
            space.Integer(55, 70),  # pm memory
            space.Integer(2, 4),  # exploration
            space.Integer(0, 1),  # exploration-update at 75
            space.Integer(16, 25),  # k_a
            space.Integer(19, 35),  # k_c
            space.Integer(9, 20),  # k_pm
            space.Real(1e-5, 1e-1, prior="log-uniform"),  # distance tolerance
            space.Real(1e-8, 1e-5, prior="log-uniform"),  # prediction tolerance
            space.Real(1e-2, 1., prior="log-uniform"),  # actor learning rate
            space.Real(1e-2, 1., prior="log-uniform"),  # critic learning rate
        ]

        res = FUNCTIONS[parsed_args.f](
            objective,
            optimization_space,
            n_calls=parsed_args.calls,
            n_random_starts=parsed_args.starts,
            callback=callback,
            verbose=parsed_args.v,
            random_state=np.random.RandomState(SEED + 10),
            x0=[
                [82, 55, 3, 1, 20, 32, 12, 1.8336910594205316e-05, 1.6651259798120438e-06, 0.8019620350814548,
                 0.035751093142586697]  # 101.6859
            ],
        )
    except KeyboardInterrupt:
        print("Shutdown requested... exiting")
    finally:
        try:
            post_message = "prematurely" if len(res["func_vals"]) < parsed_args.calls else ""
        except NameError:
            post_message = "prematurely"
        print("Stopped optimization {:s}".format(
            post_message
        ))
