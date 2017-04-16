import numpy as np

from lib.controllers.actor_critic_controller import ActorCriticController
from lib.controllers.gaussian_random_controller import GaussianRandomController
from lib.env import FIGURE_PATH

SEED = 4124135
np.random.seed(SEED)
np.set_printoptions(
    precision=4,
    linewidth=200,
    suppress=True,
)
np.seterr(divide="raise", invalid="raise")

INIT_STATE = np.array([[0, -9, 0, 0, 0, 0]]).T
LENGTH = 3.

base_path = FIGURE_PATH
file_format = "pdf"
dpi = 300

if __name__ == "__main__":
    ac = ActorCriticController.load(1491730678140478955354960)
    pm = ac.get_plant_model()
    plant = ac.get_plant()
    random_controller = GaussianRandomController(SEED, np.array([[-1, 1], [-.3, .3]]))

    axes = pm.one_step_ahead_simulation(plant, LENGTH, ac, INIT_STATE)
    axes[0].figure.set_size_inches(8, 14)
    axes[0].figure.savefig(
        base_path + "plant_model_quality_one_step_ahead.{}".format(file_format),
        bbox_inches='tight',
        pad_inches=0.1,
        format=file_format,
        transparant=True,
        dpi=dpi,
    )
    axes = pm.one_step_ahead_errors(plant, LENGTH, ac, INIT_STATE)
    axes[0].figure.set_size_inches(8, 14)
    axes[0].figure.savefig(
        base_path + "plant_model_quality_one_step_ahead_err.{}".format(file_format),
        bbox_inches='tight',
        pad_inches=0.1,
        format=file_format,
        transparant=True,
        dpi=dpi,
    )

    axes = pm.one_step_ahead_simulation(plant, LENGTH, random_controller, INIT_STATE)
    axes[0].figure.set_size_inches(8, 14)
    axes[0].figure.savefig(
        base_path + "plant_model_quality_one_step_ahead_random.{}".format(file_format),
        bbox_inches='tight',
        pad_inches=0.1
    )
    random_controller.reset()
    axes = pm.one_step_ahead_errors(plant, LENGTH, random_controller, INIT_STATE)
    axes[0].figure.set_size_inches(8, 14)
    axes[0].figure.savefig(
        base_path + "plant_model_quality_one_step_ahead_random_err.{}".format(file_format),
        bbox_inches='tight',
        pad_inches=0.1,
        format=file_format,
        transparant=True,
        dpi=dpi,
    )

    axes = pm.all_step_ahead_simulation(plant, LENGTH, ac, INIT_STATE)
    axes[0].figure.set_size_inches(8, 14)
    axes[0].figure.savefig(
        base_path + "plant_model_quality_all_step_ahead.{}".format(file_format),
        bbox_inches='tight',
        pad_inches=0.1,
        format=file_format,
        transparant=True,
        dpi=dpi,
    )
    axes = pm.all_step_ahead_errors(plant, LENGTH, ac, INIT_STATE)
    axes[0].figure.set_size_inches(8, 14)
    axes[0].figure.savefig(
        base_path + "plant_model_quality_all_step_ahead_err.{}".format(file_format),
        bbox_inches='tight',
        pad_inches=0.1,
        format=file_format,
        transparant=True,
        dpi=dpi,
    )
