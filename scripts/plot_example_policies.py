import os

from lib.env import FIGURE_PATH
from lib.simulations import Simulation

BASE_PATH = FIGURE_PATH
file_format = "pdf"
dpi = 300

best = Simulation.load(1491745618140034336380112)
medium = Simulation.load(1491745602140034336380112)
worst = Simulation.load(1491745566140034336379216)

best.static_replay().save(
    "ts-long",
    target=os.path.join(
        BASE_PATH,
        "example-policy-best-replay.{}".format(file_format),
    ),
    format=file_format,
    transparant=True,
    dpi=dpi,
)
best.plot_time_series(value=False, reward=False).save(
    "ts-long",
    target=os.path.join(
        BASE_PATH,
        "example-policy-best-ts.{}".format(file_format),
    ),
    format=file_format,
    transparant=True,
    dpi=dpi,
)

medium.static_replay().save(
    "ts-long",
    target=os.path.join(
        BASE_PATH,
        "example-policy-medium-replay.{}".format(file_format),
    ),
    format=file_format,
    transparant=True,
    dpi=dpi,
)
medium.plot_time_series(value=False, reward=False).save(
    "ts-long",
    target=os.path.join(
        BASE_PATH,
        "example-policy-medium-ts.{}".format(file_format),
    ),
    format=file_format,
    transparant=True,
    dpi=dpi,
)

worst.static_replay().save(
    "ts-long",
    target=os.path.join(
        BASE_PATH,
        "example-policy-worst-replay.{}".format(file_format),
    ),
    format=file_format,
    transparant=True,
    dpi=dpi,
)
worst.plot_time_series(value=False, reward=False).save(
    "ts-long",
    target=os.path.join(
        BASE_PATH,
        "example-policy-worst-ts.{}".format(file_format),
    ),
    format=file_format,
    transparant=True,
    dpi=dpi,
)
