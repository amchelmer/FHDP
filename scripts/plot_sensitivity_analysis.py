import os

from lib.env import FIGURE_PATH
from lib.sets import ControllerSet, RewardSet

ids = [
    1491657746140458221122640,
    1491672073140457147804432,
    1491686616140455501441680,
    1491701637140457168621456,
]

targets = [
    "sensitivity-analysis-alpha-lower",
    "sensitivity-analysis-alpha-higher",
    "sensitivity-analysis-gamma-lower",
    "sensitivity-analysis-gamma-higher",
]

BASE_PATH = FIGURE_PATH
file_format = "pdf"
dpi = 300

for i, t in zip(ids, targets):
    try:
        rc = RewardSet.load(i)
    except IOError:
        rc = RewardSet(ControllerSet.load(i))
        rc.dump()
    print("ID: {}".format(i))
    vis = rc.plot(
        conf=68,
        bounds=False,
        metric="median",
        minimum=9000,
    )
    vis.save(
        "report-3pp",
        target=os.path.join(BASE_PATH, t + "-{}.{}".format(len(rc), file_format)),
        format=file_format,
        transparant=True,
        dpi=dpi,
    )
