import os

from lib.env import FIGURE_PATH
from lib.sets import ControllerSet, RewardSet

ids = [
    1491596371139722462109584,
    1491605939139721360447184,
    1491617120139720080092752,
    1491629602139719847629648,
    1491637674139722409805584
]

ms = [
    18000,
    9000,
    9000,
    9000,
    9000
]

targets = [
    "additional-result-expansion-method-zero",
    "additional-result-expansion-method-perturb",
    "additional-result-expansion-method-uniform",
    "additional-result-expansion-method-gaussian",
    "additional-result-expansion-method-separate",
]

base_path = FIGURE_PATH
file_format = "pdf"
dpi = 300

for i, m, t in zip(ids, ms, targets):
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
        minimum=m,
    )
    vis.save(
        "report-3pp",
        target=os.path.join(base_path, t + "-{}.{}".format(len(rc), file_format)),
        format=file_format,
        transparant=True,
        dpi=dpi,
    )
