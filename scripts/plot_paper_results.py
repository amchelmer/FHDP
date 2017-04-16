import os

from lib.env import FIGURE_PATH
from lib.sets import ControllerSet, RewardSet

ids = [
    # 1490430446140349218195024,
    123,
    1491629602139719847629648,
]

targets = [
    "results-static",
    "results-flex-gaussian",
]

base_path = FIGURE_PATH
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
        minimum=12000,
    )
    vis.save(
        "report-3pp",
        target=os.path.join(base_path, t + "-{}.{}".format(len(rc), file_format)),
        format=file_format,
        transparant=True,
        dpi=dpi,
    )
