import argparse

from lib.sets.controller_set.controller_set import ControllerSet
from lib.sets.reward_set.reward_set import RewardSet
from lib.tools.runtime_tools import extract_ids

parser = argparse.ArgumentParser(description='ID parser')
parser.add_argument('ids', type=str, help='string of id', nargs="+")
args = parser.parse_args()

ids = [extract_ids(i)[0] for i in args.ids]

print("\n")
for i in ids:
    try:
        rc = RewardSet.load(i)
    except IOError:
        cs = ControllerSet.load(i)
        rc = RewardSet(cs)
        rc.dump()
    print("ID: {}".format(i))
    vis = rc.plot()
    vis.save("report-2pp")
