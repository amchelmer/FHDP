import matplotlib.pyplot as plt
import numpy as np

from ..abstract_static_visualizer import AbstractStaticVisualizer
from ....validation.object_validation import assert_in


class LearningProcessStaticVisualizer(AbstractStaticVisualizer):
    """
    Visualizer class to plot the learning process of a set of RL controllers. Requires a RewardSet object to 
    instantiate.
    """
    METRICS = {
        "mean": np.mean,
        "median": np.median,
    }

    def __init__(self, reward_set, axis=None, conf=68, bounds=False, metric="median", minimum=15e3):
        self._rs = reward_set
        super(LearningProcessStaticVisualizer, self).__init__(axis, conf=conf, bounds=bounds, metric=metric,
                                                              minimum=minimum)

    def _key(self):
        raise NotImplementedError

    def _plot(self, conf, bounds, metric, minimum):
        assert_in(metric, self.METRICS)
        rewards = self._rs.get_training_rewards()
        x_vector = np.arange(rewards.shape[1]) + 1

        if bounds:
            self._axis.plot(
                x_vector,
                rewards.min(axis=0),
                color="tomato",
                linestyle="--",
                label="Min"
            )
            self._axis.plot(
                x_vector,
                rewards.max(axis=0),
                color="tomato",
                linestyle="--",
                label="Max"
            )
        self._axis.plot(
            x_vector,
            self.METRICS[metric](rewards, axis=0),
            color="k",
            linestyle="-",
            linewidth=1.5,
            label="Median"
        )
        self._axis.fill_between(
            x_vector,
            np.percentile(rewards, 100 - conf, axis=0),
            np.percentile(rewards, conf, axis=0),
            facecolor="dodgerblue",
            alpha=0.5,
            lw=0,
            label=r"1$\sigma$ conf."
        )

        self._axis.set_ylabel("Cumulative reward per episode [-]")
        self._axis.set_xlabel("Training episode number [-]")
        self._axis.set_xlim(x_vector[0], x_vector[-1])
        self._axis.set_ylim(-minimum, 0)
        plt.legend(loc="lower right")

        self._name = "{:s}-{:d}-{:d}".format(
            self._rs.__class__.__name__,
            self._rs.get_id(),
            len(self._rs)
        )
