import matplotlib.pyplot as plt
import os

from ..abstract_object import AbstractObject
from ..env import FIGURE_PATH
from ..validation.object_validation import assert_true


class AbstractVisualizer(AbstractObject):
    """
    The AbstractVisualizer is the base class for all visualizer classes, realtime and static. It defines basic stuff 
    such as method for getting the Axes, and its properties. Also it has a method for saving figures to disk.
    """
    DEFAULT_COLOR = "dodgerblue"
    DEFAULT_LINE_STYLE = "-"
    EXPORT_SIZES = {
        "report": (14, 10.5),
        "report-2pp": (14, 7.875),
        "report-3pp": (14, 6),
        "portrait": (10.5, 14),
        "ts-long": (8, 14),
    }

    def __init__(self, axis):
        super(AbstractVisualizer, self).__init__()
        self._axis = plt.subplots()[1] if axis is None else axis
        self._name = "Figure"
        try:
            self._figure = self._axis.figure
        except AttributeError:
            try:
                self._figure = self._axis[0, 0].figure
            except IndexError:
                self._figure = self._axis[0].figure

    def _key(self):
        raise NotImplementedError

    def get_figure(self):
        """
        :return: Returns Matplotlib Figure object 
        """
        return self._figure

    def get_axis(self):
        """
        :return: Returns Matplotlib Axes object 
        """
        return self._axis

    def get_ylim(self):
        """
        :return: Returns y_start and y_end as tuple 
        """
        return self._axis.get_ylim()

    def get_xlim(self):
        """
        :return: Returns x_start and x_end as tuple 
        """
        return self._axis.get_xlim()

    def save(self, size, target=None, *args, **kwargs):
        """
        Saves plot of frozen motion to disk in figs_path
        :param size: paper dimensions in inches as tuple
        :param target: target folder as string
        :param args: 
        :param kwargs: 
        """
        assert_true(size in self.EXPORT_SIZES, "Unknown size {}".format(size))
        self._figure.set_size_inches(*self.EXPORT_SIZES[size])
        self._figure.savefig(
            os.path.join(FIGURE_PATH, self._name) if target is None else target,
            bbox_inches='tight',
            pad_inches=0.1,
            *args,
            **kwargs
        )
