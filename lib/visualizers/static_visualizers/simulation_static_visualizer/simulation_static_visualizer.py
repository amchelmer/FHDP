import matplotlib.pyplot as plt
import numpy as np

from ..abstract_static_visualizer import AbstractStaticVisualizer


class SimulationStaticVisualizer(AbstractStaticVisualizer):
    """
    Visualizer class to plot the states, actions, values and rewards of a Simulation in a time-series plot.
    """

    def __init__(self, simulation, value=True, reward=True):
        _, axes = plt.subplots(
            nrows=(
                simulation.get_states().shape[0] +
                simulation.get_actions().shape[0] +
                value +
                reward),
        )
        self._simulation = simulation
        super(SimulationStaticVisualizer, self).__init__(axes, value=True, reward=True)

    def _key(self):
        raise NotImplementedError

    def _plot(self, value, reward):
        self._name = "Simulation-output"
        plant = self._simulation.get_plant()
        sim = self._simulation

        ylabels = plant.get_state_labels() + plant.get_action_labels()
        output_list = [sim.get_states(), sim.get_actions()]
        time_vector = self._simulation.get_time_vector()

        if value:
            ylabels.append("Value")
            output_list.append(sim.get_values())
        if reward:
            ylabels.append("Reward")
            output_list.append(sim.get_rewards())

        for i in range(len(self._axis)):
            ax = self._axis[i]

            output = np.vstack(output_list)[i]
            ax.plot(
                time_vector,
                output,
                color="black",
            )
            ax.grid(True)
            ax.set_ylabel(ylabels[i])
            ax.set_xlim(0, time_vector[-2])
            if i < len(self._axis) - 1:
                ax.set_xticklabels(["" for _ in ax.get_xticklabels()])

        self._axis[-1].set_xlabel("Time [s]")
