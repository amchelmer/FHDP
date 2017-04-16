import numpy as np
import tempfile

from ...simulations.simulation import Simulation
from ..eq_and_hash_base_test import EqAndHashBaseTest
from ...plants.quad_rotor_plant import QuadRotor2DPlant

from numpy.testing import assert_array_equal


class TestSimulation(EqAndHashBaseTest):
    DT = 0.01
    SYSTEM = QuadRotor2DPlant(DT)
    STATES = np.array([
        [0.5, -2.0, 3.1, 3 / 2.],
        [0.4, -1.8, 3.2, 4 / 2.],
        [0.3, -1.6, 3.3, 5 / 2.],
        [0.2, -2.2, 3.4, 6 / 2.],
    ])
    ACTIONS = np.arange(8).reshape(2, 4)
    VALUES = np.array([
        [-10, -5, -3, -1],
    ])
    REWARDS = np.array([
        [-3, -1, -0.5, -0.1]
    ])

    def _generate_simulation(self):
        return Simulation(self.SYSTEM, self.STATES, self.ACTIONS, self.VALUES, self.REWARDS)

    def _generate_other_simulation(self):
        return Simulation(self.SYSTEM, self.STATES, self.ACTIONS)

    def test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_simulation,
            self._generate_other_simulation
        )

    def test__len__(self):
        simulation = self._generate_simulation()
        self.assertEqual(
            len(simulation),
            4
        )

    def test__iter__(self):
        simulation = self._generate_simulation()
        for i, t in enumerate(simulation):
            assert_array_equal(self.STATES[:, i:i + 1], t[0])
            assert_array_equal(self.ACTIONS[:, i:i + 1], t[1])
            assert_array_equal(self.VALUES[:, i:i + 1], t[2])
            assert_array_equal(self.REWARDS[:, i:i + 1], t[3])

    def test_get_rewards(self):
        simulation = self._generate_simulation()
        assert_array_equal(
            simulation.get_rewards(),
            simulation._rewards
        )

    def test_get_actions(self):
        simulation = self._generate_simulation()
        assert_array_equal(
            simulation.get_actions(),
            simulation._actions
        )

    def test_get_state(self):
        simulation = self._generate_simulation()
        assert_array_equal(
            simulation.get_states(),
            simulation._states
        )

    def test_get_values(self):
        simulation = self._generate_simulation()
        assert_array_equal(
            simulation.get_values(),
            simulation._values
        )

    def test_get_total_reward(self):
        simulation = self._generate_simulation()
        self.assertEqual(
            simulation.get_total_reward(),
            -3 - 1 - 0.5 - 0.1
        )

    def test_get_plant(self):
        simulation = self._generate_simulation()
        self.assertEqual(
            simulation.get_plant(),
            simulation._plant
        )

    def test_get_time_vector(self):
        simulation = self._generate_simulation()
        assert_array_equal(
            simulation.get_time_vector(),
            np.array([0, 0.01,0.02,0.03])
        )

    def test_plot_time_series(self):
        pass

    def test_replay(self):
        pass

    def test_dump_and_load(self):
        file_handle = tempfile.TemporaryFile()
        simulation = self._generate_simulation()
        simulation.dump(file_handle)

        file_handle.seek(0)
        simulation_loaded = Simulation.load(file_handle)
        self.assertEqual(
            simulation,
            simulation_loaded
        )
