import numpy as np

from ..set_base_test import SetBaseTest
from ....sets import SimulationSet
from ....plants.quad_rotor_plant import QuadRotor2DPlant, QuadRotor3DPlant
from ....simulations import Simulation

from numpy.testing import assert_array_almost_equal


class TestSimulationSet(SetBaseTest):
    DT = 0.01
    SYSTEM = QuadRotor2DPlant(DT)

    SIMULATION_1 = Simulation(
        SYSTEM,
        np.array([
            [0.5, -2.0, 3.1, 3 / 2.],
            [0.4, -1.8, 3.2, 4 / 2.],
            [0.3, -1.6, 3.3, 5 / 2.],
            [0.2, -2.2, 3.4, 6 / 2.],
        ]),
        np.arange(8).reshape(2, 4),
        np.array([[-15, -4, -2, -1]]),
        np.array([[-3, -1, -0.5, -0.1]])
    )
    SIMULATION_2 = Simulation(
        SYSTEM,
        np.array([
            [-6.54321427, 6.1089904, -10.72152071, -0.46258695],
            [-14.92345621, -5.59148855, 20.64284906, 11.37941275],
            [12.15810213, 9.54029269, -13.11787504, 8.35383212],
            [1.00058595, -12.74399473, -2.63719739, 7.59025181]
        ]),
        np.arange(1, 9).reshape(2, 4),
        np.array([[-20, -10, -2, -1]]),
        np.array([[-5, -2, -0.5, -0.1]])
    )
    SIMULATION_3 = Simulation(
        SYSTEM,
        np.array([
            [0.5, -2.0, 3.1, 3 / 2.],
            [0.4, -1.8, 3.2, 4 / 2.],
            [0.3, -1.6, 3.3, 5 / 2.],
            [0.2, -2.2, 3.4, 6 / 2.],
        ]),
        np.arange(2, 10).reshape(2, 4),
        np.array([[-10, -5, -3, -1]]),
        np.array([[-1, -1, -0.5, -0.1]])
    )
    OBJECT_IN_SET = SIMULATION_1
    OBJECT_NOT_IN_SET = Simulation(
        SYSTEM,
        np.array([
            [0.4, -2.0, 3.1, 3 / 2.],
            [0.123, -1.8, 3.2, 4 / 2.],
            [0.4315, -1.6, 3.3, 5 / 2.],
            [0.35221, -2.2, 3.4, 6 / 2.],
        ]),
        np.arange(3, 11).reshape(2, 4),
        np.array([[10, -5, -3, -1]]),
        np.array([[1, 1, +0.5, -0.1]])
    )

    def _get_set_cls(self):
        return SimulationSet

    def _get_set_kwargs(self):
        return {
            "simulations": [self.SIMULATION_1, self.SIMULATION_2, self.SIMULATION_3],
        }

    def _get_other_set_kwargs(self):
        return {
            "simulations": [self.SIMULATION_1, self.SIMULATION_2],
        }

    def set_base_test(self):
        self._set_base_test()

    def test_validate_parameters(self):
        set_cls = self._get_set_cls()
        self.assertRaises(
            TypeError,
            set_cls._validate_parameters,
            [self.SIMULATION_1, self.SIMULATION_2, self.SIMULATION_1.get_plant()],
        )
        self.assertRaises(
            AssertionError,
            set_cls._validate_parameters,
            [self.SIMULATION_1, self.SIMULATION_2, Simulation(
                QuadRotor3DPlant(self.DT),
                np.array([
                    [0.5, -2.0, 3.1, 3 / 2.],
                    [0.4, -1.8, 3.2, 4 / 2.],
                    [0.3, -1.6, 3.3, 5 / 2.],
                    [0.2, -2.2, 3.4, 6 / 2.],
                ]),
                np.arange(2, 10).reshape(2, 4),
                np.array([[-10, -5, -3, -1]]),
                np.array([[-1, -1, -0.5, -0.1]])
            )],
        )
        self.assertRaises(
            AssertionError,
            set_cls._validate_parameters,
            [self.SIMULATION_1, self.SIMULATION_2, Simulation(
                QuadRotor2DPlant(0.5),
                np.array([
                    [0.5, -2.0, 3.1, 3 / 2.],
                    [0.4, -1.8, 3.2, 4 / 2.],
                    [0.3, -1.6, 3.3, 5 / 2.],
                    [0.2, -2.2, 3.4, 6 / 2.],
                ]),
                np.arange(2, 10).reshape(2, 4),
                np.array([[-10, -5, -3, -1]]),
                np.array([[-1, -1, -0.5, -0.1]])
            )],
        )
        self.assertRaises(
            AssertionError,
            set_cls._validate_parameters,
            [self.SIMULATION_1, self.SIMULATION_2, Simulation(
                QuadRotor2DPlant(0.5),
                np.array([
                    [0.5, -2.0, 3.1, 3 / 2., 0.1],
                    [0.4, -1.8, 3.2, 4 / 2., 0.1],
                    [0.3, -1.6, 3.3, 5 / 2., 0.1],
                    [0.2, -2.2, 3.4, 6 / 2., 0.1],
                ]),
                np.arange(2, 12).reshape(2, 5),
                np.array([[-10, -5, -3, -1, 0.1]]),
                np.array([[-1, -1, -0.5, -0.1, 0.4]])
            )],
        )

    def test_get_plant(self):
        simulation_set = self._generate_set_instance()
        self.assertEqual(
            simulation_set.get_plant(),
            simulation_set[0].get_plant(),
        )

    def test_get_best_simulation(self):
        simulation_set = self._generate_set_instance()
        self.assertEqual(
            simulation_set.get_best_simulation(),
            simulation_set[2],
        )

    def test_get_worst_simulation(self):
        simulation_set = self._generate_set_instance()
        self.assertEqual(
            simulation_set.get_worst_simulation(),
            simulation_set[1],
        )

    def test_get_total_reward_list(self):
        simulation_set = self._generate_set_instance()
        assert_array_almost_equal(
            simulation_set.get_total_reward_list(),
            np.array([
                -4.6,
                -7.6,
                -2.6
            ]),
        )

    def test_plot_total_rewards(self):
        simulation_set = self._generate_set_instance()
        simulation_set.plot_total_rewards()

    def test_replay(self):
        pass
