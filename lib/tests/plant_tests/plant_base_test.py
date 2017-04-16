import numpy as np

from ..eq_and_hash_base_test import EqAndHashBaseTest

from numpy.testing import assert_array_almost_equal, assert_array_equal


class PlantBaseTest(EqAndHashBaseTest):
    SEED = 1803
    RANDOM = np.array([-0.211105838, 0.306614385, -1.106875617, 1.650906526, -0.248679715, 0.448056975, 1.091740717,
                       -0.492293311, -1.295319425, -1.002650526, 1.119139609, -0.038033527, -0.271572789, 0.555476239,
                       -0.457411939, 1.018206949, -0.065453654, 1.617702874, 0.269887626, -0.337770054])
    STATE = None
    PREPROCESSED_STATE = None
    ACTION = None
    PREPROCESSED_ACTION = None
    OUT_OF_BOUNDS_STATE = None
    NEXT_STATE = None
    DERIVATIVE = None
    INITIAL_STATE = None
    LIKE_ME_ARRAY = None
    LIKE_ME_OTHER_FEATURE_SET = None
    SIMULATION_STATES = None
    SIMULATION_ACTIONS = None
    SIMULATION_LENGTH = None
    SIMULATION_CONTROLLER = None
    SIMULATION_INITIAL_STATE = None

    @staticmethod
    def _get_plant_cls():
        raise NotImplementedError

    def _get_plant_parameters(self):
        raise NotImplementedError

    def _get_other_plant_parameters(self):
        raise NotImplementedError

    def _generate_plant(self):
        plant_cls = self._get_plant_cls()
        return plant_cls(**self._get_plant_parameters())

    def _generate_other_plant(self):
        plant_cls = self._get_plant_cls()
        return plant_cls(**self._get_other_plant_parameters())

    def _plant_base_test(self):
        self._test_eq_and_hash()
        self._test_get_visualizer_cls()
        self._test_set_mod()
        self._test_compute_derivative()
        self._test_get_next_state()
        self._test_get_feature_set()
        self._test_get_time_step()
        self._test_get_number_of_state_vars()
        self._test_get_number_of_action_vars()
        self._test_get_initial_state()
        self._test_get_state_labels()
        self._test_get_action_labels()
        self._test_like_me()
        self._test_simulate()
        self._test_get_state_modulus()
        self._test_is_out_of_bounds()
        self._test_get_bounds()

    def _test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_plant,
            self._generate_other_plant
        )

    def _test_get_visualizer_cls(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_visualizer_cls(),
            plant._VISUALIZER_CLS
        )

    def _test_set_mod(self):
        plant = self._generate_plant()
        assert_array_equal(
            plant._state_modulus,
            plant._MOD,
        )
        plant.set_mod(False)
        self.assertEqual(
            plant._state_modulus,
            None
        )

    def _test_compute_derivative(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant.compute_derivative(
                self.PREPROCESSED_STATE,
                self.PREPROCESSED_ACTION
            ),
            self.DERIVATIVE,
            decimal=7
        )

    def _test_get_next_state(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant.get_next_state(
                self.STATE,
                self.ACTION
            ),
            self.NEXT_STATE,
            decimal=8
        )

    def _test_get_feature_set(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_feature_set(),
            plant._FEATURE_SET
        )

    def _test_get_time_step(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_time_step(),
            plant._time_step
        )

    def _test_get_number_of_state_vars(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_number_of_state_vars(),
            len(plant._FEATURE_SET.get_state_set())
        )

    def _test_get_number_of_action_vars(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_number_of_action_vars(),
            len(plant._FEATURE_SET.get_action_set())
        )

    def _test_get_initial_state(self):
        plant = self._generate_plant()
        plant.set_rng(self.SEED)
        r = self.RANDOM[:plant.get_number_of_state_vars()].reshape(-1, 1)
        assert_array_almost_equal(
            plant.get_initial_state(),
            plant._initial_state_mean + plant._initial_state_sigma * r
        )

    def _test_get_state_labels(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_state_labels(),
            plant._FEATURE_SET.get_state_set().get_names(),
        )

    def _test_get_action_labels(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_action_labels(),
            plant._FEATURE_SET.get_action_set().get_names(),
        )

    def _test_like_me(self):
        plant = self._generate_plant()
        assert_array_almost_equal(
            plant.like_me(
                self.LIKE_ME_ARRAY,
                self.LIKE_ME_OTHER_FEATURE_SET,
            ),
            plant._FEATURE_SET.like_me(
                self.LIKE_ME_ARRAY,
                self.LIKE_ME_OTHER_FEATURE_SET,
            )
        )

    def _test_simulate(self):
        plant = self._generate_plant()
        simulation = plant.simulate(
            self.SIMULATION_LENGTH,
            self.SIMULATION_CONTROLLER,
            self.SIMULATION_INITIAL_STATE,
        )
        assert_array_almost_equal(
            simulation.get_states(),
            self.SIMULATION_STATES,
            decimal=8
        )
        assert_array_almost_equal(
            simulation.get_actions(),
            self.SIMULATION_ACTIONS,
            decimal=8
        )

    def _test_state_generator(self):
        plant = self._generate_plant()
        sim_states, sim_actions = zip(
            *[(state, action) for (state, action) in plant._state_generator(
                self.SIMULATION_LENGTH,
                self.SIMULATION_CONTROLLER,
                self.INITIAL_STATE,
            )]
        )
        assert_array_almost_equal(
            np.concatenate(sim_states, axis=1),
            self.SIMULATION_STATES,
            decimal=8
        )
        assert_array_almost_equal(
            np.concatenate(sim_actions, axis=1),
            self.SIMULATION_ACTIONS,
            decimal=8
        )

    def _test_get_state_modulus(self):
        plant = self._generate_plant()
        self.assertEqual(
            plant.get_state_modulus(),
            plant._state_modulus,
        )

    def _test_is_out_of_bounds(self):
        plant = self._generate_plant()
        self.assertTrue(plant.is_out_of_bounds(self.OUT_OF_BOUNDS_STATE))
        self.assertFalse(plant.is_out_of_bounds(self.STATE))

    def _test_get_bounds(self):
        plant = self._generate_plant()
        assert_array_equal(
            plant.get_bounds(),
            plant._bounds,
        )
