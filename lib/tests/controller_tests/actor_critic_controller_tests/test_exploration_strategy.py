from ...eq_and_hash_base_test import EqAndHashBaseTest
from ....controllers.actor_critic_controller import ExplorationStrategy


class TestExplorationStrategy(EqAndHashBaseTest):
    EXPLORATION_DICT = {1: 2, 5: 3, 10: 8}
    OTHER_EXPLORATION_DICT = {1: 2, 4: 3, 12: 6}

    def _generate_exploration_strategy(self):
        return ExplorationStrategy(self.EXPLORATION_DICT)

    def _generate_other_exploration_strategy(self):
        return ExplorationStrategy(self.OTHER_EXPLORATION_DICT)

    def test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_exploration_strategy,
            self._generate_other_exploration_strategy,
        )

    def test__get_item__(self):
        es = self._generate_exploration_strategy()
        self.assertRaises(
            KeyError,
            es.__getitem__,
            0
        )
        self.assertEqual(es[1], 2)
        self.assertEqual(es[4], 2)
        self.assertEqual(es[5], 3)
        self.assertEqual(es[9], 3)
        self.assertEqual(es[10], 8)
        self.assertEqual(es[100], 8)
