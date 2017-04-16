from ..eq_and_hash_base_test import EqAndHashBaseTest


class RewardFunctionBaseTest(EqAndHashBaseTest):

    def _generate_reward_function(self):
        reward_function_cls = self._get_reward_function_cls()
        return reward_function_cls(**self._get_reward_kwargs())

    def _generate_other_reward_function(self):
        reward_function_cls = self._get_reward_function_cls()
        return reward_function_cls(**self._get_other_reward_kwargs())

    def _get_reward_function_cls(self):
        raise NotImplementedError

    def _get_reward_kwargs(self):
        raise NotImplementedError

    def _get_other_reward_kwargs(self):
        raise NotImplementedError

    def _reward_function_base_test(self):
        self._test_eq_and_hash()
        self._test_get_reward()
        self._test_get_derivative_to_action()

    def _test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_reward_function,
            self._generate_other_reward_function
        )

    def _test_get_reward(self):
        raise NotImplementedError

    def _test_get_derivative_to_action(self):
        raise NotImplementedError
