from ..eq_and_hash_base_test import EqAndHashBaseTest


class FunctionApproximatorBaseTest(EqAndHashBaseTest):

    def _get_function_approximator_cls(self):
        raise NotImplementedError

    def _get_function_approximator_kwargs(self):
        raise NotImplementedError

    def _generate_function_approximator(self):
        raise NotImplementedError

    def _generate_other_function_approximator(self):
        raise NotImplementedError

    def _function_approximator_base_test(self):
        self._test_eq_and_hash()

    def _test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_function_approximator,
            self._generate_other_function_approximator
        )
