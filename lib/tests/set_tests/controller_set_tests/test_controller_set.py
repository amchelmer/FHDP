from ..set_base_test import SetBaseTest


class TestControllerSet(SetBaseTest):
    OBJECT_IN_SET = NotImplemented
    OBJECT_NOT_IN_SET = NotImplemented

    def _get_set_cls(self):
        raise NotImplementedError

    def _get_set_kwargs(self):
        raise NotImplementedError

    def _get_other_set_kwargs(self):
        raise NotImplementedError

    def test_get_controller_ids(self):
        raise NotImplementedError

    def test_get_train_results(self):
        raise NotImplementedError

    def test_shrink(self):
        raise NotImplementedError

    def test_lookback_result(self):
        raise NotImplementedError

    def test_get_best(self):
        raise NotImplementedError

