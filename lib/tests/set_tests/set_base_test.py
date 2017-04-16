import tempfile

from ..eq_and_hash_base_test import EqAndHashBaseTest


class SetBaseTest(EqAndHashBaseTest):
    OBJECT_IN_SET = NotImplemented
    OBJECT_NOT_IN_SET = NotImplemented

    def _get_set_cls(self):
        raise NotImplementedError

    def _get_set_kwargs(self):
        raise NotImplementedError

    def _get_other_set_kwargs(self):
        raise NotImplementedError

    def _generate_set_instance(self):
        return self._get_set_cls()(**self._get_set_kwargs())

    def _generate_other_set_instance(self):
        return self._get_set_cls()(**self._get_other_set_kwargs())

    def _set_base_test(self):
        self._test_eq_and_hash()
        self._test__len__()
        self._test__iter__()
        self._test__getitem__()
        self._test__contains__()
        self._test_get_iterable()
        self._test_dump_load()

    def _test_eq_and_hash(self):
        self.assert_eq_and_hash_implemented_correctly(
            self._generate_set_instance,
            self._generate_other_set_instance
        )

    def _test__len__(self):
        set_instance = self._generate_set_instance()
        self.assertEqual(
            len(set_instance),
            len(set_instance._iterable)
        )

    def _test__iter__(self):
        set_instance = self._generate_set_instance()
        self.assertEqual(
            [c for c in set_instance],
            list(set_instance._iterable),
        )

    def _test__getitem__(self):
        set_instance = self._generate_set_instance()
        self.assertEqual(
            set_instance[:2],
            self._get_set_cls()(set_instance._iterable[:2])
        )
        self.assertEqual(
            set_instance[2],
            set_instance._iterable[2]
        )

    def _test__contains__(self):
        set_instance = self._generate_set_instance()
        self.assertTrue(self.OBJECT_IN_SET in set_instance)
        self.assertFalse(self.OBJECT_NOT_IN_SET in set_instance)

    def _test_get_iterable(self):
        set_instance = self._generate_set_instance()
        self.assertEqual(
            set_instance.get_iterable(),
            set_instance._iterable
        )

    def _test_dump_load(self):
        file_handle = tempfile.TemporaryFile()
        set_instance = self._generate_set_instance()
        set_instance.dump(file_handle)

        file_handle.seek(0)
        simulation_loaded = self._get_set_cls().load(file_handle)
        self.assertEqual(
            set_instance,
            simulation_loaded
        )
