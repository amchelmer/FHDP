def assert_is_subclass(obj, parent_type):
    if not issubclass(obj, parent_type):
        raise TypeError("Object '{}' is of type '{}', but a subclass of '{}'.".format(
            obj,
            type(obj),
            parent_type)
        )


def assert_list_of_subclass(lst, parent_type):
    for obj in lst:
        assert_is_subclass(obj, parent_type)


def assert_is_type(obj, expected_type):
    if not isinstance(obj, expected_type):
        raise TypeError("Object '{}' is of type '{}', but expecting type '{}'.".format(
            obj,
            type(obj),
            expected_type)
        )


def assert_type_in(obj, expected_types):
    if not any([isinstance(obj, expected_type) for expected_type in expected_types]):
        raise TypeError("Object '{}' is of type '{}', but expecting one of the following types '{}'.".format(
            obj,
            type(obj),
            expected_types)
        )


def assert_list_of_type(lst, expected_type):
    for obj in lst:
        assert_is_type(obj, expected_type)


def assert_list_of_types(lst, expected_types):
    for obj in lst:
        assert_type_in(obj, expected_types)
