from .type_validation import assert_is_type


def assert_has_attribute(obj, attr):
    assert_is_type(attr, str)
    if not hasattr(obj, attr):
        raise AssertionError(
            "Object {} has no attribute {}".format(obj, attr)
        )


def assert_in(obj, iterable):
    if obj not in iterable:
        raise AssertionError(
            "Object {} does not exist in {}".format(obj, iterable)
        )


def assert_not_in(obj, iterable):
    if obj in iterable:
        raise AssertionError(
            "Object {} exists in {}".format(obj, iterable)
        )


def assert_unique(iterable):
    if not len(iterable) == len(set(iterable)):
        raise AssertionError(
            "Iterable {} does not contain unique items".format(iterable)
        )


def assert_true(test, error):
    assert test, error


def assert_false(test, error):
    assert_true(not test, error)


def assert_equal(a, b):
    if not a == b:
        raise AssertionError("Object {} and object {} are not equal".format(a, b))


def assert_not_equal(a, b):
    if a == b:
        raise AssertionError("Object {} and object {} are equal".format(a, b))
