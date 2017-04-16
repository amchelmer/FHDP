from .object_validation import assert_has_attribute


def assert_shape(obj, expected_shape):
    assert_has_attribute(obj, "shape")
    if not obj.shape == expected_shape:
        raise AssertionError(
            "Expected shape {}, got {}.".format(
                expected_shape,
                obj.shape
            )
        )


def assert_shape_like(obj, obj_with_desired_shape,):
    assert_has_attribute(obj_with_desired_shape, "shape")
    assert_shape(obj, obj_with_desired_shape.shape)


def assert_same_length(*objects):
    assert_has_attribute(objects, "__len__")
    assert_has_attribute(objects[0], "__len__")
    for obj in objects[1:]:
        assert_has_attribute(obj, "__len__")
        if len(objects[0]) != len(obj):
            raise AssertionError(
                "Length of object {} is {}, while length of object {} is {}".format(
                    objects[0],
                    len(objects[0]),
                    obj,
                    len(obj)
                )
            )


def assert_length(obj, length):
    if not len(obj) == length:
        raise AssertionError(
            "Length of object is {}, expected {}.".format(len(obj), length)
        )


def assert_list_of_value(lst, value=None):
    if value is None:
        value = lst[0]
    for item in lst:
        if not item == value:
            raise AssertionError(
                "Object {} is not equal to {}".format(item, value)
            )
