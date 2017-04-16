import numpy as np


def center_mod(state, mod_list):
    """
    Center mods around zero
    :param state:
    :param mod_list:
    :return:
    """
    if mod_list is None:
        return state
    for i, m in enumerate(mod_list):
        if m is False:
            pass
        else:
            state[i] = (state[i] + m / 2.) % m - m / 2.
    return state


def saturate(state, saturation):
    """
    Saturates a numpy object with a minimum and maximum
    :param state: numpy.array or numpy.matrix of shape (n, 1)
    :param saturation: numpy.array or numpy.matrix of shape (n, 2)
    :return: numpy object with shape (n, 1)
    """
    return state if saturation is None else state.clip(saturation[:, :1], saturation[:, 1:])


def euler2quat(euler_angles):
    """
    Converts Euler angles to quaternions
    :param euler_angles: angles in radians
    :return: quaternions
    """
    phi, theta, psi = euler_angles.flatten() / 2.

    cphi, ctheta, cpsi = np.cos([phi, theta, psi])
    sphi, stheta, spsi = np.sin([phi, theta, psi])
    quaternions = np.array([
        [cphi * ctheta * cpsi + sphi * stheta * spsi],
        [sphi * ctheta * cpsi - cphi * stheta * spsi],
        [cphi * stheta * cpsi + sphi * ctheta * spsi],
        [cphi * ctheta * spsi - sphi * stheta * cpsi]
    ])
    return normalize_quaternions(quaternions)


def quat2euler(quaternions):
    """
    Converts quaternions to Euler angles. Catches a rare singular case.
    :param quaternions: numpy.array or numpy.matrix
    :return: numpy.array with shape (3,1)
    """
    qw, qx, qy, qz = normalize_quaternions(quaternions).flatten()
    try:
        return np.array([
            np.arctan2(2 * (qw * qx + qy * qz), qw ** 2 + qz ** 2 - qx ** 2 - qy ** 2),
            np.arcsin(2 * (qw * qy - qz * qx)),
            np.arctan2(2 * (qw * qz + qx * qy), qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2),
        ]).reshape(3, 1)
    except FloatingPointError:
        try:
            qw, qx, qy, qz = normalize_quaternions(quaternions + 1e-3).flatten()
            return np.array([
                np.arctan2(2 * (qw * qx + qy * qz), qw ** 2 + qz ** 2 - qx ** 2 - qy ** 2),
                np.arcsin(2 * (qw * qy - qz * qx)),
                np.arctan2(2 * (qw * qz + qx * qy), qw ** 2 + qx ** 2 - qy ** 2 - qz ** 2),
            ]).reshape(3, 1)
        except FloatingPointError:
            raise FloatingPointError(
                "Invalid value encountered with values: {:.16f}, {:.16f}, {:.16f}, {:.16f}".format(qw, qx, qy, qz)
            )


def normalize_quaternions(quaternions):
    """
    Normalize quaternions so they have a total length of 1
    :param quaternions: numpy.array or numpy.matrix
    :return: quaternions in same shape as input
    """
    return quaternions / np.linalg.norm(quaternions.flatten(), ord=2)


def hashable(obj):
    """
    Ensure object is hashable. Support for lists, numpy arrays and numpy matrices.
    :param obj: list, numpy.array or numpy.matrix
    :return: hashable object
    """
    try:
        hash(obj)
        obj_out = obj
    except TypeError:
        if isinstance(obj, np.ndarray):
            try:
                obj_out = tuple(obj.flatten())
            except AttributeError:
                obj_out = tuple(obj.A.flatten())
        elif isinstance(obj, list):
            obj_out = tuple(obj)
        else:
            raise NotImplementedError("Unsupported type {} of object {}".format(
                type(obj),
                obj
            ))
    try:
        hash(obj_out)
    except TypeError:
        raise TypeError("Unable to hash object obj {}".format(obj))
    return obj_out
