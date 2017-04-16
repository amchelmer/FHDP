import numpy as np
import re

from multiprocessing.pool import Pool


def parallelize(threads, func, arg_list):
    """
    Parallelizes a function over multiple cores.
    :param threads: number of parallel jobs
    :param func: function to be parallelized
    :param arg_list: list of arguments to be passed to func
    :return: 
    """
    pool = Pool(processes=threads)
    results = pool.map(func, arg_list)
    pool.close()
    pool.join()
    return results


def extract_ids(s):
    """
    Extracts ids from filenames. Useful when parsing arguments.
    :param s: 
    :return: 
    """
    l = s.split()
    return [int(
        re.search("(?<=-)\d+", x).group()
    ) for x in l]


def matlab_string_to_matrix(s):
    """
    Converts a string to numpy.matrix. Allows copy-pasting of MATLAB matrices.
    :param s: matrix as string (copied from MATLAB).
    :return: numpy.matrix
    """
    while "  " in s:
        s = s.replace("  ", " ")
    return np.matrix(s.replace("\n ", "; "))
