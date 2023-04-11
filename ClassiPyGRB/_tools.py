# Author: Garcia-Cifuentes, K. <ORCID:0009-0001-2607-6359>
# Author: Becerra, R. L. <ORCID:0000-0002-0216-3415>
# Author: De Colle, F. <ORCID:0000-0002-3137-4633>
# License: GNU General Public License version 2 (1991)

# This file contains second-level functions to manipulate and visualize data from Swift/BAT data
# Details about Swift Data can be found in https://swift.gsfc.nasa.gov/about_swift/bat_desc.html

import os
import numpy as np
import pandas as pd
from typing import Union
from collections.abc import Sequence, Mapping


def directory_maker(
        path_save: str
):
    """Function to create a directory.
    Args:
        path_save (str): Path where to create directory.

    Returns:
        None
    """
    try:
        os.makedirs(path_save)
    except FileExistsError:
        pass


def check_name(
        name_i: str,
        table_array: Union[list, tuple, np.ndarray]
):
    """Function to check name in a table array.

    Args:
        name_i (str): Name to check in table.
        table_array (list): Table to search.

    Returns:
        Array with values from table_array associated with name_i
    """
    rows, columns_i = np.where(table_array == name_i)
    return table_array[rows]


def slice_array(
        array: Union[Sequence, Mapping, np.ndarray, pd.Series],
        length: int,
):
    """Slice an array into subarrays of a fixed length, preserving the initial points.

    It works like numpy.array_split, but the initial points between blocks are shared.

    Args:
        array (array-like): 1D array to be divided.
        length (int): Length of subarrays. It needs to be higher than 1.

    Returns:
        An array with n elements equal sized, but allowing non-equal size on the last block.

    Examples:
        >>> slice_array([0., 1., 2., 3., 4.], length=2)
        [array([0., 1.]), array([1., 2.]), array([2., 3.]), array([3., 4.])]
        >>> slice_array([0., 1., 2., 3., 4.], length=4)
        [array([0., 1., 2., 3.]), array([3., 4.])]
        >>> slice_array([0., 1., 2., 3., 4.], length=6)
        [array([0., 1., 2., 3., 4.])]
    """
    if not isinstance(length, int) or length <= 1:
        raise ValueError(f"Length is an integer higher than 1. Received an {type(length)} with value {length}.")
    if not isinstance(array, (Sequence, Mapping, np.ndarray, pd.Series)):
        raise ValueError(f"Array needs to be an 1D array-like (i.e., list, tuple). Received a {type(array)}.")
    slices = []
    array = np.asarray(array)
    array_copy = array
    while len(array_copy) != 0:
        elements = array_copy[:length]
        slices.append(elements)
        if len(array_copy) > length:
            array_copy = np.delete(array_copy, [i for i in range(length-1)])
        else:
            array_copy = np.delete(array_copy, np.s_[:])
    return slices


def get_index(array_1, array_2):
    """Get the index of the first element of the array_1 that is greater or equal to the first value of array_2.

    Args:
        array_1 (array-like): 1D array to be divided.
        array_2 (array-like): 1D array of reference.

    Returns:
        A tuple of two elements. The first element is the index of the last element of the array_1 that is lower or
        equal to the first value of array_2. The second element is the index of the first element of the array_1 that
        is greater or equal to the last value of array_2.

    Examples:
        >>> get_index([0., 1., 2., 3., 4.], [1.5, 3.5])
        (1, 5)
        >>> get_index([0., 1., 2., 3., 4.], [3.5])
        (3, 5)
    """
    if not isinstance(array_1, (Sequence, Mapping, np.ndarray, pd.Series)):
        raise ValueError(f"Array needs to be an 1D array-like (i.e., list, tuple). Received a {type(array_1)}.")
    if not isinstance(array_2, (Sequence, Mapping, np.ndarray, pd.Series)):
        raise ValueError(f"Other needs to be an 1D array-like (i.e., list, tuple). Received a {type(array_2)}.")
    array = np.asarray(array_1)
    other = np.asarray(array_2)
    if len(array) == 0:
        raise ValueError(f"Array_1 is empty. Try again with a different array.")
    if len(other) == 0:
        raise ValueError(f"Array_2 is empty. Try again with a different array.")
    index = np.where(array <= other[0])[0]
    index2 = np.where(array >= other[-1])[0]
    if len(index) == 0:
        if np.allclose(array[0], other[0]):
            print('Index es falso')
        raise ValueError(f"Array does not contain any element lower or equal to {other[0]}.")
    elif len(index2) == 0:
        if np.allclose(array[-1], other[-1]):
            index2 = [len(array)]
        else:
            raise ValueError(f"Array does not contain any element higher or equal to {other[-1]}.")
    return index[-1], index2[0]+1
