# Author: Garcia-Cifuentes, K. <ORCID:0009-0001-2607-6359>
# Author: Becerra, R. L. <ORCID:0000-0002-0216-3415>
# Author: De Colle, F. <ORCID:0000-0002-3137-4633>
# License: GNU General Public License version 2 (1991)

# This file contains second-level functions to manipulate and visualize data from Swift/BAT data
# Details about Swift Data can be found in https://swift.gsfc.nasa.gov/about_swift/bat_desc.html

import os
import sklearn
import numpy as np
import pandas as pd
from time import time
from numpy import linalg
from typing import Union
from sklearn.manifold import TSNE
from scipy.signal import convolve2d
from collections.abc import Sequence, Mapping


def save_data(
        data: pd.DataFrame,
        name: str,
        filename: str,
        directory: str
):
    """Function to save data in a hdf5 file.

    Args:
        data (pd.DataFrame): Data to save.
        name (str): Name of the data.
        filename (str): Name of the file.
        directory (str): Path where to save the file.

    Returns:
        None
    """
    if not os.path.exists(directory):
        directory_maker(directory)
    path = os.path.join(directory, filename)
    os.remove(path) if os.path.exists(path) else None
    data.to_hdf(path_or_buf=path, key=name, complevel=0, format='fixed')


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
            array_copy = np.delete(array_copy, [i for i in range(length - 1)])
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
        raise ValueError(f"Array does not contain any element lower or equal to {other[0]}.")
    elif len(index2) == 0:
        if np.allclose(array[-1], other[-1]):
            index2 = [len(array)]
        else:
            raise ValueError(f"Array does not contain any element higher or equal to {other[-1]}.")
    return index[-1], index2[0] + 1


def estimate_noise(data):
    """ Estimate the RMS noise of an image

    Taked from https://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
    Reference: J. Immerkaer, “Fast Noise Variance Estimation”, Computer Vision and Image Understanding,
    Vol. 64, No. 2, pp. 300-302, Sep. 1996 [PDF]
    """
    H, W = data.shape
    data = np.nan_to_num(data)
    M = [[1, -2, 1],
         [-2, 4, -2],
         [1, -2, 1]]
    sigma = np.sum(np.sum(np.abs(convolve2d(data, M))))
    sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W - 2) * (H - 2))
    return sigma


def size_maker(z_array):
    """Function to get redshift values without raising an error."""
    result_array = []
    for value in z_array:
        try:
            result_array.append(float(value[1]))
        except (ValueError, IndexError, TypeError):
            result_array.append(0)
    return np.array(result_array)


def get_steps(data, **step_kwargs):
    """Function to get the of steps in a TSNE embedding of scikit-learn.

    This function uses the _gradient_descent function from the scikit-learn library version 1.2.2:
    (https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/manifold/_t_sne.py).
    The original function was written by L. van der Maaten and G. E. Hinton and is licensed under the BSD 3-Clause
    License. Additionally, the instance implemented here is based on the tsne_animate repository on GitHub:
    (https://github.com/sophronesis/tsne_animate). We are grateful to the authors for their work.

    Args:
        data (array-like): Array with the data to be embedded.
        step_kwargs (dict): Keyword arguments to be passed to the TSNE instance of scikit-learn.

    Returns:
        A list of positions of the data in the embedding space for each iteration.
    """
    old_gradient = sklearn.manifold._t_sne._gradient_descent  # Save original gradient descent function
    positions = []  # Array to save data positions

    def _gradient_descent(
            objective,
            p0,
            it,
            n_iter,
            n_iter_check=1,
            n_iter_without_progress=300,
            momentum=0.8,
            learning_rate=200.0,
            min_gain=0.01,
            min_grad_norm=1e-7,
            verbose=0,
            args=None,
            kwargs=None,
    ):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(float).max
        best_error = np.finfo(float).max
        best_iter = i = it

        tic = time()
        for i in range(it, n_iter):
            positions.append(p.copy())  # Save current position

            check_convergence = (i + 1) % n_iter_check == 0
            # only compute the error when needed
            kwargs["compute_error"] = check_convergence or i == n_iter - 1

            error, grad = objective(p, *args, **kwargs)

            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            if check_convergence:
                toc = time()
                duration = toc - tic
                tic = toc
                grad_norm = linalg.norm(grad)

                if verbose >= 2:
                    print(
                        "[t-SNE] Iteration %d: error = %.7f,"
                        " gradient norm = %.7f"
                        " (%s iterations in %0.3fs)"
                        % (i + 1, error, grad_norm, n_iter_check, duration)
                    )

                if error < best_error:
                    best_error = error
                    best_iter = i
                elif i - best_iter > n_iter_without_progress:
                    if verbose >= 2:
                        print(
                            "[t-SNE] Iteration %d: did not make any progress "
                            "during the last %d episodes. Finished."
                            % (i + 1, n_iter_without_progress)
                        )
                    break
                if grad_norm <= min_grad_norm:
                    if verbose >= 2:
                        print(
                            "[t-SNE] Iteration %d: gradient norm %f. Finished."
                            % (i + 1, grad_norm)
                        )
                    break

        return p, error, i

    sklearn.manifold._t_sne._gradient_descent = _gradient_descent  # Change original gradient function
    TSNE(random_state=42, **step_kwargs).fit_transform(data)  # Perform TSNE
    sklearn.manifold._t_sne._gradient_descent = old_gradient  # Return old gradient descent function
    return np.array(positions)  # Return all positions, texts in format (iteration, message)
