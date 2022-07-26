"""A script containing majority of repetitive functions needed in Swift Data manipulation"""
import os  # Import os to handle folders and files
import numpy as np  # Import module to handle arrays
from scipy.signal import convolve2d  # Import module to noise estimation


def directory_maker(path_save):
    """Function to create a new directory in any specific path"""
    try:
        os.makedirs(path_save)
    except FileExistsError:
        pass


def length_of_array(array_i):
    """
    Function to check max length of data to be zero-padded
    :param array_i: List with n values
    :return: Length of array_i
    """
    try:
        return len(array_i)
    except TypeError:  # If error, send length 0 without bug the results, we are only seeing the max value
        return 0


def size_maker(z_array):
    result_array = []
    for value in z_array:
        try:
            result_array.append(float(value[1]))
        except (ValueError, IndexError, TypeError):
            result_array.append(0)
    return np.array(result_array)


def check_name(name_i, table_array):
    """
    Function to check name in a table array
    :param name_i: Name to check
    :param table_array: Table array
    :return: Values from name_i in table_array, if available
    """
    rows, columns_i = np.where(table_array == name_i)
    return table_array[rows]


def estimate_noise(data):
    """ Estimate the RMS noise of an image
    from http://stackoverflow.com/questions/2440504/noise-estimation-noise-measurement-in-image
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
