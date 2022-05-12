"""A script containing majority of repetitive functions needed in Swift Data manipulation"""
import os  # Import os to handle folders and files
import numpy as np  # Import module to handle arrays


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
    rows, columns_i = np.where(table_array == name_i)
    return table_array[rows]
