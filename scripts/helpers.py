"""A script containing majority of repetitive functions needed in Swift Data manipulation"""
import os  # Import os to handle folders and files


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
