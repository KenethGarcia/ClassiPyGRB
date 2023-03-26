# Author: Garcia-Cifuentes, K. <ORCID:0009-0001-2607-6359>
# Author: Becerra, R. L. <ORCID:0000-0002-0216-3415>
# Author: De Colle, F. <ORCID:0000-0002-3137-4633>
# License: GNU General Public License version 2 (1991)

# This file contains second-level functions to manipulate and visualize data from Swift/BAT data
# Details about Swift Data can be found in https://swift.gsfc.nasa.gov/about_swift/bat_desc.html

import os
import numpy as np
from typing import Union


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
