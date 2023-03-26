# Author: Garcia-Cifuentes, K. <ORCID:0009-0001-2607-6359>
# Author: Becerra, R. L. <ORCID:0000-0002-0216-3415>
# Author: De Colle, F. <ORCID:0000-0002-3137-4633>
# License: GNU General Public License version 2 (1991)

# This file contains the functions to manipulate and visualize Swift/BAT data
# Details about Swift Data can be found in https://swift.gsfc.nasa.gov/about_swift/bat_desc.html

import os

import matplotlib.axes
import tables
import warnings
import requests
import numpy as np
import pandas as pd
import concurrent.futures
from . import _tools
from tqdm import tqdm
from typing import Union
from itertools import repeat
import matplotlib.pyplot as plt
from scipy.fft import next_fast_len
from tables import NaturalNameWarning
from collections.abc import Sequence, Mapping
warnings.filterwarnings('ignore', category=NaturalNameWarning)


class SWIFT:
    """SWIFT

    SWIFT is a tool to download, analyze and visualize data from SWIFT/BAT. There is a lot of options of binning modes
    (covering from 2ms to 10s), visualization options (noise-reduction, band-by-band, concatenated, etc), and
    advanced tools (i.e., perform visualizations in real-time using animations). SWIFT has a wide variety of functions
    and path variables, where the user can customize step-by-step where to save data, results of processing these data
    and save visualizations (figures and animations).

    Current implementation of SWIFT works in a parallel way, where the user can customize the number of threads of CPU
    being used in any moment. Moreover, it offers some features as write and read in compressed files, allowing to
    optimize disk use without loosing speed of computing.

    Read more in the docs section available in GitHub: https://github.com/KenethGarcia/ClassiPyGRB

    Attributes:
        workers (int): Number of CPU threads to use within the SWIFT Class. Defaults to max threads available.
        res (int): Binning resolution used to download/manipulate SWIFT data, expressed in ms. Defaults to 64.
    """

    workers = os.cpu_count()
    column_labels = ['Time(s)', '15-25keV', '15-25Err', '25-50keV', '25-50Err', '50-100keV', '50-100Err', '100-350keV',
                     '100-350Err', '15-350keV', '15-350Err']

    def __init__(
            self,
            root_path: str,
            res: int = 64,
            n_bands: Union[Sequence, np.ndarray] = (1, 2, 3, 4),
            data_path: str = None,
            original_data_path: str = None,
            noise_data_path: str = None,
            results_path: str = None,
            noise_images_path: str = None,
            table_path: str = None,
    ):
        """
        Constructor of SWIFT Class

        Args:
            root_path (str): Main path to save data/results from SWIFT.
                Unique mandatory path to ensure the functionality of SWIFT Class.
            res (int): Binning resolution used to download/manipulate SWIFT data, expressed in ms. Defaults to 64.
                Current supported resolutions are 2 (2 ms), 8 (8 ms), 16 (16 ms), 64 (64 ms), 256 (256 ms), 1000 (1 s),
                and 10000 (10 s).
            n_bands (array-like): An array of bands to be considered, ranging from 1 to 5. Defaults to (1, 2, 3, 4, 5).
                Probably the most important parameter for SWIFT. It sets the bands from SWIFT/BAT to be used in each
                process involving data manipulating (except for download data). In order, 1 represent the 15-25 keV
                band, 2 represent the 25-50 keV band, 3 represent the 50-100 keV band, 4 represent the 100-350 keV band,
                and 5 represent the 15-350 keV band.
            data_path (str, optional): Path to save data from SWIFT. Defaults to Data folder inside root_path.
            original_data_path (str, optional): Path to save non-manipulated data from SWIFT. Defaults to Original_Data
                folder inside data_path.
            noise_data_path (str, optional): Path to save noise-reduced data from SWIFT. Defaults to Noise_Filtered_Data
                folder inside data_path.
            results_path (str, optional): Path to save non-manipulated data from SWIFT. Defaults to Results folder
                inside root_path.
            noise_images_path (str, optional): Path to save noise-reduced plots from SWIFT. Defaults to
                Noise_Filter_Images folder inside results_path.
            table_path (str, optional): Path to save tables from SWIFT. Defaults to Tables folder inside root_path.

        Raises:
            ValueError: If res is not in (2, 8, 16, 64, 256, 1000, 10000)

        Examples:
            >>> SWIFT(root_path='/home', res=2, n_bands=(1, 3))  # SWIFT object of 15-25 and 50-100 keV bands at 2 ms
            >>> SWIFT('/home/Documents', 10000, [5])  # SWIFT object of 15-350 keV band at 10 s binning
            >>> SWIFT('/home')  # SWIFT object of 15-25, 25-50, 50-100, 100-350, and 15-350 keV bands at 64 ms binning
        """
        if res not in (2, 8, 16, 64, 256, 1000, 10000):
            raise ValueError(f"{res} not supported. Current supported resolutions (res) are 2 (2 ms), 8 (8 ms), "
                             f"16 (16 ms), 64 (64 ms), 256 (256 ms), 1000 (1 s), and 10000 (10 s).")
        self.res = res
        self.end = f"1s" if self.res == 1000 else f"sn5_10s" if self.res == 10000 else f"{self.res}ms"

        if any(elem not in (1, 2, 3, 4, 5) for elem in n_bands):
            raise ValueError(f"{n_bands} not supported. n_bands can be only an array with integer elements ranging from"
                             f" 1 to 5.")
        self.n_bands = n_bands

        self.root_path = os.path.join(os.getcwd()) if root_path is None else root_path
        self.table_path = os.path.join(self.root_path, r"Tables") if table_path is None else table_path
        self.data_path = os.path.join(self.root_path, r"Data") if data_path is None else data_path
        self.original_data_path = os.path.join(self.data_path, r"Original_Data") if original_data_path is None \
            else original_data_path
        self.noise_data_path = os.path.join(self.data_path, r"Noise_Filtered_Data") if noise_data_path is None \
            else noise_data_path
        self.results_path = os.path.join(self.root_path, r"Results") if results_path is None else results_path
        self.noise_images_path = os.path.join(self.results_path, r"Noise_Filter_Images") if noise_images_path is None \
            else noise_images_path
        self.bands_selected = [self.column_labels[2 * i - 1] for i in n_bands]
        self.bands_selected.insert(0, self.column_labels[0])

    @staticmethod
    def tables_update():
        """ Function to update or download summary tables needed from Swift/BAT.

        Returns:
            dict: Boolean array saving which tables was downloaded.
        """
        checks = {}
        main_url = 'https://swift.gsfc.nasa.gov/results/batgrbcat/summary_cflux/'
        urls = {'summary_general.txt': 'summary_general_info', 'summary_burst_durations.txt': 'summary_general_info',
                'GRBlist_redshift_BAT.txt': 'summary_general_info'}
        for key, value in urls.items():
            try:
                r = requests.get(f"{main_url}{value}/{key}", timeout=5)
                r.raise_for_status()
            except requests.exceptions.RequestException as err:
                warnings.warn_explicit(f"Table {key} can not be downloaded --> {err}", RuntimeWarning,
                                       filename='_swift.py', lineno=126)
                checks[key] = 'Failure'
            else:
                with open(os.path.join('../tables', key), 'wb') as f:
                    f.write(r.content)
                checks[key] = 'Success'
        return checks

    @staticmethod
    def summary_table():
        """Query a Dataframe with Summary Table from Swift.

        Returns:
            A pandas Dataframe containing the summary info from Swift.
        """
        columns = ('GRBname', 'Trig_ID', 'Trig_time_met', 'Trig_time_UTC', 'RA_ground', 'DEC_ground',
                   'Image_position_err', 'Image_SNR', 'T90', 'T90_err', 'T50', 'T50_err', 'Evt_start_sincetrig',
                   'Evt_stop_sincetrig', 'pcode', 'Trigger_method', 'XRT_detection', 'comment')
        table = np.genfromtxt('../tables/summary_general.txt', delimiter="|", dtype=str, unpack=True, autostrip=True)
        df = pd.DataFrame()
        for i in range(len(table)):
            df[columns[i]] = table[i]
        return df

    def obtain_data(
            self,
            name: str,
            check_disk: bool = False,
    ):
        """Obtain Swift/BAT data for any GRB Name.

        Args:
            name (str): GRB name in format 'GRBXXXXXXX'.
            check_disk (bool): Flag indicating whether to check the disk for previously downloaded data.

        Returns:
            Dataframe with GRB data if successful. Otherwise, a string representing the exception description.

        Raises:
            RuntimeError: If there are any error concerning to table reading.
            ValueError: If name is not a string character.

        Examples:
            >>> SWIFT.obtain_data(name='GRB060614')
            >>> SWIFT.obtain_data(name='GRB220715B')
        """
        if check_disk:
            file_path = os.path.join(self.original_data_path, f"{name}_{self.end}.h5")
            try:  # Search in disk if it was downloaded earlier
                df = pd.read_hdf(file_path, key=name)
                return df
            except FileNotFoundError:  # Otherwise, search into Swift website
                warnings.warn(f"No such file: {file_path} -> Trying to query from Swift Website...", UserWarning)
                return self.obtain_data(name=name, check_disk=False)
        else:
            grb_names, ids = np.genfromtxt('../tables/summary_general.txt', delimiter="|", dtype=str, usecols=(0, 1),
                                           unpack=True, autostrip=True)
            if len(grb_names) == 0 or not isinstance(grb_names, (Sequence, Mapping, np.ndarray)):
                raise TypeError(f"Error when reading Table: Expected array-like of grb_names, obtained "
                                f"{type(grb_names)}. Try to download tables again using tables_update method...")
            if not isinstance(name, str):
                raise ValueError(f"Expected GRB name of class 'str'. Got {type(name)}.")
            index = np.where(grb_names == name)
            t_id, *other = ids[index]
            i_d = f"0{t_id}000" if len(t_id) == 7 else f"00{t_id}000" if len(t_id) == 6 else t_id
            root_url = 'https://swift.gsfc.nasa.gov/results/batgrbcat/'
            url = f"{root_url}{name}/data_product/{i_d}-results/lc/{self.end}_lc_ascii.dat"
            try:
                r = requests.get(url, timeout=5)
                r.raise_for_status()
            except requests.exceptions.RequestException as err:
                return err
            else:
                df = pd.read_csv(url, delimiter=r'\s+', header=None, names=self.column_labels)
                return df

    def single_download(
            self,
            name: str
    ):
        """Download Swift/BAT data for a specific GRB with ID identifier.

        Args:
            name (str): GRB name in format 'GRBXXXXXXX'.

        Returns:
            None if the data download is successful, otherwise a string representing the RequestException description.

        Examples:
            >>> SWIFT.single_download(name='GRB060614', t_id=1069788)
            None
            >>> SWIFT.single_download(name='GRB220715B', t_id='1116441')
            None
        """
        _tools.directory_maker(self.original_data_path)
        file_name = f"{name}_{self.end}.h5"
        path = os.path.join(self.original_data_path, file_name)
        try:
            os.remove(path) if os.path.exists(path) else None
            df = self.obtain_data(name=name)
            if not isinstance(df, pd.DataFrame):
                return df
            df.to_hdf(path_or_buf=path, key=name, complevel=0, format='fixed')
        except tables.exceptions.HDF5ExtError as e:
            return e
        else:
            return None

    def multiple_downloads(
            self,
            names: Union[list, tuple, np.ndarray],
            error: bool = True
    ):
        """Downloads Swift/BAT data for a list of GRBs using multiple threads.

        Args:
            names (list of str): List of GRB names to download.
            error (bool, optional): Flag to write error log. Defaults to True.

        Returns:
            None

        Examples:
            >>> SWIFT.multiple_downloads(names=['GRB060614', 'GRB220715B'], t_ids=[1069788, 1116441])
            None
            >>> SWIFT.multiple_downloads(names=('GRB220715B', 'GRB060614'), t_ids=('1116441', '1069788'))
            None
        """
        # Try to create the folder in the associated path, unless it already has been created.
        # It mitigates a FileNotFoundError with open(path) when the directory is created while Threading is executing.
        _tools.directory_maker(self.original_data_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = list(tqdm(executor.map(self.single_download, names), total=len(names), desc='Downloading: ',
                                unit='GRB'))
            if error:
                with open(os.path.join(self.original_data_path, f"Errors_{self.end}.txt"), 'w') as f:
                    f.write("## GRB_Name\tTrigger_ID\tError_Description\n")
                    for i in range(len(results)):
                        f.write(f"{names[i]}\t{results[i]}\n") if results[i] is not None else None

    def duration_limits(
            self,
            name: Union[str, list, tuple, np.ndarray] = None,
            t: int = 100
    ):
        """Function to extract GRB duration intervals.

        Args:
            name (str, list of str): GRB Name or list of GRB Names. Defaults to None.
                If None, then all GRBs available will be indexed.
            t (int):  Duration interval. Defaults to 100.
                Duration is defined as the time interval during which t% of the total observed counts have been
                detected. Supported values are 50, 90, and 100.

        Returns:
            list: Array in format [[Name_i, T_initial, T_final], ...] for each i-esim GRB name.

        Raises:
            ValueError: If t is not in (50, 90, 100).
            ValueError: If name is not a str or array-like.
            FileNotFoundError: If there are not any table with durations_table name in table path of SWIFT.

        Examples:
            >>>SWIFT.duration_limits(name='GRB061210')
            [['GRB061210' '-0.004' '89.392']]
            >>>SWIFT.duration_limits(name='GRB061210', t=50)
            [['GRB061210' '6.576' '56.004']]
            >>>SWIFT.duration_limits(name=('GRB061210', 'GRB060614'), t=50)
            [[['GRB061210' '6.576' '56.004']], [['GRB060614' '21.116' '64.352']]]
        """
        if t not in (50, 90, 100):
            raise ValueError(f"Duration {t} not supported. Current supported durations (t) are 50, 90, and 100.")
        columns = {50: (0, 7, 8), 90: (0, 5, 6), 100: (0, 3, 4)}  # Dictionary to extract the correct columns
        keys_extract = np.genfromtxt(f'../tables/summary_burst_durations.txt', delimiter="|", dtype=str,
                                     usecols=columns.get(t), autostrip=True)
        # Extract all values from summary_burst_durations.txt:
        if isinstance(name, str):  # If a specific name is entered, then we search the value in the array
            return _tools.check_name(name, keys_extract)
        elif isinstance(name, (Sequence, Mapping, np.ndarray)):  # If a name's array is specified, search recursively
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
                results = list(
                    tqdm(executor.map(_tools.check_name, name, np.array([keys_extract] * len(name))), total=len(name),
                         desc='Finding Durations: ', unit='GRB'))
                return np.array(results)
        elif name is None:  # Else return data for all GRBs
            return keys_extract
        else:
            raise ValueError(f"{type(name)} not supported. Try changing to a str, list or array-like.")

    def total_durations(
            self,
            names: Union[list, tuple, np.ndarray],
            t: int = 100
    ):
        """Total Duration Calculator.

        Args:
            names (list of str): List of GRB Names. Defaults to None.
                If None, then all GRBs available will be indexed.
            t (int): Duration interval. Defaults to 100.
                Duration is defined as the time interval during which t% of the total observed counts have been
                detected. Supported values are 50, 90, and 100.

        Returns:
            Array with [T_final - T_initial, ...] for each i-esim GRB name.

        Raises:
            ValueError: If t is not in (50, 90, 100).
            ValueError: If names is not a list or array-like.

        Examples:
            >>>SWIFT.total_durations(name=['GRB060614'])
            180.576
            >>>SWIFT.total_durations(name=['GRB061210'])
            89.396
            >>>SWIFT.total_durations(name=('GRB061210', 'GRB060614'), t=50)
            [49.428 43.236]
        """
        if t not in (50, 90, 100):
            raise ValueError(f"Duration {t} not supported. Current supported durations (t) are 50, 90, and 100.")
        if not isinstance(names, (Sequence, Mapping, np.ndarray)):
            raise ValueError(f"GRB names needs to be an array-like (i.e., list, tuple). Received a {type(names)}.")
        durations_array = self.duration_limits(names, t=t)  # Check for name, t_start, and t_end
        start_t, end_t = durations_array[:, :, 1].astype(float), durations_array[:, :, 2].astype(float)
        duration = np.reshape(end_t - start_t, len(durations_array))  # T_90 is equal to t_end - t_start
        return duration

    def redshifts(
            self,
            name: Union[str, Sequence, Mapping, np.ndarray, pd.Series] = None,
            redshift_table: str = "GRBlist_redshift_BAT.txt"
    ):
        """GRB Redshift extractor.

        Args:
            name (str, list of str): GRB Name or list of GRB Names. Defaults to None.
                If None, then all GRBs available will be indexed.
            redshift_table (str, optional): Durations table name, related with table path class variable. Defaults to
                'summary_burst_durations.txt'.

        Returns:
            list: Array in format [[Name_i, Z_i], ...] for each i-esim GRB name.

        Raises:
            FileNotFoundError: If there are not any table with redshift_table name in table path of SWIFT.

        Examples:
            >>>SWIFT.redshifts()
            [['GRB220611A' '2.3608'], ['GRB220521A' '5.6'], ['GRB220117A' '4.961', ['GRB220101A' '4.61'], ...]
            >>>SWIFT.redshifts(name='GRB181020A')
            [['GRB181020A' '2.938']]
            >>>SWIFT.redshifts(name=['GRB220611A', 'GRB220521A'])
            [['GRB220611A' '2.3608'], ['GRB220521A' '5.6']]
        """
        path = os.path.join(self.table_path, redshift_table)
        keys_extract = np.genfromtxt(path, delimiter="|", dtype=str, usecols=(0, 1), autostrip=True)
        # Extract all values from summary_burst_durations.txt:
        if isinstance(name, str):  # If a specific name is entered, then we search the redshift in the array
            return _tools.check_name(name, keys_extract)
        elif isinstance(name, (np.ndarray, list, tuple)):  # If a name's array is specified, search them recursively
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
                results = list(
                    tqdm(executor.map(_tools.check_name, name, np.array([keys_extract] * len(name))), total=len(name),
                         desc='Finding Redshifts: ', unit='GRB'))
                returns = []
                for i in range(len(name)):
                    returns.append(results[i][0]) if len(results[i]) != 0 else returns.append([name[i], None])
                return np.array(returns)
        else:  # Else return data for all GRBs
            return keys_extract

    def lc_limiter(
            self,
            name: str,
            t: int = 100,
            limits: Union[Sequence, Mapping, np.ndarray, pd.DataFrame] = None
    ):
        """Limit out of any Swift/BAT GRB light curve by condition.

        Args:
            name (str): GRB name in format 'GRBXXXXXXX'.
            t (int): Duration interval. Defaults to 100.
                Duration is defined as the time interval during which t% of the total observed counts have been
                detected. Supported values are 50, 90, and 100.
            limits (list, optional): Custom limits [t_start, t_end] to limit. Defaults to None.

        Returns:
            List with [name, '', '', Error Desc] if data does not exist, there are no valid limits or there are less
            than 4 points after limiting. Otherwise, array with values between selected limits.

        Raises:
            ValueError: If t is not in (50, 90, 100).
            ValueError: If name is not a str.

        Examples:
            >>>SWIFT.lc_limiter(name='GRB060614')  # Limit GRB060614 data between T_100
            >>>SWIFT.lc_limiter(name='GRB060510A')
            ('GRB060510A', -6.752, 16.748, 'Only zeros')
            >>>SWIFT.lc_limiter(name='GRB060614', intervals=(-2, 100))  # Limit GRB060614 data between -2 to 100 s
            >>>SWIFT.lc_limiter(name='GRB050925', t=100)
            ('GRB050925', -0.036, 0.068, 'Length=2')
        """
        if t not in (50, 90, 100) and limits is None:
            raise ValueError(f"Duration {t} not supported. Current supported durations (t) are 50, 90, and 100.")
        if not isinstance(name, str):
            raise ValueError(f"{type(name)} not supported as name. Only str GRB name supported in current version.")
        try:
            if limits is None:
                intervals = self.duration_limits(name=name, t=t)
                t_start, t_end = float(intervals[0, 1]), float(intervals[0, 2])
            else:
                t_start, t_end, *other = limits
                t_start, t_end = float(t_start), float(t_end)
            df = self.obtain_data(name=name, check_disk=True)
            if not isinstance(df, pd.DataFrame):
                raise FileNotFoundError
            df = df[(df[self.column_labels[0]] < t_end) & (df[self.column_labels[0]] > t_start)]
            df = df[self.bands_selected]
        except FileNotFoundError:  # If file is not found, return error
            return name, ' ', ' ', 'FileNotFoundError'
        except ValueError:  # If there aren't any valid T_start or T_end, return error
            return name, ' ', ' ', 'ValueError'
        except IndexError:  # If there aren't row in burst durations, return
            return name, ' ', ' ', 'IndexError'
        else:
            if len(df[self.column_labels[0]]) < 3:
                return name, t_start, t_end, f'Length={len(df[self.column_labels[0]])}'
            elif df[self.bands_selected[1:]].eq(0).all().all():
                return name, t_start, t_end, 'Only zeros'
            else:  # Check if there are more than 2 points in light curve
                return df

    def parallel_lc_limiter(
            self,
            names: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            t: int = 100,
            limits: Union[Sequence, Mapping, np.ndarray, pd.DataFrame] = None,
    ):
        """Parallel version of lc_limiter function. It limits any Swift/BAT light curve based on condition.

        Args:
            names (list of str): List of GRB names in format 'GRBXXXXXXX'.
            t (int): Duration interval. Defaults to 100.
                Duration is defined as the time interval during which t% of the total observed counts have been
                detected. Supported values are 50, 90, and 100.
            limits (list, optional): Array with custom limits [t_start, t_end] to each GRB. Defaults to None.
                It must have the same size of names array.

        Returns:
            A tuple of 3 elements: A list of Dataframes containing the limited light curves, a list of GRB names for
            each Dataframe, and a Dataframe with the Errors log while limiting.

        Raises:
            ValueError: If t is not in (50, 90, 100).
            ValueError: If name is not a Sequence, Mapping or array-like.
            AttributeError: If len(limits) is not equal to len(names).
        """
        errors = pd.DataFrame(columns=['Names', 't_start', 't_end', 'Error'])
        non_errors, new_names = [], []
        if t not in (50, 90, 100):
            raise ValueError(f"Duration {t} not supported. Current supported durations (t) are 50, 90, and 100.")
        if not isinstance(names, (Sequence, Mapping, np.ndarray, pd.Series)):
            raise ValueError(f"GRB names needs to be an array-like (i.e., list, tuple). Received a {type(names)}.")
        if limits is not None and (len(limits) != len(names)):
            raise AttributeError(f"You must have the same length of names and limit intervals. Received {len(names)} "
                                 f"GRBs and {len(limits)} limit intervals.")

        if limits is None:  # If no limits are entered, then only send None array
            limits = repeat(None, len(names))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = list(tqdm(executor.map(self.lc_limiter, names, repeat(t, len(names)), limits), total=len(names),
                                desc='LC limiting: ', unit='GRB'))
        for name, array in zip(names, results):
            if isinstance(array, tuple):
                new_row = pd.DataFrame([array], columns=['Names', 't_start', 't_end', 'Error'])
                errors = pd.concat([errors, new_row])
            else:
                non_errors.append(array)
                new_names.append(name)
        return non_errors, new_names, errors

    def plot_any_grb(
            self,
            name: str,
            t: int = None,
            limits: Union[Sequence, Mapping, np.ndarray, pd.DataFrame] = None,
            kind: str = 'Default',
            ax=None,
            legend: bool = True,
            **fig_kwargs
    ):
        """Function to plot any GRB.

        Args:
            name (str): GRB name in format 'GRBXXXXXXX'.
            t (int, optional): Duration interval. Defaults to None.
                Duration is defined as the time interval during which t% of the total observed counts have been
                detected. Supported values are 50, 90, 100, and None (the total light curve will be plotted).
            limits (list, optional): Custom limits [t_start, t_end] to limit. Defaults to None.
            kind (str, optional): Type of plot. Defaults to 'Default'.
                Current supported kind of plot: 'Interpolated' (Change plot to background-gray scaled), 'Concatenated'
                (plot all bands in one single panel), 'Default' (plots each band in one panel).
            ax (matplotlib.axes.Axes, optional): The Axes object to plot the light curve onto. Defaults to None.
                In case of 'Default' or 'Interpolated' kind, ax must have the same number of columns as bands selected.
            legend (bool, optional): Flag to put band legend in plot. Defaults to True.
            **fig_kwargs: Additional arguments to be passed in matplotlib.pyplot.subplots if kind is 'Concatenated'.
                Otherwise, arguments to be passed in matplotlib.pyplot.figure.

        Returns:
            The matplotlib.pyplot.axes object of the plot.

        Raises:
            ValueError: If t is not in (50, 90, 100, None).
            ValueError: If name is not a str.
            ValueError: If kind is not in ('Default', 'Interpolated', 'Concatenated').
        """
        if t not in (50, 90, 100, None):
            raise ValueError(f"Duration {t} not supported. Current supported durations (t) are 50, 90, 100, and None.")
        if kind.lower() not in ('default', 'interpolated', 'concatenated') or not isinstance(kind, str):
            raise ValueError(f"Kind {kind} not supported. Current supported kinds are 'Default', 'Interpolated', "
                             f"and 'Concatenated'.")
        if not isinstance(name, str):
            raise ValueError(f"{type(name)} not supported as name. Only str GRB name supported in current version.")
        aux_inter = True
        if len(self.bands_selected)-1 == 1:
            aux_inter = False if kind.lower() == 'interpolated' else True
            kind = 'Concatenated'
        if ax is None:  # If there aren't any previous axes, create new one
            if kind.lower() == "concatenated":
                fig5, ax = plt.subplots(dpi=150, **fig_kwargs)
                ax.set_ylabel(r"Counts/sec/det", weight='bold').set_fontsize('10')
            else:
                fig5 = plt.figure(dpi=150, **fig_kwargs)
                gs = fig5.add_gridspec(nrows=len(self.bands_selected)-1, hspace=0)
                ax = gs.subplots(sharex=True)
                ax[(len(self.bands_selected)-1)//2].set_ylabel(r"Counts/sec/det", weight='bold').set_fontsize('10')
        low_sub = ax if kind.lower() == "concatenated" else ax[-1]
        high_sub = ax if kind.lower() == "concatenated" else ax[0]
        low_sub.set_xlabel('Time since BAT Trigger time (s)', weight='bold').set_fontsize('10')
        if t in (50, 90, 100) or limits is not None:
            df = self.lc_limiter(name=name, t=t, limits=limits)
            if isinstance(df, tuple):
                warnings.warn(fr"Error when limiting out of T_{t} {name}: {df[3]}. Plotting all data...",
                              RuntimeWarning)
                df = self.obtain_data(name=name)
                df = df[self.bands_selected]
                high_sub.set_title(f"{self.end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        else:
            df = self.obtain_data(name=name)
            df = df[self.bands_selected]
            high_sub.set_title(f"{self.end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        colors = ['#2d0957', '#12526f', '#04796a', '#7ab721', '#d9c40a']
        columns = list(df.columns)
        columns.pop(0)
        for i in range(len(self.bands_selected)-1):
            x_val = df[self.column_labels[0]]
            if kind.lower() == "interpolated":
                ax[i].plot(x_val, df[columns[i]], label=columns[i], alpha=0.3, ms=0.5, c='gray')
            elif kind.lower() == "concatenated":
                ax.plot(x_val, df[columns[i]], label=columns[i], linewidth=0.5, c=colors[i] if aux_inter else 'gray',
                        alpha=1 if aux_inter else 0.3) if i < 4 else None
            else:
                ax[i].plot(x_val, df[columns[i]], label=columns[i], linewidth=0.5, c=colors[i])
            if legend:
                ax[i].legend(fontsize='xx-small', loc="upper right") if kind.lower() != "concatenated" else None
                ax.legend(fontsize='xx-small', loc="upper right") if (
                            kind.lower() == "concatenated" and i == len(columns)-1) else None
        return ax

    @staticmethod
    def lc_normalize(
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            base: int = -1
    ):
        """Normalize light curves over any band from Swift/BAT. Data is assumed to have time axis at first column.

        This function divides all columns of the array by the desired column integral, except for the first column.

        Args:
            data (array-like): Pandas Dataframe or Array-like with data to be normalized.
                The format must follow [[t_0, x_0, y_0, z_0, ...], [t_1, x_1, y_1, z_1, ...],
                [t_2, x_2, y_2, z_2, ...], ...] with t_i the time elements, and x_i, y_i, ... the i-esim bands.

            base (int): Reference column to standardize. Defaults to last column in array.

        Returns:
            Array with columns normalized, except for the time column. If the input data array is a pandas Dataframe,
            it returns a new normalized Pandas Dataframe.
        """
        if not isinstance(base, int):
            raise ValueError(f"Column to index needs to be a integer. Received a {type(base)}.")
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        arr = np.asarray(data)
        total_flux = np.trapz(y=arr[:, base], x=arr[:, 0])
        arr[:, 1:] /= total_flux
        if isinstance(data, pd.DataFrame):
            return pd.DataFrame(arr, columns=data.columns)
        else:
            return arr

    def parallel_lc_normalize(
            self,
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            base: Union[int, Sequence, Mapping, np.ndarray, pd.DataFrame] = -1
    ):
        """Parallel version of lc_normalize function. It normalizes any Swift/BAT data based on a reference column.

        Args:
            data (list of arrays): Array containing all individual Swift/BAT data.
                Each array must follow the format [[t_0, x_0, y_0, z_0, ...], [t_1, x_1, y_1, z_1, ...],
                [t_2, x_2, y_2, z_2, ...], ...] with t_i the time elements, and x_i, y_i, ... the i-esim bands.
            base (int or list of int): Array or integer of column to standardize. Defaults to last column in array.

        Returns:
            Array with normalized light curves. If the input array elements are Pandas Dataframe, the returned array
            will contain normalized Pandas Dataframes.

        Raises:
            ValueError: If base is not an array-like or integer.
            ValueError: If data is not an array-like.
            ValueError: If base is an array and its length is not equal to the data length.
        """
        if not isinstance(base, (Sequence, Mapping, np.ndarray, pd.Series, int)):
            raise ValueError(f"Column to index needs to be a integer or list of integers. Received a {type(base)}.")
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.Series)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        if isinstance(base, (Sequence, Mapping, np.ndarray, pd.Series)) and len(base) != len(data):
            raise ValueError(f"Length of base needs to match len of data array. Received {len(base)} elements in base"
                             f" and {len(data)} elements in data.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
            results = list(tqdm(executor.map(self.lc_normalize, data), total=len(data), desc='LC Normalizing: ',
                                unit='GRB'))
        return results

    def zero_pad(
            self,
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            length: int
    ):
        """Function to zero pad any array-like.

        Args:
            data (array-like): Pandas Dataframe or Array-like with data to be zero-padded.
                The format must follow [[x_0, y_0, z_0, ...], [x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...]
                with x_i, y_i, ... the i-esim bands. If data is a Pandas Dataframe, the time column will be deleted.
            length (int): Length of the final array.

        Returns:
            Array with zero-padded light curves. If the input array is a Pandas Dataframe, the returned array
            will contain zero padded Pandas Dataframe without time column.

        Raises:
            ValueError: If data is not an array-like.
            ValueError: If length is not an integer or its value is less or equal than data length.
        """
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        if not isinstance(length, int):
            raise ValueError(f"Length needs to be an integer. Received a {type(data)}.")
        if len(data) >= length:
            raise ValueError(f"Cannot zero pad to {length} columns if the input array has {len(data)}. Check your input"
                             f" and try again.")
        diff = length - len(data)  # Difference between actual and optimal array size
        bool_data = isinstance(data, pd.DataFrame)
        if bool_data:  # Remove Time column if the input array is a Pandas Dataframe
            data = data.drop(columns=[self.column_labels[0]], inplace=False)
            input_columns = data.columns
        data = np.asarray(data)
        data_plus_zeros = np.pad(data, ((0, diff), (0, 0)))  # Zero pad array
        if bool_data:
            return pd.DataFrame(data_plus_zeros, columns=input_columns)
        else:
            return data_plus_zeros

    def parallel_zero_pad(
            self,
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            length: int = None):
        """Zero-pad GRB light curves in a parallel way.

        Args:
            data (array-like): List of Pandas Dataframe or Array-like with data to be zero-padded.
                Each array must follow the form [[x_0, y_0, z_0, ...], [x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...]
                with x_i, y_i, ... the i-esim bands. If data is a Pandas Dataframe, the time column will be deleted.
            length (int, optional): Length of the final array. Defaults to the best suitable length to perform DFT.

        Returns:
            Array with zero-padded light curves. If the input array is a Pandas Dataframe, the returned array
            will contain zero padded Pandas Dataframe without time columns.
        """
        if not isinstance(length, int) and length is not None:
            raise ValueError(f"Length of zero-pad must be an integer or None. Received a {type(length)}.")
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            if length is None:
                unit_lengths = set(executor.map(len, data))
                length = next_fast_len(max(unit_lengths))
            results = list(tqdm(executor.map(self.zero_pad, data, repeat(length, len(data))), total=len(data),
                                desc='LC Zero-Padding: ', unit='GRB'))
            return results

    @staticmethod
    def concatenate(
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame]
    ):
        """Function to concatenate light curve energy bands in ascendant mode, only works on Swift data format.

        Args:
            data (array-like): Pandas Dataframe or Array-like with data to be zero-padded.
                The format must follow [[x_0, y_0, z_0, ...], [x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...]
                with x_i, y_i, ... the i-esim bands or time-column.

        Returns:
            Array concatenated by column in ascendant mode.

        Raises:
            ValueError: If data is not an array-like.

        """
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        if isinstance(data, pd.DataFrame):
            return data.values.flatten('F')
        else:
            array = np.array(data)
            return array.reshape(-1, order='F')

    def parallel_concatenate(
            self,
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame]
    ):
        """Concatenate light curves in a parallel way.

        Args:
            data (array-like): List of Pandas Dataframe or Array-like with data to be zero-padded.
                Each array must follow the form [[x_0, y_0, z_0, ...], [x_1, y_1, z_1, ...], [x_2, y_2, z_2, ...], ...]
                with x_i, y_i, ... the i-esim bands. If data is a Pandas Dataframe, the time column will be deleted.

        Returns:
            Array with all concatenated arrays. The order is preserved.

        Raises:
            ValueError: If data is not an array-like.
        """
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")

        m = len(data)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
            cc_results = list(tqdm(executor.map(self.concatenate, data), total=m, desc='Concatenating: ', unit='GRB'))
        return np.array(cc_results)

    @staticmethod
    def dft_spectrum(
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
    ):
        """Perform Discrete Fourier Transform (DFT) to any data and obtain its Spectrum.

        Args:
            data (array-like): Pandas Dataframe or Array-like with signal data.
                It is assumed to be a 1D array of n elements.

        Returns:
            Array concatenated by column in ascendant mode.

        Raises:
            ValueError: If data is not an array-like.
        """
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        n = len(data)
        x = np.fft.fft(data)
        mag_x = np.abs(x)/n  # Normalize amplitude
        return mag_x[:len(mag_x)//2]

    def parallel_dft_spectrum(
            self,
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame]
    ):
        """Function to obtain Fourier Spectrum for a list of signals in a parallel way.

        Args:
            data (array-like): List of Pandas Series or Array-like with signals.

        Returns:
            Array with all Fourier spectrum for each signal. The order is preserved.
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            sp = list(tqdm(executor.map(self.dft_spectrum, data), total=len(data), unit='Performing DFT',
                           desc='Performing DFT: '))
        return np.array(sp)

    def dft_plot(
            self,
            spectrum: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            ax: matplotlib.axes.Axes = None,
            name: str = None):
        """Plot the discrete Fourier transform (DFT) amplitude spectrum of a given signal.

        It is assumed that the input signal starts at t = 0 s and has a constant sampling rate of SWIFT resolution.

            Args:
                spectrum (array-like): An array-like object representing the DFT of a signal.
                ax (optional): Sequence of matplotlib axes objects to plot the spectrum on. If None, the function
                    creates a new figure with two subplots.
                name (str, optional): A string representing the name of the signal being plotted. Defaults to None.

            Returns:
                ax: A matplotlib axes object or a sequence of matplotlib axes objects representing the plot.

            Raises:
                ValueError: If the length of `ax` is less than 2 and `ax` is not None, or if `spectrum` is not an
                    array-like object.
        """
        if ax is not None:
            if len(ax) < 2:
                raise ValueError(f"matplotlib.pyplot.axes needs to have at least two elements. "
                                 f"Received {len(ax)} element.")
        if not isinstance(spectrum, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Spectrum needs to be an array-like (i.e., list, tuple). Received a {type(spectrum)}.")
        if ax is None:
            fig, ax = plt.subplots(2, 1, dpi=150, figsize=[10, 7], gridspec_kw={'height_ratios': [0.7, 0.4]})
        freq = np.fft.fftfreq(2*spectrum.size, d=self.res * 1e-3)
        freq = freq[:len(freq) // 2]
        ax[0].set_title(fr"{name} DFT", weight='bold').set_fontsize('12') if name is not None else None
        ax[0].plot(freq[:len(freq) // 5], spectrum[:len(freq) // 5], linewidth=0.5, c='k')
        ax[1].plot(freq[len(freq) // 5:], spectrum[len(freq) // 5:], linewidth=0.5, c='k')
        ax[1].set_xlabel('Frequency (Hz)', weight='bold').set_fontsize('9')
        ax[1].set_ylabel('Amplitude', weight='bold').set_fontsize('9')
        ax[0].set_ylabel('Amplitude', weight='bold').set_fontsize('9')
        return ax

    def save_results(
            self,
            file_name: str,
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            names: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            **kwargs):
        """Saves the given data and names to a npz file in the results path of the SWIFT class instance.

            Args:
                file_name (str): Name of the npz file to be created.
                names (List[str]): List of names of the columns/variables in the data array.
                data (np.ndarray): Numpy array containing the data to be saved.
                **kwargs: Additional keyword arguments to be passed to np.savez function.

            Returns:
                None
        """
        if not isinstance(data, (Sequence, Mapping, np.ndarray)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        if not isinstance(names, (Sequence, Mapping, np.ndarray)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        _tools.directory_maker(self.results_path)
        np.savez(os.path.join(self.results_path, file_name), names=names, data=data, **kwargs)
