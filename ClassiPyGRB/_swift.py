# Author: Garcia-Cifuentes, K. <ORCID:0009-0001-2607-6359>
# Author: Becerra, R. L. <ORCID:0000-0002-0216-3415>
# Author: De Colle, F. <ORCID:0000-0002-3137-4633>
# License: GNU General Public License version 2 (1991)

# This file contains the functions to manipulate and visualize Swift/BAT data
# Details about Swift Data can be found in https://swift.gsfc.nasa.gov/about_swift/bat_desc.html

import os
import tables
import warnings
import requests
import numpy as np
import pandas as pd
import matplotlib.axes
import concurrent.futures
import matplotlib.pyplot as plt
from . import _tools
from tqdm import tqdm
from typing import Union
from fabada import fabada
from itertools import repeat
from scipy.fft import next_fast_len
from tables import NaturalNameWarning
from scipy.interpolate import interp1d
from collections.abc import Sequence, Mapping
from openTSNE import TSNE as open_tsne
from sklearn.manifold import TSNE as sklearn_tsne
warnings.filterwarnings('ignore', category=NaturalNameWarning)


class SWIFT:
    """SWIFT

    SWIFT is a tool to download, analyze and visualize data from SWIFT/BAT. There are several options of binning modes
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
            root_path: str = None,
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
            root_path (str): Main path to save data/results from SWIFT. Defaults to None.
                Unique mandatory path to ensure the functionality of saving data/figures in SWIFT Class.
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
            >>> SWIFT.single_download(name='GRB060614')
            None
            >>> SWIFT.single_download(name='GRB220715B')
            None
        """
        _tools.directory_maker(self.original_data_path)
        file_name = f"{name}_{self.end}.h5"
        try:
            df = self.obtain_data(name=name)
            if not isinstance(df, pd.DataFrame):
                return df
            _tools.save_data(data=df, name=name, filename=file_name, directory=self.original_data_path)
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
            >>> SWIFT.multiple_downloads(names=['GRB060614', 'GRB220715B'])
            None
            >>> SWIFT.multiple_downloads(names=('GRB220715B', 'GRB060614'))
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
            name: Union[str, Sequence, Mapping, np.ndarray, pd.Series] = None
    ):
        """GRB Redshift extractor.

        Args:
            name (str, list of str): GRB Name or list of GRB Names. Defaults to None.
                If None, then all GRBs available will be indexed.

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
        keys_extract = np.genfromtxt(f'../tables/GRBlist_redshift_BAT.txt', delimiter="|", dtype=str, usecols=(0, 1), autostrip=True)
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
            >>>SWIFT.lc_limiter(name='GRB060614',limits=(-2, 100))  # Limit GRB060614 data between -2 to 100 s
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
            check_disk: bool = False,
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
            check_disk (bool, optional): Flag to check if light curve is in disk. Defaults to False.
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
        if len(self.bands_selected) - 1 == 1:
            aux_inter = False if kind.lower() == 'interpolated' else True
            kind = 'Concatenated'
        if ax is None:  # If there aren't any previous axes, create new one
            if kind.lower() == "concatenated":
                fig5, ax = plt.subplots(dpi=150, **fig_kwargs)
                ax.set_ylabel(r"Counts/sec/det", weight='bold').set_fontsize('10')
            else:
                fig5 = plt.figure(dpi=150, **fig_kwargs)
                gs = fig5.add_gridspec(nrows=len(self.bands_selected) - 1, hspace=0)
                ax = gs.subplots(sharex=True)
                ax[(len(self.bands_selected) - 1) // 2].set_ylabel(r"Counts/sec/det", weight='bold').set_fontsize('10')
        low_sub = ax if kind.lower() == "concatenated" else ax[-1]
        high_sub = ax if kind.lower() == "concatenated" else ax[0]
        low_sub.set_xlabel('Time since BAT Trigger time (s)', weight='bold').set_fontsize('10')
        if t in (50, 90, 100) or limits is not None:
            df = self.lc_limiter(name=name, t=t, limits=limits)
            if isinstance(df, tuple):
                warnings.warn(fr"Error when limiting out of T_{t} {name}: {df[3]}. Plotting all data...",
                              RuntimeWarning)
                df = self.obtain_data(name=name, check_disk=check_disk)
                df = df[self.bands_selected]
                high_sub.set_title(f"{self.end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        else:
            df = self.obtain_data(name=name, check_disk=check_disk)
            df = df[self.bands_selected]
            high_sub.set_title(f"{self.end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        colors = ['#2d0957', '#12526f', '#04796a', '#7ab721', '#d9c40a']
        columns = list(df.columns)
        columns.pop(0)
        for i in range(len(self.bands_selected) - 1):
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
                        kind.lower() == "concatenated" and i == len(columns) - 1) else None
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
        mag_x = np.abs(x) / n  # Normalize amplitude
        return mag_x[:len(mag_x) // 2]

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
            sp = list(tqdm(executor.map(self.dft_spectrum, data), total=len(data), unit='GRB', desc='Performing DFT: '))
        return np.array(sp)

    def dft_plot(
            self,
            spectrum: Union[Sequence, Mapping, np.ndarray, pd.Series],
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
        if not isinstance(spectrum, (Sequence, Mapping, np.ndarray, pd.Series)):
            raise ValueError(f"Spectrum needs to be an array-like (i.e., list, tuple). Received a {type(spectrum)}.")
        if ax is None:
            fig, ax = plt.subplots(2, 1, dpi=150, figsize=[10, 7], gridspec_kw={'height_ratios': [0.7, 0.4]})
        freq = np.fft.fftfreq(2 * spectrum.size, d=self.res * 1e-3)
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
            names: Union[Sequence, Mapping, np.ndarray, pd.Series],
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
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        if not isinstance(names, (Sequence, Mapping, np.ndarray, pd.Series)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        _tools.directory_maker(self.results_path)
        np.savez(os.path.join(self.results_path, file_name), names=names, data=data, **kwargs)

    @staticmethod
    def grb_interpolate(
            data: Union[Sequence, Mapping, np.ndarray, pd.Series],
            new_time: Union[Sequence, Mapping, np.ndarray, pd.Series] = None,
            res: float = 64,
            kind: str = 'linear',
            pack_num: int = 10,
    ):
        """Function to interpolate one band from any GRB light curve

        Interpolate a light curve in a Pandas DataFrame or an array where the time data is in the first column, and the
        band measurements are in the next columns. If a non-linear interpolation is selected, it can use an additional
        input to indicate how much points per band will be used to interpolate.

        Args:
            data (array-like): Input data array.
                The time data is assumed to be in the first column, and the band measurements are in the next columns.
            new_time (array-like, optional): Output Time Array.
            res (float, optional): Resolution of the data interpolated in milliseconds. Defaults to 64.
            kind (str, optional): Kind of interpolation. Defaults to 'linear'.
                From Scipy Docs: Specifies the kind of interpolation as a string or as an integer specifying the order
                of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’
                , ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
            pack_num (int, optional): Number of data grouped per packet to interpolate. Defaults to 10.

        Returns:
            Pandas Dataframe with first column as time and the rest as interpolated counts.

        Raises:
            ValueError: If data is not an array, if kind is not a string, if pack_num is not an integer, and if
            new_time is not an array-like object or res is not an integer or float.
        """
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise ValueError(f"Data needs to be an array-like (i.e., list, tuple). Received a {type(data)}.")
        if not isinstance(kind, str):
            raise ValueError(f"Kind needs to be an string. Received a {type(kind)}.")
        if not isinstance(pack_num, int):
            raise ValueError(f"pack_num needs to be an integer. Received a {type(pack_num)}.")
        if res is not None:
            if not isinstance(res, (int, float)):
                raise ValueError(f"Resolution needs to be an integer or float. Received a {type(res)}.")
        if new_time is not None:
            if not isinstance(new_time, (Sequence, Mapping, np.ndarray, pd.Series)):
                raise ValueError(f"Output time needs to be an array-like (i.e., list, tuple). Received a {type(new_time)}.")
        else:
            if res is None:
                raise ValueError("If new_time is None, resolution interval cannot be NoneType.")
            else:
                # If there is not any new_time but a resolution, create a new time array:
                new_time = np.arange(data.iloc[0, 0], data.iloc[-1, 0], res / 1000)
                if len(new_time) < 2:
                    raise ValueError("Resolution interval is so high that it does not allow to interpolate the data. "
                                     f"Received resolution: {res} ms, but data has a time interval of "
                                     f"{round((data.iloc[-1, 0] - data.iloc[0, 0]) * 1000, 5)} ms.")
        if not isinstance(data, pd.DataFrame):
            columns = []
            data = pd.DataFrame(data)
        else:
            columns = data.columns
        time_data = np.asarray(data.iloc[:, 0])
        band_data = np.asarray(data.iloc[:, 1:])
        # Improve the interpolation intervals by removing redundant intervals:
        limits = _tools.get_index(time_data, new_time)
        time_data = time_data[limits[0]:limits[1]]
        # Interpolate the light curve:
        lc_interp = []
        time_intervals = _tools.slice_array(time_data, length=pack_num) if kind.lower() != 'linear' else None
        for i in range(band_data.shape[1]):
            band_i = band_data[:, i]
            band_i = band_i[limits[0]:limits[1]]
            if kind.lower() == 'linear':
                f = interp1d(time_data, band_i, kind=kind, fill_value="extrapolate")
                lc_interp.append(f(new_time))
            else:
                inter_i = np.array([])
                band_intervals = _tools.slice_array(band_i, length=pack_num)
                for j in range(0, len(time_intervals)):
                    time_j = np.asarray(time_intervals[j])
                    band_i_j = np.asarray(band_intervals[j])
                    try:
                        f = interp1d(time_j, band_i_j, kind=kind)
                    except ValueError:
                        warnings.warn(f"Error when using kind={kind} in {j} step, changing to linear interpolation.")
                        f = interp1d(time_j, band_i_j, kind='linear')
                    if j == 0:
                        new_times_j = new_time[(new_time <= time_j[-1]) & (new_time >= time_j[0])]
                    else:
                        new_times_j = new_time[(new_time <= time_j[-1]) & (new_time > time_j[0])]
                    inter_i = np.append(inter_i, f(new_times_j))
                lc_interp.append(inter_i)
        lc_interp = np.column_stack([new_time, *lc_interp])
        # Convert to Pandas Dataframe
        try:
            lc_interp = pd.DataFrame(lc_interp, columns=columns)
        except ValueError:
            lc_interp = pd.DataFrame(lc_interp)
        return lc_interp

    def parallel_grb_interpolate(
            self,
            data: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            new_times: Union[Sequence, Mapping, np.ndarray, pd.DataFrame] = None,
            res: float = 64,
            kind: str = 'linear',
            pack_num: int = 10,
    ):
        """Function to interpolate GRBs in a parallel way.

        Args:
            data (array-like): Input data array.
                The time data is assumed to be in the first column and the band measurements are in the next columns
                of each i-esim element of data.
            new_times (array-like): Output Time Array.
                It is assumed that the time array has the same length as data.
            res (float, optional): Resolution of the data interpolated in milliseconds. Defaults to 64.
            kind (str, optional): Kind of interpolation. Defaults to 'linear'.
                From Scipy Docs: Specifies the kind of interpolation as a string or as an integer specifying the order
                of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’
                , ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
            pack_num (int, optional): Number of data grouped per packet to interpolate. Defaults to 10.

        Returns:
            A list of Pandas Dataframes with first column as time and the rest as interpolated counts.
        """
        if new_times is None:
            new_times = repeat(None)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = list(tqdm(executor.map(self.grb_interpolate, data, new_times, repeat(res), repeat(kind),
                                             repeat(pack_num)), total=len(data), unit='GRB', desc='Interpolating: '))

        return results

    def noise_reduction_fabada(
            self,
            name: str,
            sigma: Union[Sequence, Mapping, np.ndarray, pd.Series, float] = None,
            save_data: bool = True,
    ):
        """Function to perform non-parametric noise reduction technique from FABADA to any GRB.

        Source (GitHub): https://github.com/PabloMSanAla/fabada
        FABADA is a novel non-parametric noise reduction technique which arise from the point of view of Bayesian
        inference that iteratively evaluates possible smoothed models of the data, obtaining an estimation of the
        underlying signal that is statistically compatible with the noisy measurements.

        Args:
            name (str): Name of the GRB.
            sigma (float or array, optional): Variance to use in the FABADA algorithm. Defaults to None.
                If sigma is None, the variance is estimated from the data using the RMS noise of an image. If sigma is a
                float, the same variance is used for all the bands. If sigma is an array, the variance is used for each
                band in ascending order.
            save_data (bool, optional): Whether to save the data or not. Defaults to True.

        Returns:
            A Pandas Dataframe with first column as time and the rest as reduced counts. If any error occurs, it returns
            a tuple of original data and error description.
        """
        if sigma is not None:
            if not isinstance(sigma, (Sequence, Mapping, np.ndarray, pd.Series, float)):
                raise TypeError(f"sigma must be a float, array or None. Obtained: {type(sigma)}")
        if not isinstance(save_data, bool):
            raise TypeError(f"save_data must be a bool. Obtained: {type(save_data)}")
        if not isinstance(name, str):
            raise TypeError(f"name must be a str. Obtained: {type(name)}")
        sig_check = True if sigma is None else False
        data = self.obtain_data(name=name, check_disk=True)
        try:
            limits = self.duration_limits(name=name, t=100)[0]
            low_lim, upper_lim = float(limits[1]), float(limits[2])
        except (ValueError, IndexError) as err:
            warnings.warn(f"Error when obtaining limits for {name}: {err}. It is not possible to reduce noise.",
                          RuntimeWarning)
            if save_data:  # Create an empty file to avoid errors when limiting data
                _tools.save_data(data=pd.DataFrame(), name=name, filename=f"{name}_{self.end}.h5",
                                 directory=self.noise_data_path)
            return data, err
        else:
            out_t100 = data[(data.iloc[:, 0] < low_lim) | (data.iloc[:, 0] > upper_lim)]
            out_t100 = out_t100[out_t100.iloc[:, 1:].any(axis=1)]  # Remove rows with all zeros to avoid errors
            if len(out_t100) > 0:
                columns_aux = [self.column_labels[i] for i in range(1, len(self.column_labels), 2)]
                out_t100 = out_t100[columns_aux]
                for i, column in enumerate(out_t100):
                    data_i = np.asarray(data[column])
                    if sig_check:
                        out_i = np.asarray([out_t100[column]])
                        sigma = np.square(_tools.estimate_noise(out_i))
                    data[column] = fabada(data_i, data_variance=sigma if isinstance(sigma, float) else sigma[i])
                if save_data:
                    file_name = f"{name}_{self.end}.h5"
                    _tools.save_data(data=data, name=name, filename=file_name, directory=self.noise_data_path)
                return data
            else:
                warnings.warn(f"No data outside T_100 found for {name}. It is not possible to reduce noise.",
                              RuntimeWarning)
                if save_data:
                    _tools.save_data(data=pd.DataFrame(), name=name, filename=f"{name}_{self.end}.h5",
                                     directory=self.noise_data_path)
                return data, f'length = {len(out_t100)}'

    def parallel_noise_reduction_fabada(
            self,
            names: Union[Sequence, Mapping, np.ndarray, pd.Series],
            sigma: Union[Sequence, Mapping, np.ndarray, pd.Series] = None,
            save_data: bool = True,
    ):
        """Function to perform non-parametric noise reduction technique from FABADA in a parallel way.

        Args:
            names (array-like): Names of the GRBs.
            sigma (float or array, optional): Variance to use in the FABADA algorithm. Defaults to None.
                If sigma is None, the variance is estimated from the data using the RMS noise of an image. If each
                sigma_i in the array is a float, the same variance is used for all bands. If instead each sigma_i is an
                array, each element of the i-esim array is used as variance for each band in ascending order.
            save_data (bool, optional): Whether to save the data or not. Defaults to True.

        Raises:
            TypeError: If names is not an array-like.
            TypeError: If sigma is not a float, array or None.
            TypeError: If save_data is not a bool.

        Returns:
            A list of Pandas Dataframes with first column as time and the rest as reduced counts (without modify error
            columns).
        """
        if not isinstance(names, (Sequence, Mapping, np.ndarray, pd.Series)):
            raise TypeError(f"names must be an array-like. Obtained: {type(names)}")
        if sigma is not None:
            if not isinstance(sigma, (Sequence, Mapping, np.ndarray, pd.Series, float)):
                raise TypeError(f"sigma must be a float, array or None. Obtained: {type(sigma)}")
        if not isinstance(save_data, bool):
            raise TypeError(f"save_data must be a bool. Obtained: {type(save_data)}")
        if sigma is None:
            sigma = repeat(None)
        if not isinstance(names, np.ndarray):
            names = np.asarray(names)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            results = list(tqdm(executor.map(self.noise_reduction_fabada, names, sigma, repeat(save_data)),
                                total=len(names), unit='GRB', desc='Noise reduction: '))
            if save_data:
                with open(os.path.join(self.noise_data_path, f"Errors_{self.end}.txt"), 'w') as f:
                    f.write("## GRB_Name\tError_Description\n")
                    for i in range(len(names)):
                        result = results[i]
                        f.write(f"{names[i]}\t{result[-1]}\n") if isinstance(result, tuple) else None
        return results

    @staticmethod
    def perform_tsne(
            data: Union[Sequence, Mapping, np.ndarray, pd.Series],
            library: str = 'sklearn',
            **kwargs
    ):
        """Function to perform tSNE using scikit-Learn or openTSNE implementations.

        Args:
            data (array-like): Preprocessed data to be embedded. Must be a 2D array.
            library (str, optional): Library to use. Defaults to 'sklearn'.
                Options: 'sklearn' or 'openTSNE'.
            **kwargs: Additional arguments to configure tSNE implementation, avoid to use 'random_state' as arg.

        Returns:
            2D array with the embedded data. Each row corresponds to a pair (x_i, y_i) of data transformed.
        """
        if not isinstance(data, (Sequence, Mapping, np.ndarray, pd.Series)):
            raise TypeError(f"data must be an array-like. Obtained: {type(data)}")
        if not isinstance(library, str):
            raise TypeError(f"library must be a str. Obtained: {type(library)}")
        if library.lower() == 'opentsne':  # OpenTSNE has by default init='pca'
            tsne = open_tsne(n_components=2, n_jobs=-1, random_state=42, **kwargs)
            data_reduced_tsne = tsne.fit(data)  # Perform tSNE to data
        else:  # However, sklearn_TSNE has by default init='random'
            tsne = sklearn_tsne(n_components=2, n_jobs=-1, random_state=42, **kwargs)
            data_reduced_tsne = tsne.fit_transform(data)  # Perform tSNE to data
        return data_reduced_tsne

    @staticmethod
    def plot_tsne(
            positions: Union[Sequence, Mapping, np.ndarray, pd.DataFrame],
            durations: Union[Sequence, Mapping, np.ndarray, pd.Series] = None,
            names: Union[Sequence, Mapping, np.ndarray, pd.Series] = None,
            special_cases: Union[Sequence, Mapping, np.ndarray, pd.Series] = None,
            redshifts: Union[Sequence, Mapping, np.ndarray, pd.Series] = None,
            marker_size: Union[Sequence, Mapping, np.ndarray, pd.Series, int] = 10,
            color_limits: tuple = None,
            legend_redshifts: bool = True,
            legend_special_cases: bool = True,
            redshift_kwargs: dict = None,
            special_marker: Union[Sequence, Mapping, np.ndarray, pd.Series, str] = None,
            special_marker_size: Union[Sequence, Mapping, np.ndarray, pd.Series, int] = None,
            special_marker_color: Union[Sequence, Mapping, np.ndarray, pd.Series, str] = None,
            non_special_marker_color: Union[Sequence, Mapping, np.ndarray, pd.Series, str] = None,
            kwargs_plot: dict = None,
            special_kwargs_plot: dict = None,
            special_kwargs_legend: dict = None,
            color_bar_kwargs: dict = None,
            picker: bool = False,
            return_colorbar: bool = False,
            ax=None
    ):
        """Function to plot tSNE results.

        Args:
            positions (array-like): 2D array with the embedded data. Each row corresponds to a pair (x_i, y_i) of data
                transformed.
            durations (array-like, optional): Duration of each GRB in seconds. Defaults to None.
            names (array-like, optional): Names of the GRBs. Defaults to None.
            special_cases (array-like, optional): Array with the special cases to highlight in map. Defaults to None.
            redshifts (array-like, optional): Array with the redshifts of each GRB. Defaults to None.
            marker_size (int or array-like, optional): Size of the markers. Defaults to 10.
                Unused when redshifts is not None. If array-like, each element of the i-esim array is used as size for
                the i-esim GRB.
            color_limits (tuple, optional): Limits of the colorbar. Defaults to None.
            legend_redshifts (bool, optional): Whether to show legend of redshifts or not. Defaults to True.
            legend_special_cases (bool, optional): Whether to show legend of special cases or not. Defaults to True.
            redshift_kwargs (dict, optional): Additional arguments to configure redshifts legend. Defaults to None.
            special_marker (str or array-like, optional): Marker to use for special cases. Defaults to None.
                If array-like, each i-esim element of the array is used as marker for the i-esim special GRB.
            special_marker_size (int or array-like, optional): Size of the markers for special cases. Defaults to None.
                If array-like, each i-esim element of the array is used as size for the i-esim special GRB. If int, all
                special markers will have the same size. It overrides marker_size if specified.
            special_marker_color (str or array-like, optional): Markers color for special cases. Defaults to None.
                If array-like, each i-esim element of the array is used as color for the i-esim special GRB. If str, all
                special markers will have the same color. It overrides durations colormap if specified.
            non_special_marker_color (str or array-like, optional): Non-special cases marker color. Defaults to None.
                If array-like, each i-esim element of the array is used as color for the i-esim non-special GRB. If str,
                all non-special markers will have the same color. It overrides durations colormap if specified.
            kwargs_plot (dict, optional): Additional arguments to configure main scatter. Defaults to None.
                It does not affect special cases scatter.
            special_kwargs_plot (dict, optional): Additional arguments to special cases scatter. Defaults to None.
            special_kwargs_legend (dict, optional): Additional arguments to special cases legend. Defaults to None.
            color_bar_kwargs (dict, optional): Additional arguments to configure color bar. Defaults to None.
            picker (bool, optional): Whether to enable picker or not. Defaults to False.
            return_colorbar (bool, optional): Whether to return color bar or not. Defaults to False.
            ax (matplotlib.axes.Axes, optional): Axes to plot in. Defaults to None.

        Raises:
            TypeError: If positions is not an array-like.
            TypeError: If durations is not an array-like.
            TypeError: If names is not an array-like.
            TypeError: If special_cases is not an array-like.
            TypeError: If redshifts is not an array-like.
            TypeError: If ax is not a matplotlib.axes.Axes.

        Returns:
            matplotlib.axes.Axes: Axes with the plot.
        """
        if not isinstance(positions, (Sequence, Mapping, np.ndarray, pd.DataFrame)):
            raise TypeError(f"positions must be an array-like. Obtained: {type(positions)}")
        if durations is not None:
            if not isinstance(durations, (Sequence, Mapping, np.ndarray, pd.Series)):
                raise TypeError(f"durations must be an array-like. Obtained: {type(durations)}")
            if len(durations) != len(positions):
                raise ValueError(f"Durations must have the same length as positions. Obtained: {len(durations)} "
                                 f"and {len(positions)}")
        if names is not None:
            if not isinstance(names, (Sequence, Mapping, np.ndarray, pd.Series)):
                raise TypeError(f"names must be an array-like. Obtained: {type(names)}")
            if len(names) != len(positions):
                raise ValueError(f"Names must have the same length as positions. Obtained: {len(names)} "
                                 f"and {len(positions)}")
        if special_cases is not None:
            if not isinstance(special_cases, (Sequence, Mapping, np.ndarray, pd.Series)):
                raise TypeError(f"special_cases must be an array-like. Obtained: {type(special_cases)}")
        if redshifts is not None:
            if not isinstance(redshifts, (Sequence, Mapping, np.ndarray, pd.Series)):
                raise TypeError(f"redshifts must be an array-like. Obtained: {type(redshifts)}")
            if len(redshifts) != len(positions):
                raise ValueError(f"Redshifts must have the same length as positions. Obtained: {len(redshifts)} "
                                 f"and {len(positions)}")
        if ax is not None:
            if not isinstance(ax, plt.Axes):
                raise TypeError(f"ax must be a matplotlib.axes.Axes. Obtained: {type(ax)}")
        if isinstance(marker_size, (Sequence, Mapping, np.ndarray, pd.Series)):
            if len(marker_size) != len(positions):
                raise ValueError(f"Marker size must have the same length as positions. Obtained: {len(marker_size)} "
                                 f"and {len(positions)}.")
        if redshift_kwargs is None:
            redshift_kwargs = {}
        if kwargs_plot is None:
            kwargs_plot = {}
        if special_kwargs_plot is None:
            special_kwargs_plot = {}
        if special_kwargs_legend is None:
            special_kwargs_legend = {}
        if color_bar_kwargs is None:
            color_bar_kwargs = {}
        if ax is None:
            fig, ax = plt.subplots(dpi=300)
        sca, color_bar = [], ()  # Define a default array to group scatter for Legends and color bar
        # Convert positions to numpy array if not:
        if not isinstance(positions, np.ndarray):
            positions = np.asarray(positions)
        x_i, y_i = positions[:, 0], positions[:, 1]  # Unpack tSNE results
        # Initialize sizes based on redshifts or marker_size and add a legend if required:
        sizes = np.ones(len(positions)) * marker_size if isinstance(marker_size, int) else marker_size
        if redshifts is not None:
            redshift = _tools.size_maker(redshifts)
            sizes = redshift * 600 / np.max(redshift)
            if legend_redshifts:
                for area in [1, 2, 4]:  # Do phantom scatter to put legend size
                    sca.append(ax.scatter([], [], c='k', alpha=0.3, s=area * 600 / max(redshift), label=f"$z={area}$"))
                first_leg = ax.legend(handles=sca, scatterpoints=1, frameon=False, labelspacing=1, fontsize='small',
                                      **redshift_kwargs)
                ax.add_artist(first_leg)  # Add legend to plot
                sca.clear()  # Reset variable to add further legends later
        # Initialize colors based on durations:
        colors, normalize = np.array(['darkslategray'] * len(positions)), None
        if durations is not None:
            colors = np.log10(durations)
            if color_limits is None:
                min_color, max_color = np.min(colors), np.max(colors)
            else:
                min_color, max_color = color_limits[0], color_limits[1]
            normalize = plt.Normalize(min_color, max_color)
        # Now divide the GRBs in two groups: special cases and normal cases:
        special_cases = [] if special_cases is None else special_cases
        match = np.where(np.isin(names, special_cases))[0]  # Check if there are special cases matching GRB Names
        non_match = np.arange(0, len(x_i), 1) if names is None else \
            np.where(np.isin(names, special_cases, invert=True))[0]
        if len(match) > 0:  # Plot special cases:
            if isinstance(special_marker, str):  # If special_marker is a string, use it for all special cases
                special_marker = np.array([special_marker] * len(match))
            if isinstance(special_marker_size, int):  # If special_marker_size is an int, use it for all special cases
                special_marker_size = np.array([special_marker_size] * len(match))
            if isinstance(special_marker_color, str):  # If special_marker_color is a string, use it for all special
                special_marker_color = np.array([special_marker_color] * len(match))
            mpl_markers = list(matplotlib.lines.Line2D.filled_markers[2:])  # Make a list of all possible markers
            markers = mpl_markers * len(match) if special_marker is None else special_marker  # Define multiple markers
            for j, i in enumerate(match):  # If there are special cases, plot them
                sc = ax.scatter(x_i[i], y_i[i], marker=markers[j], zorder=10, norm=normalize, edgecolor='k',
                                cmap='jet' if durations is not None else None,
                                c=colors[i] if special_marker_color is None else special_marker_color[j],
                                s=sizes[i] if special_marker_size is None else special_marker_size[j],
                                label=f'{names[i][:3]} {names[i][3:]}' if names is not None else None,
                                **special_kwargs_plot)
                sca.append(sc)
            if legend_special_cases:
                leg_args = {'loc': 'lower left', 'borderaxespad': 0., **special_kwargs_legend}
                second_legend = ax.legend(**leg_args, handles=sca, fontsize='small', frameon=False)
                ax.add_artist(second_legend)
        if isinstance(non_special_marker_color, str):
            non_special_marker_color = [non_special_marker_color] * len(non_match)
        main_sca = ax.scatter(x_i[non_match], y_i[non_match], zorder=5, cmap='jet' if durations is not None else None,
                              norm=normalize, s=sizes[non_match], picker=picker, **kwargs_plot,
                              c=colors[non_match] if non_special_marker_color is None else non_special_marker_color)
        main_sca.set_picker(picker) if picker else None
        if durations is not None:
            color_bar = plt.colorbar(main_sca, ax=ax, label=r'log$_{10}\left(T_{90}\right)$', **color_bar_kwargs)
        plt.tight_layout()
        plt.axis('off')
        if return_colorbar:  # Return color bar if required
            return ax, color_bar
        else:
            return ax

