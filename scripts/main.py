import os  # Import os to handle folders and files
import gzip  # Module to compress files
import sklearn  # Module to perform ML
import inspect  # Module to get information about live objects such as modules, classes
import warnings  # Module to print warnings
import requests  # Import the module needed to download from the website
import matplotlib  # Import module to edit and create plots
import numpy as np  # Import numpy module to read tables, manage data, etc
import concurrent.futures  # Import module to do threading over the bursts
import moviepy.editor as mpy  # Module to do Matplotlib animations
import matplotlib.pyplot as plt  # Import module to do figures, animations
from time import time  # Import Module to check times
from tqdm import tqdm  # Script to check progress bar in concurrency steps
from numpy import linalg  # Function to help sklearn
from scripts import helpers  # Script to do basics functions to data
from itertools import repeat  # Function to repeat some action
from scipy import integrate  # Module to integrate using Simpson's rule
from fabada import fabada  # Module to remove noise
from scipy.spatial.distance import cdist  # Module to compute distance between points
from scipy.interpolate import interp1d  # Module to interpolate data
from openTSNE import TSNE as open_TSNE  # Alternative module to do tSNE
from sklearn.manifold import TSNE as sklearn_TSNE  # Module to do tSNE
from scipy.fft import next_fast_len, fft, fftfreq, fftshift  # Function to look the best suitable array size to do FFT
from moviepy.video.io.bindings import mplfig_to_npimage  # Function to do mpy images

matplotlib.use("TkAgg")


def getSteps(data, **step_kwargs):
    old_gradient = sklearn.manifold._t_sne._gradient_descent  # Save original gradient descent function
    positions = []  # Array to save data positions

    def _gradient_descent_scikit(objective, p0, it, n_iter, n_iter_check=1, n_iter_without_progress=300, momentum=0.8,
                                 learning_rate=200.0, min_gain=0.01, min_grad_norm=1e-7, verbose=0, args=None,
                                 kwargs=None):
        """Batch gradient descent with momentum and individual gains.

        Parameters
        ----------
        objective : callable
            Should return a tuple of cost and gradient for a given parameter
            vector. When expensive to compute, the cost can optionally
            be None and can be computed every n_iter_check steps using
            the objective_error function.

        p0 : array-like of shape (n_params,)
            Initial parameter vector.

        it : int
            Current number of iterations (this function will be called more than
            once during the optimization).

        n_iter : int
            Maximum number of gradient descent iterations.

        n_iter_check : int, default=1
            Number of iterations before evaluating the global error. If the error
            is sufficiently low, we abort the optimization.

        n_iter_without_progress : int, default=300
            Maximum number of iterations without progress before we abort the
            optimization.

        momentum : float within (0.0, 1.0), default=0.8
            The momentum generates a weight for previous gradients that decays
            exponentially.

        learning_rate : float, default=200.0
            The learning rate for t-SNE is usually in the range [10.0, 1000.0]. If
            the learning rate is too high, the data may look like a 'ball' with any
            point approximately equidistant from its nearest neighbours. If the
            learning rate is too low, most points may look compressed in a dense
            cloud with few outliers.

        min_gain : float, default=0.01
            Minimum individual gain for each parameter.

        min_grad_norm : float, default=1e-7
            If the gradient norm is below this threshold, the optimization will
            be aborted.

        verbose : int, default=0
            Verbosity level.

        args : sequence, default=None
            Arguments to pass to objective function.

        kwargs : dict, default=None
            Keyword arguments to pass to objective function.

        Returns
        -------
        p : ndarray of shape (n_params,)
            Optimum parameters.

        error : float
            Optimum.

        i : int
            Last iteration.
        """
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
            positions.append(p.copy())  # Save the actual position
            check_convergence = (i + 1) % n_iter_check == 0
            # only compute the error when needed
            kwargs["compute_error"] = check_convergence or i == n_iter - 1

            error, grad = objective(p, *args, **kwargs)
            grad_norm = linalg.norm(grad)

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

                if verbose >= 2:
                    print("[t-SNE] Iteration %d: error = %.7f, gradient norm = %.7f (%s iterations in %0.3fs)"
                          % (i + 1, error, grad_norm, n_iter_check, duration))

                if error < best_error:
                    best_error = error
                    best_iter = i
                elif i - best_iter > n_iter_without_progress:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: did not make any progress during the last %d episodes. Finished."
                              % (i + 1, n_iter_without_progress))
                    break
                if grad_norm <= min_grad_norm:
                    if verbose >= 2:
                        print("[t-SNE] Iteration %d: gradient norm %f. Finished." % (i + 1, grad_norm))
                    break

        return p, error, i

    sklearn.manifold._t_sne._gradient_descent = _gradient_descent_scikit  # Change original gradient function
    sklearn_TSNE(random_state=42, **step_kwargs).fit_transform(data)  # Perform tSNE
    sklearn.manifold._t_sne._gradient_descent = old_gradient  # Return old gradient descent function
    return np.array(positions)  # Return all positions, texts in format (iteration, message)


class SwiftGRBWorker:
    col_bands = (0, 1, 3, 5, 7, 9)  # Columns in data containing measurements
    workers = os.cpu_count()  # Set how many workers you will use

    def __init__(self, root_path, res=64, end=None, data_path=None, original_data_path=None, noise_data_path=None,
                 results_path=None, noise_images_path=None, table_path=None, animations_path=None, n_bands=5):
        self.n_bands = n_bands  # Number of bands to be used through the object
        self.res = res
        if end is None:
            self.end = f"{res}ms" if res in (2, 8, 16, 64, 256) else f"1s"
        else:
            self.end = end
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
        self.animations_path = os.path.join(self.results_path, r"Animations") if animations_path is None \
            else animations_path

    def summary_tables_download(self, other=False, name=None):
        """
        Function to download summary tables needed from Swift
        :param other: Boolean to indicate if you need another specific table from Lists of GRBs with special comments
        :param name: Name of file in Lists of GRBs with special comments in Swift Data
        :return: Nothing, only download Data
        """
        path = self.table_path
        helpers.directory_maker(path)
        main_url = 'https://swift.gsfc.nasa.gov/results/batgrbcat/summary_cflux/'
        urls = {'GRBlist_single_pulse_GRB.txt': 'summary_GRBlist', 'summary_general.txt': 'summary_general_info',
                'summary_burst_durations.txt': 'summary_general_info',
                'GRBlist_redshift_BAT.txt': 'summary_general_info'}
        if other:
            urls[name] = 'summary_GRBlist'
        for key, value in urls.items():
            r = requests.get(f"{main_url}{value}/{key}", timeout=5)
            with open(os.path.join(path, key), 'wb') as f:
                f.write(r.content)

    def download_data(self, name, t_id):
        """
        Function to download a file for a GRB and ID name in Swift data in a specific folder
        :param name: GRB Name
        :param t_id: For the GRB,special ID associated with the trigger
        :returns: A tuple (GRB Name, None/<Error message>) to indicate if the file was created
        """
        helpers.directory_maker(self.original_data_path)  # Try to create the folder, unless it already has been created
        # First, we join the url with GRB name and ID (checking if it has 6 o 11 digits)
        i_d = f"00{t_id}000" if len(t_id) == 6 else t_id
        url = f"https://swift.gsfc.nasa.gov/results/batgrbcat/{name}/data_product/{i_d}-results/lc/{self.end}_lc_ascii.dat"
        try:  # Try to access to Swift Page
            r = requests.get(url, timeout=5)
            r.raise_for_status()  # We use this code to elevate the HTTP errors (i.e. wrong links) to exceptions
        # Then, if any exception has occurred, we send a message to indicate the error type:
        except requests.exceptions.RequestException as err:  # Notification message for all errors
            return name, err  # Return a tuple of GRB Name and error description
        else:
            file_name = f"{name}_{self.end}.gz"  # Define a unique file name for any name-resolution pair
            with gzip.open(os.path.join(self.original_data_path, file_name), mode="wb") as f:
                f.write(r.content)
            return name, None  # Return a tuple of GRB Name and None to indicate that file was created

    def so_much_downloads(self, names, t_ids, error=True):
        """
        Function to faster download a lot of GRB light curves, based in downloadGRBdata and Threads, it creates a
        'Errors.txt' file with all the failed processes if needed
        :param names: A list with all GRB names
        :param t_ids: A list with all Trigger_IDs associated with the GRBs
        :param error: Boolean to indicate if you need 'Errors.txt' file, default is True
        :return: Nothing
        """
        # Try to create the folder in the associated path, unless it already has been created --> This is to mitigate an
        # FileNotFoundError with open(path) when the directory is created while Threading is executing
        helpers.directory_maker(self.original_data_path)

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            results = tqdm(executor.map(self.download_data, names, t_ids),
                           total=len(names), desc='Downloading: ', unit='GRB')

            if error:
                with open(os.path.join(self.original_data_path, f"Errors_{self.end}.txt"), 'w') as f:
                    f.write("## GRB Name|Error Description\n")
                    for name, value in results:
                        f.write(f"{name}|{value}\n") if value is not None else None

    def durations_checker(self, name=None, t=100, durations_table="summary_burst_durations.txt"):
        """
        Function to extract GRB duration
        :param name: GRB Name, if None then all GRBs available in durations table will be returned
        :param t: Duration interval needed, can be 50, 90, 100 (default)
        :param durations_table: Durations table name, related with table path class variable
        :return: Array with format [[Name_i, T_initial, T_final], ...]
        """
        columns = {50: (0, 7, 8), 90: (0, 5, 6), 100: (0, 3, 4)}  # Dictionary to extract the correct columns
        path = os.path.join(self.table_path, durations_table)
        keys_extract = np.genfromtxt(path, delimiter="|", dtype=str, usecols=columns.get(t), autostrip=True)
        # Extract all values from summary_burst_durations.txt:
        if isinstance(name, str):  # If a specific name is entered, then we search the value in the array
            return helpers.check_name(name, keys_extract)
        elif isinstance(name, (np.ndarray, list, tuple)):  # If a name's array is specified, search them recursively
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
                results = list(
                    tqdm(executor.map(helpers.check_name, name, np.array([keys_extract] * len(name))), total=len(name),
                         desc='Finding Durations: ', unit='GRB'))
                return np.array(results)
        else:  # Else return data for all GRBs
            return keys_extract

    def redshifts_checker(self, name=None, redshift_table="GRBlist_redshift_BAT.txt"):
        """
        Function to extract GRB redshift, if available
        :param name: GRB Name
        :param redshift_table: Redshift table name, related with table path class variable
        :return: Array with format [[Name_i, Z], ...]
        """
        path = os.path.join(self.table_path, redshift_table)
        keys_extract = np.genfromtxt(path, delimiter="|", dtype=str, usecols=(0, 1), autostrip=True)
        # Extract all values from summary_burst_durations.txt:
        if isinstance(name, str):  # If a specific name is entered, then we search the redshift in the array
            return helpers.check_name(name, keys_extract)
        elif isinstance(name, (np.ndarray, list, tuple)):  # If a name's array is specified, search them recursively
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
                results = list(
                    tqdm(executor.map(helpers.check_name, name, np.array([keys_extract] * len(name))), total=len(name),
                         desc='Finding Redshifts: ', unit='GRB'))
                returns = []
                for i in range(len(name)):
                    returns.append(results[i][0]) if len(results[i]) != 0 else returns.append([name[i], None])
                return np.array(returns)
        else:  # Else return data for all GRBs
            return keys_extract

    def lc_limiter(self, name, t=100, limits=None):
        """
        Function to extract GRB data out of (t_start, t_end), it is needed to specify these limits or the T_{i} duration
        :param name: GRB Name
        :param t: Duration interval needed, can be 50, 90, 100 (default)
        :param limits: List of customized [t_start, t_end] if needed (default is None)
        :return: A list with values into t_start and t_end, or None if data doesn't exist or has length < 4
        """
        assert t in (50, 90, 100), f"Valid duration intervals: 50, 90, 100. Got: {t}"
        try:
            if limits is None:  # If not have been defined any limits, then use T_{t} durations from file
                limit_interval = self.durations_checker(name, t)  # Upload the values of t_start and t_durations
                t_start, t_end = float(limit_interval[0, 1]), float(limit_interval[0, 2])
            else:
                t_start, t_end, *other = limits
                t_start, t_end = float(t_start), float(t_end)
            # Unpack the downloaded data for the file without errors, if it exists:
            data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{self.end}.gz"), autostrip=True,
                                 usecols=self.col_bands)
            # Filter values between t_start and t_end in data:
            data = data[(data[:, 0] > t_start) & (data[:, 0] < t_end)]
            # data = np.array([value for value in data if t_end >= value[0] >= t_start])
        except FileNotFoundError:  # If file is not found, return error
            return name, ' ', ' ', 'FileNotFoundError'
        except ValueError:  # If there aren't any valid T_start or T_end, return error
            return name, ' ', ' ', 'ValueError'
        except IndexError:  # If there aren't row in burst durations, return
            return name, ' ', ' ', 'IndexError'
        else:
            if len(data) < 3:
                return name, t_start, t_end, f'Length={len(data)}'
            elif np.all((data[:, 1:] == 0)):  # Check if data is only zeros
                return name, t_start, t_end, 'Only zeros'
            else:  # Check if there are more than 2 points in light curve
                return data

    def so_much_lc_limiters(self, names, t=100, limits=None):
        """
        Function to faster limit a lot of GRB light curves, based in lc_limiter, it returns (results, errors) tuple
        :param names: A list with all GRB names
        :param t: Duration interval needed, can be 50, 90, 100 (default)
        :param limits: List of customized [t_start, t_end] if needed (default is None)
        :return: A tuple (results, new_names, errors), where errors have formatted (name, t_start, t_end, ErrorType) and
        new_names is an array with GRB names in the same order as results (without errors)
        """
        errors = []  # Define errors
        non_errors = []  # Define non-errors array
        new_names = []  # Define new_names array

        if limits is None:  # If no limits are entered, then only send None array
            limits = repeat(None, len(names))
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
            results = list(tqdm(executor.map(self.lc_limiter, names, repeat(t, len(names)), limits), total=len(names),
                                desc='LC Limiting: ', unit='GRB'))
        assert len(results) == len(names), f"The length of names array and returning data are not the same. Ending..."
        for i in range(len(results)):
            array = results[i]
            if isinstance(array[0], str):
                errors.append(array)
            else:
                non_errors.append(array)
                new_names.append(names[i])
        return non_errors, new_names, np.array(errors)

    def lc_normalizer(self, data, print_area=False):
        """
        Function to normalize GRB light curve data, it needs to have the total time-integrated counts as last column
        :param data: Array with data in format time, (band, error) for 15-25, 25-50, 50-100, 100-350 and 15-350 keV
        :param print_area: Boolean to indicate if returns Total time-integrated flux
        :return: Array with values normalized using the 15-350 keV integral
        """
        data = np.array(data)
        area = integrate.simpson(data[:, -1], x=data[:, 0])  # Integral for 15-350 KeV data
        data[:, 1:] /= area  # Normalize the light curve
        if print_area:
            return data[:, 1:self.n_bands+1], area
        else:
            return data[:, 1:self.n_bands+1]

    def so_much_normalize(self, more_data, print_area=False):
        """
        Function to faster normalize data using the lc_normalizer instance
        :param more_data: Array containing all individual data to be normalized
        :param print_area: Boolean to indicate if returns Total time-integrated flux
        :return:
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
            results = list(tqdm(executor.map(self.lc_normalizer, more_data, repeat(print_area, len(more_data))),
                                total=len(more_data), desc='LC Normalizing: ', unit='GRB'))
        return results

    @staticmethod
    def zero_pad(data, length):
        """
        Function to zero pad array preserving order in time axis, it will delete time axis (assumed in first column)
        :param data: Data array to be zero-padded
        :param length: Length of the final array (need to be more than data length)
        :return: New array zero-padded, with added values of time basis for a given resolution
        """
        diff = length - len(data)  # Difference between actual and optimal array size
        data_plus_zeros = np.pad(data, ((0, diff), (0, 0)))  # Zero pad array
        return data_plus_zeros

    def so_much_zero_pad(self, more_data):
        """
        Function to faster zero pad data using the zero_pad instance
        :param more_data: Data arrays to be zero padded
        :return: New array zero-padded, with added values of time basis for a given resolution
        """
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:
            lengths = set(executor.map(len, more_data))
            max_length = next_fast_len(max(lengths))
            results = list(tqdm(executor.map(self.zero_pad, more_data, repeat(max_length, len(more_data))),
                                total=len(more_data), desc='LC Zero-Padding: ', unit='GRB'))
        return np.array(results, dtype=float)

    @staticmethod
    def only_concatenate(array):
        """
        Function to concatenate light curve energy bands in ascendant mode, only works on Swift data format
        :param array: Data for a GRB without error columns
        :return: One single array with concatenated data
        """
        energy_bands = array.transpose()  # Erase the time-error columns and transpose
        concatenate = np.reshape(energy_bands, len(energy_bands) * len(energy_bands[0]))  # Concatenate all data columns
        return concatenate

    def so_much_concatenate(self, arrays):
        """
        Function to faster concatenate data using the only_concatenate instance
        :param arrays: Data array, energy bands need to be in 2, 4, 6, 8 and 10th column
        :return: One single N-array with concatenated data (N = arrays length)
        """
        m = len(arrays)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
            cc_results = list(tqdm(executor.map(self.only_concatenate, arrays), total=m,
                                   desc='Concatenating: ', unit='GRB'))
        return np.array(cc_results)

    def fourier_concatenate(self, array, plot=False, name=None):
        """
        Function to do DFT to a GRB data from Swift, data need to be normalized before executed here
        :param array: GRB data array to be transformed, it would be sent in swift format (11 columns with time first)
        :param plot: Boolean value to indicate if plot is needed, default is False
        :param name: Name of GRB to save plot, default is None, it is needed if plot=True
        :return: An array with the Fourier Amplitude Spectrum of data
        """
        concatenated = self.only_concatenate(array)
        sp = fftshift(np.abs(fft(concatenated)))  # Perform DFT to data and get the Fourier Amplitude Spectrum
        sp_one_side = sp[:len(sp) // 2]  # Sampling below the Nyquist frequency
        if plot:
            spacing = self.res * 1e-3 if self.res in (2, 8, 16, 64, 256) else 1  # Get resolution from object
            freq = fftfreq(sp.size, d=spacing)[:len(sp) // 2]  # Get frequency spectrum basis values for one side
            dft_fig, axs1 = plt.subplots(2, 1, dpi=150, figsize=[10, 7], gridspec_kw={'height_ratios': [0.7, 0.4]})
            axs1[0].set_title(fr"{name} DFT", weight='bold').set_fontsize('12')
            axs1[1].plot(freq[:4 * (len(freq) // 5)], sp_one_side[:4 * (len(freq) // 5)], linewidth=0.5, c='k')
            axs1[1].set_xlim(left=-0.05, right=freq[4 * (len(freq) // 5)])
            axs1[0].plot(freq[4 * (len(freq) // 5):], sp_one_side[4 * (len(freq) // 5):], linewidth=0.5, c='k')
            axs1[0].set_xlim(left=freq[4 * (len(freq) // 5)])
            axs1[1].set_xlabel('Frequency (Hz)', weight='bold').set_fontsize('10')
            axs1[1].set_ylabel('Amplitude', weight='bold').set_fontsize('10')
            axs1[0].set_ylabel('Amplitude', weight='bold').set_fontsize('10')
            return sp_one_side, dft_fig
        else:
            return sp_one_side

    def so_much_fourier(self, arrays):
        """
         Function to do faster DFT to so much GRB data from Swift, data need to be normalized before executed here
         :param arrays: GRB data array to be transformed, it would be sent in swift format (11 columns with time first)
         :return: An array with the DFT Amplitude of total data
         """
        m = len(arrays)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
            sp_results = list(tqdm(executor.map(self.fourier_concatenate, arrays), total=m,
                                   desc='Performing DFT: ', unit='GRB'))
        return np.array(sp_results)

    def plot_any_grb(self, name, t=100, limits=None, kind="Default", ax=None):
        """
        Function to plot any GRB out of i-esim duration (can be 50, 90, 100), default is T_90
        :param name: GRB name
        :param t: Duration interval needed, default is 100 (can be 50, 90, 100, else plot all light curve)
        :param limits: List of customized [t_start, t_end] if needed (default is None)
        :param kind: String to indicate type of plot. It can be "Interpolated" (Change data to background-gray scaled),
        "Concatenated" (set the 5 bands in one panel) or "Default" (default Swift mode)
        :param ax: Custom Matplotlib axes element to scatter, if ax is None, it will be created
        :return: Returns the axis created
        """
        assert kind.lower() in ("interpolate", "concatenated", "default")
        sizes = {"interpolate": [7, 9], "concatenated": [6.4, 2.4], "default": [6.4, 4.8]}
        ax_bool = True if ax else False
        if ax is None:  # If there aren't any previous axes, create new one
            fig_s = sizes.get(kind.lower())
            if kind.lower() == "concatenated":
                fig5, ax = plt.subplots(dpi=150, figsize=fig_s)
                ax.set_ylabel(r"Counts/sec/det", weight='bold').set_fontsize('10')
            else:
                fig5 = plt.figure(dpi=150, figsize=fig_s)
                gs = fig5.add_gridspec(nrows=5, hspace=0)
                ax = gs.subplots(sharex=True)
            fig5.text(0.06, 0.5, r"Counts/sec/det", ha='center', va='center', rotation='vertical', weight='bold',
                      fontsize=10)
        low_sub = ax if kind.lower() == "concatenated" else ax[4]  # Define lower and upper panels in each case
        high_sub = ax if kind.lower() == "concatenated" else ax[0]
        low_sub.set_xlabel('Time since BAT Trigger time (s)', weight='bold').set_fontsize('10')
        bands = (r"$15-25\,keV$", r"$25-50\,keV$", r"$50-100\,keV$", r"$100-350\,keV$", r"$15-350\,keV$")
        colors = ('k', 'r', 'lime', 'b', 'tab:pink')
        if t in (50, 90, 100) or limits is not None:  # Limit Light Curve if is needed
            data = self.lc_limiter(name=name, t=t, limits=limits)
            high_sub.set_title(fr"{self.end} Swift {name} out of $T\_{t}$", weight='bold').set_fontsize('12')
            if isinstance(data[0], str):  # If an error occurs when limit out of T_{t}
                print(f"Error when limiting {name} Light Curve out of t={t}, plotting all data")
                data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{self.end}.gz"), autostrip=True,
                                     usecols=self.col_bands)
                high_sub.set_title(f"{self.end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        else:  # If not needed, only extract total data
            data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{self.end}.gz"), autostrip=True,
                                 usecols=self.col_bands)
            high_sub.set_title(f"{self.end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        low_sub.set_xlim(left=data[0, 0], right=data[-1, 0])
        for i in range(5):
            if kind.lower() == "interpolate":
                ax[i].plot(data[:, 0], data[:, i+1], label=bands[i], alpha=0.3, ms=0.5)
            elif kind.lower() == "concatenated":
                ax.plot(data[:, 0], data[:, i+1], label=bands[i], linewidth=0.5, c=colors[i]) if i < 4 else None
            else:
                ax[i].plot(data[:, 0], data[:, i+1], label=bands[i], linewidth=0.5, c=colors[i])
            ax[i].legend(fontsize='xx-small', loc="upper right") if kind.lower() != "concatenated" else None
            ax.legend(fontsize='xx-small', loc="upper right") if (kind.lower() == "concatenated" and i == 4) else None
        if not ax_bool:
            return fig5, ax
        else:
            return ax

    def save_data(self, file_name, names, data):
        """
        Function to save pre-processed data in a compressed format, the data will be saved in the results' folder
        :param file_name: Name of file to be created, if it is a string or a Path, the .npz extension will be appended
        :param names: Array of GRB names, needed to preserve order in data
        :param data: Array of pre-processed data in same order as names
        :return: None
        """
        helpers.directory_maker(self.results_path)
        np.savez_compressed(os.path.join(self.results_path, file_name), GRB_Names=names, Data=data)

    @staticmethod
    def perform_tsne(data, library="sklearn", **kwargs):
        """
        Function to perform tSNE using scikit-Learn or openTSNE implementations
        :param data: Pre-processed data to be embedded
        :param library: Str to indicate which library will be used, can be 'sklearn' (default) or 'openTSNE'
        :param kwargs: Additional arguments to configure tSNE implementation, avoid to use 'random_state' as arg
        :return: 2D array with transformed values (x_i, y_i) for each data
        """
        assert library.lower() in ('opentsne', 'sklearn'), f"Library only could be sklearn or openTSNE, got: {library}"
        # Create an object to initialize tSNE in any library, with default values and other variables needed:
        if library.lower() == 'opentsne':  # OpenTSNE has by default init='pca'
            tsne = open_TSNE(n_components=2, n_jobs=-1, random_state=42, **kwargs)
            data_reduced_tsne = tsne.fit(data)  # Perform tSNE to data
        else:  # However, sklearn_TSNE has by default init='random'
            tsne = sklearn_TSNE(n_components=2, n_jobs=-1, random_state=42, **kwargs)
            data_reduced_tsne = tsne.fit_transform(data)  # Perform tSNE to data
        return data_reduced_tsne

    @staticmethod
    def tsne_scatter_plot(x, duration_s=None, names=None, special_cases=None, redshift=None, ax=None, animation=False):
        """
        Function to do versatile plots of tSNE results (only 2D embedding results)
        :param x: tSNE results array
        :param duration_s: Durations (T_90, T_100, etc.) dataset array to add color bar in scatter plot if needed
        :param names: GRB Names array, used only if you need to highlight some GRBs
        :param special_cases: Special GRBs array to highlight, names parameter is needed if there are special_cases
        :param redshift: Redshift array for all GRBs, it is used to scale point size
        :param ax: Custom Matplotlib axes element to scatter, if ax is None, it will be created
        :param animation: Boolean to indicate if returns color bar to animate, only needed if you will animate tSNE
        :return: Matplotlib axis object ax with plot changes
        """
        redshift = [] if redshift is None else redshift  # Check if variables are not None
        special_cases = [] if special_cases is None else special_cases
        names = [] if names is None else names
        duration_s = [] if duration_s is None else duration_s
        x_i, y_i = x[:, 0], x[:, 1]  # Unpack tSNE results
        if ax is None:  # If there aren't any previous axes, create new one
            tsne_figure, ax = plt.subplots(dpi=300)  # Create plot instances
        sca, color_bar = [], ()  # Define a default array to group scatter for Legends and color bar
        if len(redshift) == 0:  # If there aren't any redshift, then do the same size for all points
            size = np.ones(len(x)) * 50
        else:  # If there are redshift values, get sizes and legend them
            redshift = helpers.size_maker(redshift)  # Get size for redshift returned in redshifts_checker function
            size = redshift * 600 / max(redshift)  # Scale size to usual plotting values
            for area in [1, 2, 4]:  # Do phantom scatter to put legend size
                sca.append(ax.scatter([], [], c='k', alpha=0.3, s=area * 600 / max(redshift), label=f"z={area}"))
            first_leg = ax.legend(handles=sca, scatterpoints=1, frameon=False, labelspacing=1, fontsize='small')
            ax.add_artist(first_leg)  # Add legend to plot
            sca.clear()  # Reset variable to add further legends later
        match = np.where(np.isin(names, special_cases))[0]  # Check if there are special cases matching GRB Names
        non_match = np.arange(0, len(x_i), 1) if len(names) == 0 else \
            np.where(np.isin(names, special_cases, invert=True))[0]
        alp = 0.7 if len(match) != 0 or animation else 1
        if len(duration_s) != 0:  # If there are durations, then customize scatters and add color bar
            ncolor = np.log10(duration_s)
            minimum, maximum = min(ncolor), max(ncolor)
            normalize = plt.Normalize(minimum, maximum)
            main_scatter = ax.scatter(x_i[non_match], y_i[non_match], s=size[non_match], c=ncolor[non_match],
                                      cmap='jet', norm=normalize, alpha=alp)  # Scatter non-matched GRBs
            markers = matplotlib.lines.Line2D.filled_markers[1:] * len(match)  # Define multiple markers
            for it, marker_i in zip(match, markers):  # Scatter matched GRBs
                if size[it] != 0:  # Skip special GRBs without redshift
                    sca.append(ax.scatter(x_i[it], y_i[it], s=size[it], c=ncolor[it], edgecolor='k', norm=normalize,
                                          zorder=2.5, label=f'{names[it][:3]} {names[it][3:]}', marker=marker_i,
                                          cmap='jet'))
            if len(match != 0):  # If there are special cases to show, put a customized legend
                leg_args = {'bbox_to_anchor': (-0.26, 0.7125), 'loc': 'lower left', 'borderaxespad': 0.}
                second_legend = ax.legend(**leg_args, handles=sca, fontsize='small', frameon=False)
                ax.add_artist(second_legend)
            color_bar = plt.colorbar(main_scatter if len(match) == 0 else sca[0],
                                     label=r'log$_{10}\left(T_{90}\right)$')
        else:  # If there aren't durations, do simple scatter plots
            ax.scatter(x_i, y_i, s=size)
        plt.tight_layout()  # Adjust plot to legends
        plt.axis('off')  # Erase axis, it means nothing in tSNE
        if animation:
            return color_bar  # If animation are needed, return color bar to erase it later
        else:
            return ax  # Otherwise, return axis object

    def tsne_animation(self, data, filename=None, iterable='perplexity', **kwargs):
        """
        Instance to perform animations using any iterable of scikit-Learn implementation of TSNE
        :param data: Pre-processed data to be embedded
        :param filename: Name of file to be saved, default value is None
        :param iterable: Name of TSNE iterable in scikit Learn, it needs to be exact as TSNE arguments
        :param kwargs: Iterable array and other additional arguments. It can be arguments for TSNE or tsne_scatter_plot
        function, but it is necessary to add 'iterable' array as an argument
        :return: Moviepy animation object to further editing
        """
        fig, ax_i = plt.subplots()
        fps = 1  # Images generated = fps*duration
        array_it = kwargs.pop(iterable)  # Catch iterable array
        duration = len(array_it) // fps  # Define the duration as len(iterable)/fps
        scatter_args = set(inspect.signature(self.tsne_scatter_plot).parameters)  # Check for function parameters
        scatter_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in scatter_args}  # Separate plot parameters

        def make_frame(t):  # t has a step equal to 1/fps
            ax_i.clear()  # Clear axes between iterations
            dict_iterable = {iterable: array_it[int(t // (1 / fps))]}  # Define a unique i-esim parameter to pass tSNE
            x_i = self.perform_tsne(data, **dict_iterable, **kwargs)  # Perform tSNE
            bar = self.tsne_scatter_plot(x_i, **scatter_dict, ax=ax_i, animation=True)  # Scatter results
            ax_i.set_title(f"{iterable}:{dict_iterable.get(iterable)}", loc='left', style='italic', fontsize=10)
            fig.subplots_adjust(top=0.94)  # Resize fig to make title
            frame = mplfig_to_npimage(fig)  # Convert to mpy
            bar.remove()  # Remove bar
            return frame  # Return frame fig

        animation = mpy.VideoClip(make_frame, duration=duration)  # Create animation object
        animation.write_gif(filename, fps=fps, program='imageio') if filename is not None else None  # Save animation
        return animation

    def convergence_animation(self, data, filename=None, **kwargs):
        """
        Instance to perform TSNE convergence animations using scikit-Learn implementation
        :param data: Pre-processed data to be embedded
        :param filename: Name of file to be saved, default value is None
        :param kwargs: Other additional arguments, can be TSNE or tsne_scatter_plot function arguments
        :return: Moviepy animation object to further editing
        """
        scatter_args = set(inspect.signature(self.tsne_scatter_plot).parameters)  # Check for function parameters
        scatter_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in scatter_args}  # Separate plot parameters
        positions_an = getSteps(data, **kwargs)  # Perform tSNE and save iterations
        fps = 20  # Set gif fps
        duration = len(positions_an) // fps  # Define the duration as len(iterable)/fps
        conv_fig, conv_ax = plt.subplots()  # Create figure object

        def make_iteration(t):
            conv_ax.clear()
            iteration = int(t // (1 / fps))
            position_i = positions_an[iteration].reshape(-1, 2)
            bar = self.tsne_scatter_plot(position_i, **scatter_dict, animation=True, ax=conv_ax)
            conv_ax.set_title(f"Iteration: {iteration}", loc='left', style='italic', fontsize=10)
            conv_fig.subplots_adjust(top=0.94)
            frame = mplfig_to_npimage(conv_fig)  # Convert to mpy
            bar.remove()  # Remove bar
            return frame

        animation = mpy.VideoClip(make_iteration, duration=duration)
        animation.write_gif(filename, fps=fps, program='imageio') if filename is not None else None
        return animation

    def noise_reduction_fabada(self, name, plot=False, save_data=False, save_fig=True):
        """
        Function to perform non-parametric noise reduction technique from FABADA to any GRB
        :param name: GRB Name
        :param plot: Boolean to indicate if plotting is needed
        :param save_data: Boolean to indicate if to save noise reduced data
        :param save_fig: Boolean to indicate if to save plot
        :return: noise reduced data array and figure object (if plot=True)
        """
        arr = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{self.end}.gz"), autostrip=True)
        data = arr.copy()
        limits = self.durations_checker(name=name)
        # Filter values outside T_100 ( change < signs and add & to get inside)
        outside_T100 = data[(data[:, 0] < float(limits[0, 1])) | (data[:, 0] > float(limits[0, 2]))]
        outside_T100_non_zero = outside_T100[outside_T100[:, 1:].all(1)]  # Remove zero columns in data
        if len(outside_T100_non_zero) != 0:  # If there aren't any values outside duration, let default values
            for i in range(1, 11, 2):
                band_i = np.array([outside_T100_non_zero[:, i]])  # Extract band outside T_100 and make 2D array
                sigma = np.square(helpers.estimate_noise(band_i))  # Estimate noise variance outside T_100
                data[:, i] = fabada(data[:, i], data_variance=sigma)  # Change values from i-esim band using fabada
        if save_data:
            file_name = f"{name}_{self.end}.gz"  # Define a unique file name for any name-resolution pair
            np.savetxt(os.path.join(self.noise_data_path, file_name), data)  # Save file as gz
        if plot:
            arr_T100 = arr[(arr[:, 0] > float(limits[0, 1])) & (arr[:, 0] < float(limits[0, 2]))]
            data_T100 = data[(data[:, 0] > float(limits[0, 1])) & (data[:, 0] < float(limits[0, 2]))]
            fig5 = plt.figure(dpi=200, figsize=[2 * 6.4, 6])
            gs = fig5.add_gridspec(nrows=5, ncols=2, hspace=0)
            axs = gs.subplots()
            [axs[4][i].set_xlabel('Time since BAT Trigger time (s)', weight='bold').set_fontsize('10') for i in (0, 1)]
            fig5.text(0.085, 0.5, r"Counts/sec/det", ha='center', va='center', rotation='vertical', weight='bold',
                      fontsize=10)
            bands = (r"$15-25\,keV$", r"$25-50\,keV$", r"$50-100\,keV$", r"$100-350\,keV$", r"$15-350\,keV$")
            colors = ('k', 'r', 'lime', 'b', 'tab:pink')
            fig5.suptitle(f"{self.end} Swift {name}", weight='bold').set_fontsize('10')
            axs[0][0].set_title("Total Light Curve", weight='bold').set_fontsize('10')
            axs[0][1].set_title(r"Inside $\mathbf{T_{100}}$", weight='bold').set_fontsize('10')
            for j in range(5):
                axs[j][1].plot(arr_T100[:, 0], arr_T100[:, 2 * j + 1], alpha=0.2, c='k')
                axs[j][1].plot(data_T100[:, 0], data_T100[:, 2 * j + 1], label=bands[j], linewidth=0.75, c=colors[j])
                axs[j][1].legend(fontsize='xx-small', loc="upper right")
                axs[j][0].plot(arr[:, 0], arr[:, 2 * j + 1], alpha=0.2, c='k')
                axs[j][0].plot(data[:, 0], data[:, 2 * j + 1], label=bands[j], linewidth=0.75, c=colors[j])
                axs[j][0].legend(fontsize='xx-small', loc="upper right")
                axs[j][0].set_xlim(left=data[0, 0], right=data[-1, 0])
                axs[j][1].set_xlim(left=data_T100[0, 0], right=data_T100[-1, 0])
            helpers.directory_maker(self.noise_images_path)
            if save_fig:
                fig5.savefig(os.path.join(self.noise_images_path, f"{name}_{self.end}.png"))
                plt.close()
            return data, fig5
        else:
            return data

    def so_much_noise_filtering(self, names, plot=False, save_data=False, save_fig=True):
        """
        Function to faster perform non-parametric noise reduction technique from FABADA
        :param names: A list with all GRB names
        :param plot: Boolean to indicate if plotting is needed
        :param save_data: Boolean to indicate if to save noise reduced data
        :param save_fig: Boolean to indicate if to save plot
        :return: Noise Filtered Data array, only if plot is False, else None
        """
        m = len(names)
        helpers.directory_maker(self.noise_data_path)
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
            results = list(tqdm(executor.map(self.noise_reduction_fabada, names, repeat(plot, m), repeat(save_data, m),
                                             repeat(save_fig, m)), total=m, desc='Filtering Noise: ', unit='GRB'))
        if not plot:
            return results

    @staticmethod
    def one_band_interpolate(times, counts, new_time, pack_num=10, kind='linear', name=None):
        """
        Function to interpolate one band from any GRB light curve
        :param times: Original time array
        :param counts: Original GRB counts array
        :param new_time: New times array to be extrapolated
        :param pack_num: The number of data grouped per packet to interpolate, a smaller number of points can improve
        the results, but a value too large or small can lead to poor interpolation results. Deprecated if kind='linear'
        :param kind: Specifies the kind of interpolation as a string or as an integer specifying the order
        of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’,
        ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’ (took from Scipy Docs).
        :param name: GRB Name
        :return: Interpolated array for new_time
        """
        assert len(times) == len(counts), f"Expected equal time and counts array length: {len(times)}, {len(counts)}"
        new_counts = np.array([])
        new_times = np.array([])
        if kind.lower() == 'linear':  # In linear interpolation, the pack size doesn't matter
            pack_num = len(times)
        for i in range(0, len(times), pack_num):  # Iterate over all the time array
            if i + pack_num < len(times):  # Check for right wall effect
                time_i = times[i:i + pack_num + 1]
                band_i = counts[i:i + pack_num + 1]
            else:
                time_i = times[i:]
                band_i = counts[i:]
            new_times_i = new_time[(new_time >= time_i[0]) & (new_time <= time_i[-1])]
            new_times = np.append(new_times, new_times_i)
            if len(new_times_i) != 0:
                try:
                    f = interp1d(time_i, band_i, kind=kind)  # Interpolate
                except ValueError:  # If there are any error during the interpolation, try again using kind='linear'
                    warnings.warn(f"{name} --> Error when using kind={kind}, changing to linear interpolation...")
                    f = interp1d(time_i, band_i, kind='linear')
                new_counts = np.append(new_counts, f(new_times_i))
        assert len(new_counts) == len(new_time), f"The resulting arrays doesn't have the same size for {name}: " \
                                                 f"{len(new_counts)} {len(new_time)}"
        return new_counts

    def one_grb_interpolate(self, name, resolution=1, pack_num=10, kind='linear', t=None, limits=None, plot=True,
                            save_fig=True):
        """
        Function to interpolate all bands from one GRB
        :param name: GRB Name
        :param resolution: New resolution for data in ms, used to create new time array
        :param pack_num: The number of data grouped per packet to interpolate, a smaller number of points can improve
        the results, but a value too large or small can lead to poor interpolation results. Deprecated if kind='linear'
        :param kind: Specifies the kind of interpolation as a string or as an integer specifying the order
        of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’,
        ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’ (took from Scipy Docs).
        :param t: GRB Duration interval to be interpolated, can be 50, 90, 100 (default) or None (total light curve)
        :param limits: List of customized [t_start, t_end] to interpolate if needed (default is None)
        :param plot: Boolean to indicate if plot is necessary
        :param save_fig: Boolean to indicate if to save plot
        :return: Interpolated array for custom resolution in all GRB bands
        """
        if t is None and limits is None:  # If there aren't limits use original data
            data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{self.end}.gz"), autostrip=True,
                                 usecols=self.col_bands)
        else:  # Else, limit data
            data = self.lc_limiter(name, t=t, limits=limits)
        if isinstance(data[0], str):
            return data
        else:
            new_time = np.arange(data[:, 0][0], data[:, 0][-1], resolution * 1e-3)
            new_data = np.zeros((len(new_time), len(data[0])))  # Create new array to allocate data
            new_data[:, 0] = new_time
            for i in range(1, 6):
                new_data[:, i] = self.one_band_interpolate(data[:, 0], data[:, i], new_time, pack_num, kind, name=name)
            if plot:
                fig_i, ax_k = self.plot_any_grb(name, t=t, limits=limits, kind="Interpolate")  # Get original plot
                for i in range(5):
                    ax_k[i].plot(new_data[:, 0], new_data[:, i+1], linewidth=0.3, zorder=2.5, c='r')
                if save_fig:
                    path = self.results_path + r'/Interpolation_Images'
                    helpers.directory_maker(path)
                    fig_i.savefig(os.path.join(path, f"{name}_{self.end}.png"))
                    plt.close()
                return new_data, fig_i
            else:
                return new_data

    def so_much_interpolate(self, names, resolution=1, pack_num=10, kind='linear', t=None, limits=None, plot=True,
                            save_fig=True):
        """
        Function to faster interpolate all bands from several GRBs
        :param names: GRB Names array
        :param resolution: New resolution for data in ms, used to create new time array
        :param pack_num: The number of data grouped per packet to interpolate, a smaller number of points can improve
        the results, but a value too large or small can lead to poor interpolation results. Deprecated if kind='linear'
        :param kind: Specifies the kind of interpolation as a string or as an integer specifying the order
        of the spline interpolator to use. The string has to be one of ‘linear’, ‘nearest’, ‘nearest-up’, ‘zero’,
        ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’ (took from Scipy Docs).
        :param t: GRB Duration interval to be interpolated, can be 50, 90, 100 (default) or None (total light curve)
        :param limits: List of customized [t_start, t_end] to interpolate if needed (default is None)
        :param plot: Boolean to indicate if plot is necessary
        :param save_fig: Boolean to indicate if to save plot
        :return: Interpolated array for custom resolution in all GRB bands
        """
        errors = []  # Define errors
        non_errors = []  # Define non-errors array
        new_names = []  # Define new_names array
        m = len(names)
        if limits is None:  # If no limits are entered, then only send None array
            limits = repeat(None, m)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            result = list(tqdm(executor.map(self.one_grb_interpolate, names, repeat(resolution, m), repeat(pack_num, m)
                                            , repeat(kind, m), repeat(t, m), limits, repeat(plot, m),
                                            repeat(save_fig, m)), total=m, desc='LC Interpolating: ', unit='GRB'))
        assert len(result) == len(names), f"Expected equal number of names and results array, got: {len(names)} names" \
                                          f"and {len(result)} results"
        for i in range(len(result)):
            array = result[i]
            if isinstance(array[0], str):
                errors.append(array)
            else:
                non_errors.append(array)
                new_names.append(names[i])
        return non_errors, new_names, np.array(errors)

    def flux_calculator(self, name, band=1, t=None, limits=None):
        assert band in (1, 2, 3, 4), f"Expected band --> 1, 2, 3, 4. Got: {band}"
        # Unpack the downloaded data for the file, if it exists:
        if t is None and limits is None:
            data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{self.end}.gz"), autostrip=True,
                                 usecols=self.col_bands)
        else:
            data = self.lc_limiter(name, t=t, limits=limits)
        assert isinstance(data[0], (list, np.ndarray)), f"{name}--> Expected array, got: {data}"
        area = integrate.simpson(data[:, band], x=data[:, 0])  # Integral
        return area

    def hardness_proxy(self, names):
        assert isinstance(names, (str, np.ndarray, list, tuple)), f"Expected name or names array, got: {type(names)}"
        if isinstance(names, str):  # If a specific name is entered, then we search the value in the array
            band_50_100 = self.flux_calculator(names, band=3, t=100)
            band_25_50 = self.flux_calculator(names, band=2, t=100)
        else:  # If a name's array is specified, search them recursively
            with concurrent.futures.ProcessPoolExecutor(max_workers=self.workers) as executor:  # Parallelization
                band_50_100 = list(executor.map(self.flux_calculator, names, repeat(3, len(names)), repeat(100, len(names))))
                band_25_50 = list(executor.map(self.flux_calculator, names, repeat(2, len(names)), repeat(100, len(names))))
        return np.array(band_50_100)/np.array(band_25_50)

    @staticmethod
    def nearest_neighbors(name, total_names, coord, num=5, sorted_d=False):
        assert isinstance(num, (int, np.int)), f"Expected integer number of neighbors, got: {num}"
        assert len(total_names) == len(coord), f"Expected names and coordinates arrays with the same length, got: " \
                                               f"{len(total_names)}, {len(coord)}"
        row_name = np.where(np.isin(total_names, name))[0]  # Index row of match GRB
        distances = cdist(coord[row_name], coord)[0]
        sort_array = np.sort(distances)[1:num + 1]
        near_neighbors = np.where(np.isin(distances, sort_array))[0]
        if sorted_d:
            return np.sort(total_names[near_neighbors])
        else:
            return total_names[near_neighbors]
