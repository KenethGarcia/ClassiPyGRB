import os  # Import os to handle folders and files
import gzip  # Module to compress files
import requests  # Import the module needed to download from the website
import numpy as np  # Import numpy module to read tables, manage data, etc
import concurrent.futures  # Import module to do threading over the bursts
import matplotlib.pyplot as plt  # Import module to do figures, animations
from tqdm import tqdm  # Script to check progress bar in concurrency steps
from IPython import display  # Import module to display html animations
from scripts import helpers  # Script to do basics functions to data
from scipy import integrate  # Module to integrate using Simpson's rule
from tsne_animate import tsneAnimate  # Module to do Animations in tSNE
from openTSNE import TSNE as open_TSNE  # Alternative module to do tSNE
from sklearn.manifold import TSNE as sklearn_TSNE  # Module to do tSNE
from scipy.fft import next_fast_len, fft, fftfreq  # Function to look for the best suitable array size to do FFT


class SwiftGRBWorker:

    data_path = os.getcwd() + '\Data'  # The path to add Original and Pre-processed data
    original_data_path = data_path + '\Original_Data'  # Specific path to add Original Data
    results_path = os.getcwd() + '\Results'  # Specific path to add Results
    table_path = os.getcwd() + '\Tables'  # The path where tables are
    animations_path = os.getcwd() + '\Animations'  # The path where animations will be saved
    res = 64  # Resolution for the Light Curve Data in ms, could be 2, 8, 16, 64 (default), 256 and 1 (this last in s)

    def __init__(self):
        pass

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
        end = f"{self.res}ms" if self.res in (2, 8, 16, 64, 256) else f"1s"
        url = f"https://swift.gsfc.nasa.gov/results/batgrbcat/{name}/data_product/{i_d}-results/lc/{end}_lc_ascii.dat"
        try:  # Try to access to Swift Page
            r = requests.get(url, timeout=5)
            r.raise_for_status()  # We use this code to elevate the HTTP errors (i.e. wrong links) to exceptions
        # Then, if any exception has occurred, we send a message to indicate the error type:
        except requests.exceptions.RequestException as err:  # Notification message for all errors
            return name, err  # Return a tuple of GRB Name and error description
        else:
            file_name = f"{name}_{end}.gz"  # Define a unique file name for any name-resolution pair
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

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = tqdm(executor.map(self.download_data, names, t_ids),
                           total=len(names), desc='Downloading: ', unit='GRB')

            if error:
                with open(os.path.join(self.original_data_path, f"Errors_{self.res}.txt"), 'w') as f:
                    f.write("## GRB Name|Error Description\n")
                    for name, value in results:
                        f.write(f"{name}|{value}\n") if value is not None else None

    def names_durations(self, name=None, t=100, file_name="summary_burst_durations.txt"):
        """
        Function to extract the durations intervals for Swift Data (summary_burst_durations file is required in current
        working directory)
        :param name: Name of GRB if only one is needed, default is none
        :param t: Duration interval needed, default is 100 (can be 50, 90, 100)
        :param file_name: name of Summary burst durations, default is summary_burst_durations.txt
        :return: An array of (name, t_start, t_end) for all GRB needed (default is all GRBs)
        """
        columns = {50: (0, 7, 8), 90: (0, 5, 6), 100: (0, 3, 4)}  # Dictionary to extract the correct columns
        # Extract all values from summary_burst_durations.txt:
        path = os.path.join(self.table_path, file_name)
        durations = np.genfromtxt(path, delimiter="|", dtype=str, usecols=columns.get(t), autostrip=True)
        if name:  # If a specific name is entered, then we search the value in the array
            rows, columns = np.where(durations == name)
            return durations[rows]
        else:  # Else return all data from summary_burst_durations.txt
            return durations

    def lc_limiter(self, name, t=100, limits=None):
        """
        Function to extract GRB data out of (t_start, t_end), it is needed to specify these limits or the T_{i} duration
        :param name: GRB Name
        :param t: Duration interval needed, can be 50, 90, 100 (default)
        :param limits: List of customized [t_start, t_end] if needed (default is None)
        :return: A list with values into t_start and t_end, or None if data doesn't exist or has length < 4
        """
        try:
            if limits is None:  # If not have been defined any limits, then use T_{t} durations from file
                limit_interval = self.names_durations(name, t)  # Upload the values of t_start and t_durations
                t_start, t_end = float(limit_interval[0, 1]), float(limit_interval[0, 2])
            else:
                t_start, t_end, *other = limits
                t_start, t_end = float(t_start), float(t_end)
        # Unpack the downloaded data for the file, if it exists:
            end = f"{self.res}ms" if self.res in (2, 8, 16, 64, 256) else f"1s"
            data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{end}.gz"), autostrip=True)
            # Filter values between t_start and t_end in data:
            data = np.array([value for value in data if t_end >= value[0] >= t_start])
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
            limits = [None]*len(names)
        with concurrent.futures.ProcessPoolExecutor() as executor:  # Parallelization
            results = list(tqdm(executor.map(self.lc_limiter, names, [t]*len(names), limits), total=len(names),
                                desc='LC Limiting: ', unit='GRB'))
        if len(results) != len(names):  # Check if results and names have equal length
            print(f"Oops, something got wrong: There are {len(names)} GRBs but {len(results)} light curves limited")
            raise ValueError
        else:
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
        Function to normalize GRB light curve data
        :param data: Array with data in format time, (band, error) for 15-25, 25-50, 50-100, 100-350 and 15-350 keV
        :param print_area: Boolean to indicate if returns Total time-integrated flux
        :return: Array with values normalized using the 15-350 keV integral
        """
        data = np.array(data)
        area = integrate.simpson(data[:, 9], dx=self.res*1e-3 if self.res != 1 else 1)  # Integral for 15-350 KeV data
        data[:, 1: 11: 1] /= area  # Normalize the light curve
        if print_area:
            return data, area
        else:
            return data

    def so_much_normalize(self, more_data, print_area=False):
        """
        Function to faster normalize data using the lc_normalizer instance
        :param more_data: Array containing all individual data to be normalized
        :param print_area: Boolean to indicate if returns Total time-integrated flux
        :return:
        """
        with concurrent.futures.ProcessPoolExecutor() as executor:  # Parallelization
            results = list(tqdm(executor.map(self.lc_normalizer, more_data, [print_area]*len(more_data)),
                                total=len(more_data), desc='LC Normalizing: ', unit='GRB'))
        return results

    def zero_pad(self, data, length):
        """
        Function to zero pad array preserving order in time axis
        :param data: Data array to be zero-padded
        :param length: Length of the final array (need to be more than data length)
        :return: New array zero-padded, with added values of time basis for a given resolution
        """
        diff = length - len(data)  # Difference between actual and optimal array size
        data_plus_zeros = np.pad(data, ((0, diff), (0, 0)))  # Zero pad array
        initial_time_values = data_plus_zeros[:len(data), 0]  # Extract original time values
        dt = round(self.res*1e-3, 3) if self.res in (2, 8, 16, 64, 256) else 1  # Define step for new times
        add_time_values = np.arange(1, diff+1, 1)*dt + initial_time_values[-1]  # Set new time values
        data_plus_zeros[:, 0] = np.append(initial_time_values, add_time_values)  # Append new time values to original
        return data_plus_zeros

    def so_much_zero_pad(self, more_data):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            lengths = set(executor.map(len, more_data))
            max_length = next_fast_len(max(lengths))
            results = list(tqdm(executor.map(self.zero_pad, more_data, [max_length]*len(more_data)),
                                total=len(more_data), desc='LC Zero-Padding: ', unit='GRB'))
        return np.array(results, dtype=float)

    @staticmethod
    def fourier_concatenate(array, plot=False, name=None):
        """
        Function to do DFT to a GRB data from Swift, data need to be normalized before executed here
        :param array: GRB data array to be transformed, it would be sent in swift format (11 columns with time first)
        :param plot: Boolean value to indicate if plot is needed, default is False
        :param name: Name of GRB to save plot, default is None, it is needed if plot=True
        :return: An array with the Fourier Amplitude Spectrum of data
        """
        energy_data = np.delete(array, np.s_[::2], 1).transpose()  # Erase the time-error columns and transpose matrix
        concatenated = np.reshape(energy_data, len(energy_data)*len(energy_data[0]))  # Concatenate all columns of data
        sp = np.abs(fft(concatenated))  # Perform DFT to data and get the Fourier Amplitude Spectrum
        sp_one_side = sp[:len(sp)//2]  # Sampling below the Nyquist frequency
        if plot:
            t = array[:, 0]  # Take times from one band data
            freq = fftfreq(sp.size, d=t[1] - t[0])[:len(sp)//2]  # Get frequency spectrum basis values for one side
            dft_fig, axs1 = plt.subplots(2, 1, dpi=150, figsize=[10, 6.4])
            axs1[0].set_title(fr"{name} DFT", weight='bold').set_fontsize('12')
            axs1[0].plot(freq[:len(freq)//3], sp_one_side[:len(freq)//3], linewidth=0.1, c='k')
            axs1[0].set_xlim(left=-0.1, right=freq[len(freq)//3])
            axs1[1].plot(freq[len(freq) // 3:], sp_one_side[len(freq) // 3:], linewidth=0.1, c='k')
            axs1[1].set_xlim(left=freq[len(freq)//3], right=freq[-1])
            axs1[1].set_xlabel('Frequency (Hz)', weight='bold').set_fontsize('10')
            return sp_one_side, dft_fig
        else:
            return sp_one_side

    def so_much_fourier(self, arrays):
        """
         Function to do faster DFT to so much GRB data from Swift, data need to be normalized before executed here
         :param arrays: GRB list data array to be transformed, it would be sent in swift format (11 columns with time first)
         :return: An array with the DFT Amplitude of total data
         """
        with concurrent.futures.ProcessPoolExecutor() as executor:  # Parallelization
            sp_results = list(tqdm(executor.map(self.fourier_concatenate, arrays), total=len(arrays),
                                   desc='Performing DFT: ', unit='GRB'))
        return np.array(sp_results)

    def plot_any_grb(self, name, t=100):
        """
        Function to plot any GRB out of i-esim duration (can be 50, 90, 100), default is T_90
        :param name: GRB name
        :param t: Duration interval needed, default is 100 (can be 50, 90, 100, else plot all light curve)
        :return: Returns the figure, axis created
        """
        fig5 = plt.figure(dpi=150)
        gs = fig5.add_gridspec(nrows=5, hspace=0)
        axs = gs.subplots(sharex=True)
        axs[4].set_xlabel('Time since BAT Trigger time (s)', weight='bold').set_fontsize('10')
        end = f"{self.res}ms" if self.res in (2, 8, 16, 64, 256) else f"1s"
        bands = (r"$15-25\,keV$", r"$25-50\,keV$", r"$50-100\,keV$", r"$100-350\,keV$", r"$15-350\,keV$")
        colors = ('k', 'r', 'lime', 'b', 'tab:pink')
        if t in (50, 90, 100):  # Limit Light Curve if is needed
            data = self.lc_limiter(name=name, t=t)
            axs[0].set_title(fr"{end} Swift {name} out of $T\_{t}$", weight='bold').set_fontsize('12')
            if isinstance(data[0], str):  # If an error occurs when limit out of T_{t}
                print(f"Error when limiting {name} Light Curve out of t={t}, plotting all data")
                data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{end}.gz"), autostrip=True)
                axs[0].set_title(f"{end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        else:  # If not needed, only extract total data
            data = np.genfromtxt(os.path.join(self.original_data_path, f"{name}_{end}.gz"), autostrip=True)
            axs[0].set_title(f"{end} Swift {name} Total Light Curve", weight='bold').set_fontsize('12')
        axs[4].set_xlim(left=data[0, 0], right=data[-1, 0])
        for i in range(5):
            axs[i].plot(data[:, 0], data[:, 2*i+1], label=bands[i], linewidth=0.5, c=colors[i])
            axs[i].legend(fontsize='xx-small', loc="upper right")
        return fig5, axs

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
    def tsne_animation(data, durations_data, pp=30, lr='auto'):
        """
        Function to see convergence in t-SNE algorithm
        :param data: Array of data GRB features
        :param durations_data: Array of classification of GRBs in the sample (T_90 separation recommended)
        :param pp: Perplexity to evaluate t-SNE
        :param lr: Learning Rate to evaluate t-SNE
        :return: t-SNE convergence Animation, same as FuncAnimation instance of Matplotlib
        """
        tsne = tsneAnimate(sklearn_TSNE(n_components=2, perplexity=pp, n_jobs=-1, learning_rate=lr, init='random'))
        color_values = np.array([0 if value < 2 else 1 for value in durations_data])
        anim = tsne.animate(data, color_values, useTqdm=1)
        video = anim.to_html5_video()  # Converting to a html5 video
        html = display.HTML(video)  # Embedding for the video
        display.display(html)  # Draw the animation
        plt.close()
        anim.save(f"animation_tSNE.mp4")
        return anim
