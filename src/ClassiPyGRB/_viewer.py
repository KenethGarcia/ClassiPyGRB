# Author: Garcia-Cifuentes, K. <ORCID:0009-0001-2607-6359>
# Author: Becerra, R. L. <ORCID:0000-0002-0216-3415>
# Author: De Colle, F. <ORCID:0000-0002-3137-4633>
# License: GNU General Public License version 2 (1991)

# This file contains the Tkinter GUI for ClassiPyGRB.
# Details about Swift Data can be found in https://swift.gsfc.nasa.gov/about_swift/bat_desc.html

import os
import inspect
import numpy as np
import tkinter as tk
import matplotlib
import matplotlib.pyplot as plt
from ClassiPyGRB import SWIFT
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

dpi = 150
matplotlib.use("TkAgg")
swift_args = set(inspect.signature(SWIFT).parameters) - {'self', 'res', 'n_bands'}

# Review of Swift/BAT data, Table 3 of https://ui.adsabs.harvard.edu/abs/2016ApJ...829....7L/abstract
EE_Lien16 = ['GRB150424A', 'GRB111121A', 'GRB090916', 'GRB090715A', 'GRB090531B', 'GRB080503', 'GRB071227',
             'GRB070714B', 'GRB061210', 'GRB061006', 'GRB051227', 'GRB050724']

# Search of EE GRBs of BAT and GBM: https://ui.adsabs.harvard.edu/abs/2015MNRAS.452..824K/abstract
EE_GRBs_Kaneko2015 = ["GRB050724", "GRB060614", "GRB061006", "GRB061210", "GRB070506", "GRB070714B", "GRB080503",
                      "GRB090531B", "GRB090927", "GRB100212A", "GRB100522A", "GRB110207A", "GRB110402A", "GRB111121A",
                      "GRB121014A", "GRB051016B"]

# Only at high redshift, extracted from Table 1: https://ui.adsabs.harvard.edu/abs/2021ApJ...911L..28D/abstract
EE_GRBs_Dichiara2021 = ["GRB051210", "GRB120804A", "GRB160410A", "GRB181123B", "GRB060121"]

# Extensive search of EE GRBs, extracted from Table 1: https://ui.adsabs.harvard.edu/abs/2021ApJ...914L..40D/abstract
EE_GRBs_Dainotti2021 = ["GRB050724A", "GRB050911", "GRB051016B", "GRB051227", "GRB060306", "GRB060607A", "GRB060614",
                        "GRB060814", "GRB060912A", "GRB061006", "GRB061021", "GRB061210", "GRB070223", "GRB070506",
                        "GRB070714B", "GRB080603B", "GRB080913", "GRB090530", "GRB090927", "GRB100704A", "GRB100814A",
                        "GRB100816A", "GRB100906A", "GRB111005A", "GRB111228A", "GRB150424A", "GRB160410A"]


class Viewer(tk.Tk):
    """
    Class for the Scatter Plot Window of ClassiPyGRB.
    """

    def __init__(self, **kwargs):
        """
        Constructor of the ScatterPlotWindow class.

        Args:
            **kwargs: Additional parameters for the Tkinter window and SWIFT class from ClassiPyGRB.
        """
        self.sc_kwargs = {} if kwargs is None else kwargs
        kwargs.pop('res', None)  # Remove unnecessary parameters
        kwargs.pop('n_bands', None)
        self.swift_dict = {k: kwargs.pop(k) for k in dict(kwargs) if k in swift_args}  # Separate parameters
        tk.Tk.__init__(self, **kwargs)
        self.main_plot_bar = None
        self.title("ClassiPyGRB")  # Make a title
        self.grb_names, self.feat, self.positions, self.durations, self.EE_label = ['0'], None, None, None, None
        self.rect, self.old_num_bands = Rectangle((0, 0), 5, 5, facecolor='none', edgecolor='none'), None
        self.old_perplexity, self.old_learning_rate, self.old_ee = None, None, None
        self.num_bands, self.old_dataset, self.special_label = [1, 2, 3, 4, 5], None, None
        # Create a dictionary to store the names of each point
        self.names = {}

        # Make a button to perform t-SNE in the dataset selected
        self.button_data = tk.Button(self, text="Update t-SNE", command=self.change_dataset)
        self.button_data.pack(side='top')

        # Create an Entry and button to change t-SNE parameters
        self.pp_label = tk.Label(self, text='Perplexity:')
        font_style(self.pp_label)
        self.pp_label.place(x=30, y=3)
        self.perplexity = tk.Entry(self, width=4)
        self.perplexity.insert(0, '5')
        self.perplexity.place(x=100, y=0)
        self.lr_label = tk.Label(self, text='Learning Rate:')
        font_style(self.lr_label)
        self.lr_label.place(x=140, y=3)
        self.learning_rate = tk.Entry(self, width=4)
        self.learning_rate.insert(0, '200')
        self.learning_rate.place(x=235, y=0)
        self.ee_label = tk.Label(self, text='Early Exaggeration:')
        font_style(self.ee_label)
        self.ee_label.place(x=275, y=3)
        self.ee = tk.Entry(self, width=4)
        self.ee.insert(0, '12')
        self.ee.place(x=400, y=0)

        self.num_bands_label = tk.Label(self, text='TSNE Bands (keV):')
        font_style(self.num_bands_label)
        self.num_bands_label.place(x=0, y=35)
        self.band_var = [tk.BooleanVar() for _ in range(len(self.num_bands))]
        [self.band_var[i].set(True) for i in range(len(self.num_bands))]
        self.band_boxes = []
        for i, band in enumerate([' 15-25', ' 25-50', '50-100', '100-350', '15-350']):
            self.band_boxes.append(
                tk.Checkbutton(self, text=band, variable=self.band_var[i], onvalue=True, offvalue=False))
            self.band_boxes[i].place(x=115 + 90 * i, y=35)

        self.dft_var = tk.BooleanVar()
        self.dft_var.set(True)
        self.dft_box = tk.Checkbutton(self, text='Plot DFT', variable=self.dft_var, onvalue=True, offvalue=False)
        self.dft_box.place(x=1170, y=0)

        self.t_label = tk.Label(self, text='Duration:')
        font_style(self.t_label)
        self.t_label.place(x=840, y=0)
        self.t = tk.StringVar()
        self.t.set('90')
        self.t_drop_menu = tk.OptionMenu(self, self.t, 'Full', '50', '90', '100')
        self.t_drop_menu.place(x=900, y=0)
        self.button_data = tk.Button(self, text="Update Light Curve", command=self.update_lc)
        self.button_data.pack(side='top')
        self.GRB_special = tk.Entry(self, width=15)
        self.GRB_special.insert(0, 'GRB061006')
        self.GRB_special.place(x=970, y=3)
        self.search_GRB_button = tk.Button(self, text="Search GRB", command=self.position_searcher)
        self.search_GRB_button.place(x=1070, y=0)

        # Configure the objects to use
        self.object1 = SWIFT(res=64, n_bands=self.num_bands, **self.swift_dict)
        self.object2 = SWIFT(res=10000, n_bands=self.num_bands, **self.swift_dict)

        # Add a variable to change between datasets
        self.dataset = tk.StringVar()
        self.dataset.set('64ms')
        # Make a Drop Menu to select between dataset options
        self.button_label = tk.Label(self, text="Dataset:")
        font_style(self.button_label)
        self.button_label.place(x=440, y=3)
        self.drop = tk.OptionMenu(self, self.dataset, '64ms', 'FABADA', '10s')
        self.drop.place(x=500, y=0)

        # Create the scatter plot (Figure 1)
        self.figure1, self.ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 7))
        divider = make_axes_locatable(self.ax1)
        self.cax = divider.append_axes('right', size='5%', pad=0.05)
        self.ax1.add_patch(self.rect)
        self.canvas1 = FigureCanvasTkAgg(self.figure1, self)
        self.canvas1.get_tk_widget().pack(side="left", fill="both", expand=True)

        # Create a line plot (Figure 2)
        self.gs_kw = dict(width_ratios=[1], height_ratios=[1, 1, 1, 1, 1], hspace=0)
        self.figure2, self.ax2 = plt.subplots(nrows=5, ncols=1, sharex=True, gridspec_kw=self.gs_kw, figsize=(6, 4))

        self.canvas2 = FigureCanvasTkAgg(self.figure2, self)
        self.canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)

        # Connect the canvas to the mouse event callback
        self.canvas1.mpl_connect("pick_event", self.on_pick)

        # Connect the zoom function to the canvas
        self.canvas1.mpl_connect("scroll_event", self.on_scroll)

    def change_dataset(self):
        """
        Function to reset the t-SNE plot.
        """

        def read(
                key: str,
                n_bands: list,
        ):
            """
            Function to read the data from the npz files saved.

            Args:
                key (str): Kind of dataset related to the saved file.
                n_bands (list): List of bands related to the saved file.

            Returns:
                A tuple with the GRB names and Fourier Amplitude Spectrum of the saved file.
            """
            if key == '64ms':
                data_64ms = np.load(os.path.join(self.object1.results_path,
                                                 f'SWIFT_data_{self.object1.res}res_{"".join(str(b) for b in n_bands)}'
                                                 f'bands_DFT_True.npz'))
                names_64ms, features_64ms = data_64ms['names'], data_64ms['data']
                return names_64ms, features_64ms
            elif key == 'FABADA':
                data_fabada = np.load(os.path.join(self.object1.results_path,
                                                   f'SWIFT_Noise_Reduced_data_{self.object1.res}res_'
                                                   f'{"".join(str(b) for b in n_bands)}bands_DFT_True.npz'))
                names_fabada, features_fabada = data_fabada['names'], data_fabada['data']
                return names_fabada, features_fabada
            elif key == '10s':  # CHECK THE NAME
                data_10s = np.load(os.path.join(self.object2.results_path,
                                                f'SWIFT_data_{self.object2.res}res_{"".join(str(b) for b in n_bands)}'
                                                f'bands_DFT_True_Interpolated_at_{30}ms.npz'))
                names_10s, features_10s = data_10s['names'], data_10s['data']
                return names_10s, features_10s

        # Select dataset based in Drop Menu
        self.num_bands = [i + 1 for i, b in enumerate(self.band_var) if b.get()]
        # Re-configure the objects to use
        self.object1 = SWIFT(res=64, n_bands=self.num_bands, **self.swift_dict)
        self.object2 = SWIFT(res=10000, n_bands=self.num_bands, **self.swift_dict)
        flag = True
        if not ((self.feat is not None) and (self.num_bands == self.old_num_bands) and
                (self.dataset.get() == self.old_dataset)):
            self.grb_names, self.feat = read(n_bands=self.num_bands, key=self.dataset.get())
            flag = False
        if self.dataset.get() != self.old_dataset:
            # Index durations
            self.durations = self.object1.total_durations(self.grb_names)
            # Assign a unique label to each point
            self.names.clear()
            for i in range(len(self.grb_names)):
                self.names[i] = self.grb_names[i]
        # Perform t-SNE
        lr = float(self.learning_rate.get()) if self.learning_rate.get() != 'auto' else 'auto'
        pp, ee = float(self.perplexity.get()), float(self.ee.get())
        if not(flag and (self.old_perplexity == pp) and (self.old_learning_rate == lr) and (self.old_ee == ee)):
            self.positions = self.object1.perform_tsne(self.feat, perplexity=pp, learning_rate=lr, early_exaggeration=ee)
        # Re-make main plot
        self.ax1.clear()
        self.ax1, self.main_plot_bar = self.object1.plot_tsne(self.positions, ax=self.ax1, names=self.grb_names,
                                                              durations=self.durations, return_colorbar=True,
                                                              color_bar_kwargs={'shrink': 0.6, 'cax': self.cax})
        self.canvas1.draw()
        self.old_num_bands, self.old_dataset = self.num_bands, self.dataset.get()
        self.old_perplexity, self.old_learning_rate, self.old_ee = float(self.perplexity.get()), lr, float(self.ee.get())

    def on_pick(self, event):
        """
        Function to handle the pick event of the canvas.

        Args:
            event (PickEvent): Pick event of the canvas.
        """
        flag = False
        point = event.artist
        index = event.ind[0]
        global name
        name = self.names[index]
        # delete all the labels and thicks of ax2
        try:
            [self.ax2[i].clear() for i in range(len(self.ax2))]
        except TypeError:
            pass
        if not self.dft_var.get():
            if len(self.ax2) != 5:
                flag = True
                self.canvas2.get_tk_widget().destroy()
                self.figure2, self.ax2 = plt.subplots(nrows=5, ncols=1, sharex=True, gridspec_kw=self.gs_kw, figsize=(6, 4))
            try:
                t = int(self.t.get())
            except ValueError:
                t = None
            if self.dataset.get() == '64ms':
                self.object1.plot_any_grb(name=name, t=t, ax=self.ax2, check_disk=True)
            elif self.dataset.get() == 'FABADA':
                old_path = self.object1.original_data_path  # Change path
                self.object1.original_data_path = self.object1.noise_data_path
                self.object1.plot_any_grb(name=name, t=t, ax=self.ax2, check_disk=True)
                self.object1.original_data_path = old_path  # Return path
            else:
                self.object2.plot_any_grb(name=name, t=t, ax=self.ax2, check_disk=True)
            if flag:
                self.canvas2 = FigureCanvasTkAgg(self.figure2, self)
                self.canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)
        else:
            match = np.where(np.isin(self.grb_names, name))[0]
            feat = self.feat[match]
            if len(self.ax2) != 2:
                flag = True
                self.canvas2.get_tk_widget().destroy()
                self.figure2, self.ax2 = plt.subplots(nrows=2, ncols=1, figsize=(6, 4))
            self.object1.dft_plot(spectrum=feat[0], ax=self.ax2, name=name)
            self.figure2.tight_layout()
            if flag:
                self.canvas2 = FigureCanvasTkAgg(self.figure2, self)
                self.canvas2.get_tk_widget().pack(side="right", fill="both", expand=True)
        self.canvas2.draw()
        text = f'{name} is EE GRB From:'
        if name in EE_Lien16:
            text += ' Lien16 '
        elif name in EE_GRBs_Kaneko2015:
            text += ' Kaneko15 '
        elif name in EE_GRBs_Dainotti2021:
            text += ' Dainotti21 '
        elif name in EE_GRBs_Dichiara2021:
            text += ' Dichiara21 '
        else:
            text = f'{name} is a Non Previous EE GRB'
        # Clean the label if it exists
        try:
            self.EE_label.destroy()
            self.special_label.destroy()
        except AttributeError:
            pass
        self.EE_label = tk.Label(self, text=text)
        self.EE_label.place(x=900, y=35)

    def on_scroll(self, event):
        """
        Function to handle the scroll event of the canvas.

        Args:
            event (ScrollEvent): Scroll event of the canvas.

        Returns:

        """
        # Get the current x and y limits of the axis
        xlim = self.ax1.get_xlim()
        ylim = self.ax1.get_ylim()

        # Get the mouse position
        x, y = event.xdata, event.ydata

        # Calculate the new center of the plot
        xc = x + (x - xlim[0]) * (1 - event.step / 10)
        yc = y + (y - ylim[0]) * (1 - event.step / 10)

        # Calculate the new x and y limits
        xlim = (xc - (xc - xlim[0]) / (1 - event.step / 10), xc + (xlim[1] - xc) / (1 - event.step / 10))
        ylim = (yc - (yc - ylim[0]) / (1 - event.step / 10), yc + (ylim[1] - yc) / (1 - event.step / 10))

        # Update the axis limits
        self.ax1.set_xlim(xlim)
        self.ax1.set_ylim(ylim)

        # Redraw the canvas
        self.canvas1.draw()
        self.canvas1.flush_events()

    def update_lc(self):
        """
        Function to update the light curve plot.
        """
        [self.ax2[i].clear() for i in range(len(self.ax2))]
        try:
            t = int(self.t.get())
        except ValueError:
            t = None
        if self.dataset.get() == '64ms':
            self.object1.plot_any_grb(name=name, t=t, ax=self.ax2, check_disk=True)
        elif self.dataset.get() == 'FABADA':
            old_path = self.object1.original_data_path  # Change path
            self.object1.original_data_path = self.object1.noise_data_path
            self.object1.plot_any_grb(name=name, t=t, ax=self.ax2, check_disk=True)
            self.object1.original_data_path = old_path  # Return path
        else:
            self.object2.plot_any_grb(name=name, t=t, ax=self.ax2, check_disk=True)
        self.canvas2.draw()

    def position_searcher(self):
        """
        Function to search the position of a GRB in the t-SNE plot and highlight it in the Tkinter canvas.
        """
        try:
            self.rect.remove()
        except ValueError:
            pass
        special_grb = self.GRB_special.get()
        match = np.where(np.isin(self.grb_names, special_grb))[0]
        position = self.positions[match]
        text = f'Search GRB in {np.round(position[0], 1)}' if len(position) != 0 else 'GRB not found...'
        if len(position) != 0:
            try:
                self.main_plot_bar.remove()
            except KeyError:
                pass
            colors = np.array(['lightgray'] * len(self.grb_names))
            colors[match] = 'magenta'
            alphas = np.array([0.5] * len(self.grb_names))
            alphas[match] = 1
            self.ax1.clear()
            self.ax1 = self.object1.plot_tsne(self.positions, ax=self.ax1, non_special_marker_color=colors,
                                              kwargs_plot={'alpha': alphas})
        self.canvas1.draw()
        try:
            self.EE_label.destroy()
            self.special_label.destroy()
        except AttributeError:
            pass
        special_label = tk.Label(self, text=text)
        special_label.place(x=950, y=35)


def font_style(label):
    """
    Function to change the font style of a label.

    Args:
        label: Tkinter label to change the font style
    """
    label.config(font=('Helvetica bold', 9))


if __name__ == "__main__":
    app = Viewer(root_path='/home/keneth/Documents/Swift_Data')
    app.mainloop()
