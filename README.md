![Logo](docs/Animations/images/logo.jpeg)  

**ClassiPyGRB** is a Python 3 package to download, process, visualize and classify Gamma-Ray-Bursts (GRBs) from the Swift/BAT Telescope (https://swift.gsfc.nasa.gov/about_swift/bat_desc.html) database. It is distributed over the GNU General Public License Version 2 (1991). Please, read the complete description of the method and its application to GRBs in this [publication](JOSS_Docs/paper.md).

# Statement of need
The Swift Burst Alert Telescope (BAT) is a wide-field, hard Gamma-ray detector that, since its launch in 2004, has played an important role in the high-energy astrophysical field. A preliminary query on the Astrophysics Data System (ADS) with the keyword "Swift/BAT" indicates that over 7000 research works have been uploaded to its platform (up to December 2023). Furthermore, recently has been increased the number of studies per year, evidencing the relevance and impact of the use of the Swift/BAT database.

Although the Swift/BAT database is [publicly available](https://swift.gsfc.nasa.gov/results/batgrbcat/), for first-time users it might be a challenge to download and process the data. The data is stored in multiple directories, depending on the GRB. Moreover, the data is not pre-processed, and the user must perform the data manipulation and interpolation by themselves. These issues make the data processing a time-consuming task. **ClassiPyGRB** is a Python 3 package that aims to solve these problems by providing a simple and intuitive interface to download, process, visualize, and classify the photometric data of GRBs from the Swift/BAT database.

**ClassiPyGRB** allows researchers to query light curves for any GRB in the Swift/BAT database simply and intuitively. The package also provides a set of tools to preprocess the data, including noise/duration reduction and interpolation. Moreover, The package also provides a set of facilities and tutorials to classify GRBs based on their light curves, following [Jespersen et al.(2020)](https://doi.org/10.3847/2041-8213/ab964d) and [Garcia-Cifuentes et al.(2023)](https://doi.org/10.3847/1538-4357/acd176). This method is based on a dimensionality reduction of the data using t-Distributed Stochastic Neighbour Embedding (TSNE), where the user can visualize the results using a Graphical User Interface (GUI). The user can also plot and animate the results of the TSNE analysis, allowing to perform a deeper hyperparameter grid search. The package is distributed over the GNU General Public Licence Version 2 (1991).

# Attribution
If you use this code in a publication, please refer to the package by its name and cite [Garcia-Cifuentes et al.(2023)](https://doi.org/10.3847/1538-4357/acd176) -> [Astrophysical Journal Vol. 951 No. 1](https://doi.org/10.3847/1538-4357/acd176). Any question, please email [Keneth Garcia-Cifuentes](mailto:kenet.garcia@correo.nucleares.unam.mx).

## Dependencies
This repository requires Python 3.8 or high, and a list of packages downloaded automatically ([numpy](https://github.com/numpy/numpy), [scikit-learn](https://scikit-learn.org/stable/index.html), etc). In addition, it is required to install all the dependencies related to Tkinter, Pillow, and ImageTK. In Debian-based distros you can install these packages by running the following commands:

```
$ sudo apt-get install python3-tk
$ sudo apt-get install python3-pil python3-pil.imagetk
```

Other data management packages as [Numpy](https://numpy.org/) or [Pandas](https://pandas.pydata.org/) will be required in Documentation.

## Installation
The latest sources from **ClassiPyGRB** are avaiable by cloning the repository:
```
$ git clone https://github.com/KenethGarcia/ClassiPyGRB
$ cd ClassiPyGRB
$ pip install .
```
Or, using `pip`:
```
$ pip install ClassiPyGRB@git+https://github.com/KenethGarcia/ClassiPyGRB
```
or by using the stable [PyPI](https://pypi.org/) compiled version:
```
$ pip install ClassiPyGRB
```

## Features

In **ClassiPyGRB**, it is possible to retrieve data from the Swift/BAT catalog by a three-line code:
```
from ClassiPyGRB import SWIFT
swift = SWIFT(res=64)
df = swift.obtain_data(name='GRB211211A')
```
Moreover, you can plot a light curve using one single line:
```
swift.plot_any_grb(name='GRB060614')
```
You can do specialized tasks to see the convergence of t-Distributed Stochastic Neighbor Embedding (TSNE):

![convergence](docs/Animations/animation1.gif)

or use a Graphical User Interface (GUI) to analyze the embeddings obtained by TSNE:

![GUI](docs/Animations/images/Use.png)

We strongly encourage you to read the Documentation of **ClassiPyGRB** before start. This documentation includes all the details and follow-up for managing and processing data from Swift/BAT, performing TSNE, plotting and animating their results, and how to use the internal GUI.
Moreover, we developed intuitive notebooks to support you in your research.

- 1. [Basic Usage](docs/1.Basic_Usage.ipynb)
		
- 2. [BAT: Data_Download](docs/2.BAT_Data_Download.ipynb)
	
- 3. [BAT: Preprocess](docs/3.BAT_Preprocess.ipynb)
	
- 4. [BAT: Noise_Reduction](docs/4.BAT_Noise_Reduction.ipynb)
	
- 5. [BAT: Interpolation](docs/5.BAT_Interpolate.ipynb)
	
- 6. [TSNE: Introduction](docs/6.TSNE_Introduction.ipynb)
	
- 7. [TSNE: Overview](docs/7.TSNE_Overview.ipynb)
	
- 8. [Plotting with t-SNE](docs/8.TSNE_Plotting.ipynb)
	
- 9. [Clustering Properties](docs/9.Cluster_Properties.ipynb)

- 10. [Applications and Example](docs/10.Extended_Emission.ipynb)

- 11. [Internal GUI](docs/11.Viewer_Instance.ipynb)

# Enhancement and Support

**ClassiPyGRB** is a open-source package where all kinds of contributions are welcome. Whether you want to report a bug or submit a pull request, your feedback, comments and suggestions will be very welcome.

Here are some ways you can get involved in this project:
- Report a bug or issue on our [GitHub Issues](https://github.com/KenethGarcia/ClassiPyGRB/issues) page.
- Suggest a new feature or improvement by opening a new issue.
- Submit a [pull request](https://github.com/KenethGarcia/ClassiPyGRB/pulls) with your code changes or enhancements.
- Share ClassiPyGRB on social media or with your colleagues.

We appreciate your interest in this package. Please, do not hesitate to email [Keneth Garcia](mailto:keneth.garcia@correo.nucleares.unam.mx) to discuss any topic related to **ClassiPyGRB**.
