


# **Warning: This repository is under development**

![Logo](docs/Animations/images/logo.jpeg)  

**ClassiPyGRB** is a Python 3 package to download, process, visualize and classify Gamma-Ray-Bursts (GRB) from the [Swift/BAT Telescope] database (https://swift.gsfc.nasa.gov/about_swift/bat_desc.html). It is distributed over the GNU General Public License Version 2 (1991). Please read the complete description of the method and its application to GRBs in this [publication](JOSS_Docs/paper.md)

[Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract) showed that Swift/BAT GRBs can cluster into two groups when t-SNE is performed. In this repository, we replicate this work by adding more recent data from the Swift/BAT catalog (up to July 2022). We also included a noise-reduction and  an interpolation tools for achieving a deeper analysis of these data.

# Attribution
If you use this code in a publication, please refer to the package by name and cite [Garcia-Cifuentes et al.(2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv230408666G/abstract) -> [arXiv link](https://arxiv.org/abs/2304.08666). Any question, please email [Keneth Garcia-Cifuentes](mailto:kenet.garcia@correo.nucleares.unam.mx)

## Dependencies
This repository requires Python 3.8 or high, and the packages from the [``requeriments.txt``](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/requirements.txt) file. Other packages will be required optionally in Documentation (i.e., Jupyter).


## Features

- 1. [Basic Usage](docs/Basic_Usage.ipynb)
		
- 2. [BAT: Data_Download](docs/BAT_Data_Download.ipynb)
	
- 3. [BAT: Preprocess](docs/BAT_Preprocess.ipynb)
	
- 4. [BAT: Noise_Reduction](docs/BAT_Noise_Reduction.ipynb)
	
- 5. [BAT: Interpolation](docs/BAT_Interpolate.ipynb)
	
- 6. [TSNE: Introduction](docs/TSNE_Introduction.ipynb)
	
- 7. [TSNE: Overview](docs/TSNE_Overview.ipynb)
	
- 8. [Plotting with t-SNE](docs/TSNE_Plotting.ipynb)
	
- 9. [Clustering Properties](docs/Cluster_Properties.ipynb)

- 10. [Applicatios and Example](docs/Extended_Emission.ipynb)

# Contributors:
1. [Keneth Garcia-Cifuentes](https://orcid.org/0009-0001-2607-6359)
2. [Rosa L. Becerra](https://orcid.org/0000-0002-0216-3415)
3. [Fabio De Colle](https://orcid.org/0000-0002-3137-4633)
