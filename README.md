# **Warning: This package is currently in development. Take care about the instances presented here**

![Logo](docs/Animations/images/Logo.jpeg)  

**ClassiPyGRB** is a Python 3 package to retrieve, download, visualize, process and classify Gamma-Ray-Burst (GRB) data from [Swift/BAT Telescope](https://swift.gsfc.nasa.gov/about_swift/bat_desc.html). It is distributed over the GNU General Public License Version 2 (1991).
- - -
# How to cite
If you used ClassiPyGRB as a part of your research work. We appreciate to refer this package by its name and cite [Garcia-Cifuentes et al.(2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv230408666G/abstract) -> [arXiv link](https://arxiv.org/abs/2304.08666).

# Introduction
Gamma-ray bursts (GRBs) are typically classified in two groups using their duration ([Kouveliotou et al. (1993)](https://ui.adsabs.harvard.edu/abs/1993ApJ...413L.101K/abstract)). Nevertheless, this classification is not exact. And in recent years, it has been observed that it is not reliable in determining the progenitors of such events.

# T-distributed Stochastic Neighbor Embedding (t-SNE) in Swift Data

[Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract) showed that Swift/BAT GRBs can be separated into two groups  when t-SNE is performed. In this repository, we replicate this work by adding more recent data and an in-depth analysis of t-SNE performance. Moreover, we add synthetic GRBs performed using Machine Learning instances and join into Swift and other GRB data packages.

## Dependencies
This repository requires Python 3.8 or high, and the packages from the [``requeriments.txt``](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/requirements.txt) file. Other packages will be required optionally in Documentation (i.e., Jupyter).


## What can you do with this repository?
1. **Download Swift Data for different binnings.**

![](https://github.com/KenethGarcia/GRB_ML/blob/4f5322be0ab14f37b968f98ba4400a52e0aa5eed/Documentation/README_Images/GRB060614.jpg)

2. **Pre-process Swift Data following the guidelines from [Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract)**

![](https://github.com/KenethGarcia/GRB_ML/blob/4f5322be0ab14f37b968f98ba4400a52e0aa5eed/Documentation/README_Images/Limited_GRB060614.jpg)

3. **Perform t-SNE in high-level customization (thanks to [_openTSNE_](https://opentsne.readthedocs.io/en/latest/index.html) and [_scikit Learn_](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) implementations):**
- **Create creative t-SNE animations showing its convergence:**

![](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/Documentation/Animations/convergence_animation_pp_30.gif)

- **Or changing any hyperparameter (perplexity, learning rate, etc.):**

![](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/Documentation/Animations/perplexity_animation_2.gif)

4. **Reduce noise to any custom light curve using [FABADA](https://github.com/PabloMSanAla/fabada), checking its variance assuming a gray-scaled 1-pixel height image.**

![](https://github.com/KenethGarcia/GRB_ML/blob/4f5322be0ab14f37b968f98ba4400a52e0aa5eed/Documentation/README_Images/Noise_Filtered_GRB060614.jpg)

5. **Interpolate between light curve data points using any custom n-order polynomial.**

![](https://github.com/KenethGarcia/GRB_ML/blob/4f5322be0ab14f37b968f98ba4400a52e0aa5eed/Documentation/README_Images/Interpolated_GRB060614.jpg)

## Citation
If you use this repository in a scientific publication, we would appreciate citations: 

**_FUTURE CITATION_**

---
