
![Logo](/docs/Animations/images/logo.jpeg) 

---
Title: 'ClassipyGRB: Classifying GRBs with Machine Learning t-SNE'
Tags:
  - Python
  - astronomy
  - transient astronomy
  - gamma-ray bursts
  - extragalactic astronomy
Authors:
  - name: [Keneth Garcia-Cifuentes](https://orcid.org/0009-0001-2607-6359)
    Equal-contrib: true
    Affiliation: 1
    Corresponding: True

  - name: [Rosa L. Becerra](https://orcid.org/0000-0002-0216-3415)
    Affiliation: 1

  - name: [Fabio De Colle](https://orcid.org/0000-0002-3137-4633)
    Affiliation: 1
    
Affiliation:
 - Name: Instituto de Ciencias Nucleares,  Universidad Nacional Autónoma de México, Apartado Postal 70-543, 04510 CDMX, México
 - Index: 1

date: 28 April 2023 

# **Warning: This package is under peer review**

**ClassiPyGRB** is a Python 3 package to download, process, visualize and classify Gamma-Ray-Burst (GRB) datavbase from [Swift/BAT Telescope](https://swift.gsfc.nasa.gov/about_swift/bat_desc.html). It is distributed over the GNU General Public License Version 2 (1991).
- - -

# Dependencies
This repository requires Python 3.8 or high, and the packages from the [``requeriments.txt``](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/requirements.txt) file. Other packages will be required optionally in Documentation (i.e., Jupyter).

# Introduction
Gamma-ray bursts (GRBs) are originated by the death of a massive star or from the merger of two compact objects. The classification is based on their duration ([Kouveliotou et al. (1993)](https://ui.adsabs.harvard.edu/abs/1993ApJ...413L.101K/abstract)). Nevertheless, events such as GRB 211211A ([Yang et al. (2022)](https://ui.adsabs.harvard.edu/abs/2022Natur.612..232Y/abstract)) has challenged this taxonomy.  Therefore, this method is not completely reliable for the determination of the progenitors of such events.

# Statement of need

The detection of GRBs are carried by spatial mission such as Swift with its BAT instrument. Using the Swift-BAT GRB catalog, consisting of light curves (flux/time) in four energy bands (15-25 keV, 25-50 keV, 50-100 keV, 100-350 keV) for about 1250 events up to July 2022.

`ClassipyGRB` was designed to be used by astronomers whose scientific research is focused on GRBs. This module provides interactive visualizations of the light curves of GRBs, and the similarities they share. `ClassipyGRB` allows the comparison in a few seconds with other events in order to find resembling GRBs through the proximity between them.


# T-distributed Stochastic Neighbor Embedding (t-SNE) in Swift Data

[Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract) showed that Swift/BAT GRBs can cluster into two groups when t-SNE is performed. In this repository, we replicate this work by adding more recent data from the Swift/BAT catalog (up to July 2022). We also included a noise-reduction and  an interpolation tools for achieving a deeper analysis of these data.

# Applications 

This method has been successfully applied for the correct identification of candidates for GRBs with extended emission that have not been previously identified by other groups, saving a lot of time and human effort [Garcia-Cifuentes et al.(2023)](https://ui.adsabs.harvard.edu/abs/2023arXiv230408666G/abstract). We locate extended emission GRBs reported by various authors under different criteria in our t-SNE maps and discuss its possible identification using only the gamma-ray data provided by the automatic pipeline of Swift/BAT. 

![t-SNE visualization map obtained for the noise-reduced dataset binned at $64$ ms with $pp=30$. GRBs colored in magenta are classified as Extended Emission by previous works. Image taken from. \label{fig:fig1}](https://github.com/KenethGarcia/ClassiPyGRB/blob/1d0b3e43dd4c200382538ed2a60b695c49d064a4/JOSS_Docs/Figures/EE_analysis.jpg)

# Methodology

We complement the methodology presented in [@Jespersen2020]. Additionally, we implement the possibility of improving the signal/noise (S/N) ratio in two ways:

1. We applied the non-parametric noise reduction technique called FABADA [@Sanchez-Alarcon2022] to each band for every single light curve. 
2. We use the 10 s binned light curve data from the Swift/BAT catalog.

# Visualization and features

The use of this method produces 2D visualization maps colored by the duration of GRBs. Furthermore:

- It is possible to add either of the two features to the t-SNE maps or to visualize the raw data.
- There is the possibility of working only on the desired bands of BAT.
- Specific events can be searched for and highlighted on the display.
- Provisionally, those who are part of a study of extended emission GRBs will inform the user of this and show the reference. This will be expanded with another interesting features in the future.

We remark the fact that algorithms such as t-SNE visualization maps are very sensitive to any change in the perplexity and learning rate parameters. Therefore, as is the case when using any of these visualization techniques derived from machine learning, care must be taken in the correct interpretation of the data.

# Acknowledgements

KSGC acknowledges support from CONACyT fellowship. RLB acknowledges support from CONACyT postdoctoral fellowship.

# References
