---
title: 'ClassipyGRB: Classifying GRBs with Machine Learning t-SNE'
tags:
  - Python
  - astronomy
  - transient astronomy
  - gamma-ray bursts
  - extragalactic astronomy
authors:
  - name: Keneth Garcia-Cifuentes
    orcid: 0009-0001-2607-6359
    equal-contrib: true
    affiliation: 1
    corresponding: true

  - name: Rosa L. Becerra 
    orcid: 0000-0002-0216-3415
    equal-contrib: true
    affiliation: 1
    corresponding: true

  - name: Fabio De Colle
    orcid: 0000-0002-3137-4633
    affiliation: 1
    corresponding: true

affiliations:
 - name: Instituto de Ciencias Nucleares,  Universidad Nacional Autónoma de México, Apartado Postal 70-543, 04510 CDMX, México
   index: 1
date: 3 March 2023
bibliography: paper.bib

# Summary

Gamma-ray bursts (GRBs) are the brightest events in the universe. They have been traditionally classified based on their duration (larger and shorter than 2 seconds). The increase in detected GRBs with durations of more than 2 seconds but with properties similar to those of a short GRB (as known as extended emission GRBs), makes evident the need to find another classification criterion.

The duration is determined by the parameter $T_{90}$, which measures the time interval in which 90% of the gamma ray emission is detected in some instrument.

The importance of the classification lies mainly in knowing the progenitors of these events. To date, it is known that there are at least two types of processes that can give rise to a GRB. Firstly, the death of a supermassive star and secondly, the collapse of two compact objects (a black hole and a neutron star or a binary system of neutron star) being detected with durations greater and less than $T_{90}=2$ seconds respectively.

# Statement of need

The detection of GRBs are carried by spatial mission such as Swift with its BAT instrument. Using the Swift-BAT GRB catalog, consisting of light curves (flux/time) in four energy bands (15-25 keV, 25-50 keV, 50-100 keV, 100-350 keV) for about 1250 events, we are able of analyzing the close-by regions (see Figure 1 from [@Jespersen2020]). 

`ClassipyGRB` was designed to be used by astronomical researchers who carry out studies of GRBs. It provides interactive visualizations of the light curves of GRBs, and the similarities they share. This method allows the comparison in a few seconds with other events in order to find resembling GRBs through the proximity between them comparing their discrete Fourier transforms. 

This method has been successfully applied for the correct identification of candidates for GRBs with extended emission that have not been previously identified by other groups, saving a lot of time and human effort \autoref[@Garcia-Cifuentes2023]. We locate extended emission GRBs reported by various authors under different criteria in our t-SNE maps (REF) and discuss its possible identification using only the gamma-ray data provided by the automatic pipeline of Swift/BAT (see \autoref{fig:fig1}). 

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
