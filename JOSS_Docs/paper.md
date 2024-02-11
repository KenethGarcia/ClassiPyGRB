---
title: 'ClassipyGRB: Machine Learning-Based Classification and Visualization of Gamma Ray Bursts using t-SNE'
tags:
  - Python
  - astronomy
  - transient astronomy
  - gamma ray bursts
  - extragalactic astronomy
authors:
  - name: Keneth Garcia-Cifuentes
    orcid: 0009-0001-2607-6359
    equal-contrib: true
    affiliation: 1
    corresponding: true # (This is how to denote the corresponding author)
  - name: Rosa L. Becerra
    orcid: 0000-0002-0216-3415
    affiliation: 1
  - name: Fabio De Colle
    orcid: 0000-0002-3137-4633
    affiliation: 1
affiliations:
 - name: Instituto de Ciencias Nucleares,  Universidad Nacional Autónoma de México, Apartado Postal 70-543, 04510 CDMX, México
   index: 1
date: 22 May 2023
bibliography: paper.bib
--- 

# Summary
Gamma-ray burst (GRBs) are the brightest events in the universe. Since decades, astrophysicists have known about their cosmological nature. Every year, spacecraft missions such as Fermi and SWIFT detect hundreds of them. In spite of this large sample, these phenomena show a complex taxonomy in the first seconds after their appearance, which makes it very difficult to find resemblance between them using conventional techniques.

It is known that GRBs are originated by the death of a massive star or from the merger of two compact objects. The typical classification is based on their duration [@Kouveliotou:1993]. Nevertheless, events such as GRB 211211A [@Yang:2022], whose duration of about 50 seconds lies in the group of long GRBs, has challenged this categorization by the evidence of features related with the short GRB population (the kilonova emission and the properties of its host galaxy). Therefore, a classification based only on their gamma-ray duration this is not completely reliable to determine the progenitors of such events.

Motivated by this problem, [@Jespersen:2020] and [@Steinhardt:2023] carried out analysis of GRB lightcurves by using the t-SNE algorithm, showing that Swift/BAT GRBs database, consisting of light curves in four energy bands (15-25 keV, 25-50 keV, 50-100 keV, 100-350 keV), clusters into two groups corresponding with the typical long/short classification. However, in this case, this classification is based on the information provided by their gamma-ray emission light curves. 

**ClassiPyGRB** is a Python 3 package to download, process, visualize and classify GRBs database from the [Swift/BAT Instrument](https://swift.gsfc.nasa.gov/about_swift/bat_desc.html) (up to July 2022). It is distributed over the GNU General Public License Version 2 (1991). We also included a noise-reduction and an interpolation tools for achieving a deeper analysis of the data.


# Statement of need

`ClassipyGRB` was designed to be used by astronomers whose scientific research is focused on GRBs. This module provides interactive visualizations of the light curves of GRBs, and the similarities they share. `ClassipyGRB` allows the comparison in a few seconds with other events in order to find resembling GRBs in high-frequencies through the proximity between them.

This method has been successfully applied for the correct identification of candidates for GRBs with extended emission that have not been previously identified by other groups, saving time and human effort [@Garcia-Cifuentes:2023]. We locate extended emission GRBs reported by various authors under different criteria in our t-SNE maps and discuss its possible identification using only the gamma-ray data provided by the automatic pipeline of Swift/BAT. 

![t-SNE visualization map obtained for the noise-reduced dataset binned at $64$ ms with $pp=30$. GRBs colored in magenta are classified as Extended Emission by previous works. Image taken from [@Garcia-Cifuentes:2023] \label{fig:fig1}](Figures/EE_analysis_updated.jpg)

Moreover, `ClassipyGRB` has been use to find in few seconds, similar GRBs with some specific feature, such as a bright component (Angulo Valdez et al. 2023, in prep).


# Methodology and Structure of ClassiPyGRB

`ClassipyGRB` mainly consists of three parts:

1) Retrieval and visualization of data from Swift/BAT: We implement an easy and fast code to download and plot an existing GRB post-processed data. There is the possibility to modify the time resolution of the datafiles (2ms, 8ms, 16ms, 64ms, 256ms, 1s and 10s) and analyze the data by selecting only some of the energy bands. 


![Light curve of GRB 060614A. Image taken from @Garcia-Cifuentes:2023 \label{fig:fig2}](Figures/GRB060614.png)

2) Data processing. `ClassipyGRB` is able to: 

  - constrain the light curves by their duration $T_\mathrm{100}$, $T_\mathrm{90}$ or $T_\mathrm{50}$.
  - normalise the flux.
  - standarise the temporal interval of all the sample (by zero-padding).
  - improve the signal/noise (S/N) ratio applying the non-parametric noise reduction technique called FABADA [@Sanchez-Alarcon:2022] to each band for every single light curve. 
  - interpolate the flux between two specific times.

3) Visualization and plot of t-SNE maps

`ClassipyGRB` produces 2D visualization maps colored by the duration of GRBs. It includes:

- Intuitive graphic interface.
- It is possible to add either of the two features to the t-SNE maps or to visualize the raw data.
- Manipulation of the t-SNE parameters.
- Visualization of the light curves from the events with and without noise-reduction.
- There is the possibility of working only on the desired bands of Swift/BAT.
- Specific events can be searched for and highlighted on the display.

Moreover, any plot created with `ClassipyGRB` can be customized by the user.

Note: Algorithms such as t-SNE visualization maps are very sensitive to any change in the perplexity and learning rate parameters. Therefore, as is the case when using any of these visualization techniques derived from machine learning, care must be taken in the correct interpretation of the data.

This repository requires Python 3.8 or high, and the packages from the [``requeriments.txt``](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/requirements.txt) file. Other packages will be required optionally in Documentation (i.e., Jupyter).

# Acknowledgements

KSGC acknowledges support from CONACyT fellowship. RLB acknowledges support from CONACyT postdoctoral fellowship.

# References
