
![Logo](/docs/Animations/images/logo.jpeg) 

---
Title: 'ClassipyGRB: Machine Learning-Based Classification and Visualization of Gamma Ray Bursts using t-SNE'
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

date: 30 April 2023 

# Summary
Gamma-ray burst (GRBs) are the brightest events in the universe. Since decades, astrophysics have known about their cosmological nature. Every year, spacecraft missions such as Fermi and SWIFT, detect hundred of them. In spite of this large sample, these phenomena show a complex taxonomy in the first seconds after their appearence, it makes it very difficult to find resemblance between them using conventional techniques.

It is known that GRBs bursts are originated by the death of a massive star or from the merger of two compact objects. The typical classification is based on their duration [@Kouveliotou1993]. Nevertheless, events such as GRB 211211A [@Yang2022)], whose duration about 50 seconds lies in the group of long GRBs, has challenged this categorization by the evidence of features related with the short GRB population (the kilonova emission and the properties of its host galaxy). Therefore, this method is not completely reliable for the determination of the progenitors of such events.

Motivated by this problem, [@Jespersen2020] and [@Steinhardt2023] carried out analysis with t-SNE algorithm, showing that Swift/BAT GRBs database (consisting of light curves (flux/time) in four energy bands (15-25 keV, 25-50 keV, 50-100 keV, 100-350 keV)) cluster into two groups corresponding with the typical long/short classification. However, in this case, this classification is based on the information provided by their gamma-ray emission light curves. 

**ClassiPyGRB** is a Python 3 package to download, process, visualize and classify GRBs database from the [Swift/BAT Instrument](https://swift.gsfc.nasa.gov/about_swift/bat_desc.html) (up to July 2022). It is distributed over the GNU General Public License Version 2 (1991). We also included a noise-reduction and an interpolation tools for achieving a deeper analysis of these data.


# Statement of need

`ClassipyGRB` was designed to be used by astronomers whose scientific research is focused on GRBs. This module provides interactive visualizations of the light curves of GRBs, and the similarities they share. `ClassipyGRB` allows the comparison in a few seconds with other events in order to find resembling GRBs in high-frequencies through the proximity between them.

This method has been successfully applied for the correct identification of candidates for GRBs with extended emission that have not been previously identified by other groups, saving a lot of time and human effort [@Garcia-Cifuentes2023]. We locate extended emission GRBs reported by various authors under different criteria in our t-SNE maps and discuss its possible identification using only the gamma-ray data provided by the automatic pipeline of Swift/BAT. 

![t-SNE visualization map obtained for the noise-reduced dataset binned at $64$ ms with $pp=30$. GRBs colored in magenta are classified as Extended Emission by previous works. Image taken from [@Garcia-Cifuentes2023] \label{fig:fig1}](Figures/EE_analysis_updated.jpg)

Moreover, `ClassipyGRB` has been use to find in few seconds, similar GRBs with some specific feature, such as a bright component [@Angulo-Valdez2023].


# Methodology and Structure of ClassiPyGRB

`ClassipyGRB` mainly consists in three parts:

1) Retrieval and visualization of data from Swift/BAT: We implement an easy and fast code to download and plot an existing GRB post-processed data. There is the possibility to modify the resolution (2ms, 8ms, 16ms, 64ms, 256ms, 1s and 10s) and select those bands to work with. 


![Light curve of GRB 060614A. Image taken from [@Garcia-Cifuentes2023] \label{fig:fig2}](Figures/GRB060614.png)

2) Data processing: `ClassipyGRB` is able to: 

  - constrain the light curves in its duration $T_\mathrm{100}$ (this can be modified to $T_\mathrm{90}$ or $T_\mathrm{50}$).
  - normalise the flux.
  - standarise the temporal interval of all the sample (by zero-padding).
  - improve the signal/noise (S/N) ratio applying the non-parametric noise reduction technique called FABADA [@Sanchez-Alarcon2022] to each band for every single light curve. 
  - interpolate the flux between two specific times.

3) Visualization and plot of t-SNE maps

`ClassipyGRB` produces 2D visualization maps colored by the duration of GRBs. It includes:

- Intuitive graphic interface.
- It is possible to add either of the two features to the t-SNE maps or to visualize the raw data.
- Manupulation of the t-SNE parameters.
- Visualization of the light curves from the events with and without noise-reduction.
- There is the possibility of working only on the desired bands of Swift/BAT.
- Specific events can be searched for and highlighted on the display.

Moreover, any plot created with `ClassipyGRB` can be customized by the user.

Note: Algorithms such as t-SNE visualization maps are very sensitive to any change in the perplexity and learning rate parameters. Therefore, as is the case when using any of these visualization techniques derived from machine learning, care must be taken in the correct interpretation of the data.

This repository requires Python 3.8 or high, and the packages from the [``requeriments.txt``](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/requirements.txt) file. Other packages will be required optionally in Documentation (i.e., Jupyter).

# Acknowledgements

KSGC acknowledges support from CONACyT fellowship. RLB acknowledges support from CONACyT postdoctoral fellowship.

# References
