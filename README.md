_This repository was put together by [Keneth Garcia](https://stivengarcia7113.wixsite.com/kenethgarcia) and supervised by **Dr. Rosa Becerra** and **Dr. Fabio De Colle**. Source and license info are on [GitHub](https://github.com/KenethGarcia/GRB_ML)._
- - -
# T-distributed Stochastic Neighbor Embedding (t-SNE) in Swift Data
As suggested by [Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract), Swift GRBs can be separated into two groups when t-SNE is performed. In this repository, we replicate this work by adding more recent data and an in-depth analysis of t-SNE performance. Moreover, we add synthetic GRBs performed using Machine Learning instances and join into Swift and other GRB data packages.

The project started on February 1st, 2022, and is expected to end in 2024. However, any posterior contribution will be received.



## Dependencies
This repository requires Python 3.8 or high, and the packages from the [``requeriments.txt``](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/requirements.txt) file. Other packages will be required optionally in Documentation (i.e., Jupyter).


## What can you do with this repository?
1. **Download Swift Data for different binnings.**

![](https://github.com/KenethGarcia/GRB_ML/blob/d7565536d1780a7da892ad2dcf35270f97ea3d6f/Documentation/README_Images/GRB060614.png)

2. **Pre-process Swift Data following the guidelines from [Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract)**

![](https://github.com/KenethGarcia/GRB_ML/blob/d7565536d1780a7da892ad2dcf35270f97ea3d6f/Documentation/README_Images/Limited_GRB060614.png)

3. **Perform t-SNE in high-level customization (thanks to [_openTSNE_](https://opentsne.readthedocs.io/en/latest/index.html) and [_scikit Learn_](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) implementations)**


4. **Create creative t-SNE animations showing its convergence:**

![](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/Documentation/Animations/convergence_animation_pp_30.gif)

**Or changing any hyperparameter (perplexity, learning rate, etc.):**

![](https://github.com/KenethGarcia/GRB_ML/blob/51482eecd01d8bea10a951ba3e9b0b108cea3c08/Documentation/Animations/perplexity_animation_2.gif)

5. **Reduce noise to any custom light curve using [FABADA](https://github.com/PabloMSanAla/fabada), checking its variance assuming a gray-scaled 1-pixel height image.**

![](https://github.com/KenethGarcia/GRB_ML/blob/d7565536d1780a7da892ad2dcf35270f97ea3d6f/Documentation/README_Images/Noise_Filtered_GRB060614.png)

6. **Interpolate between light curve data points using any custom n-order polynomial.**

![](https://github.com/KenethGarcia/GRB_ML/blob/d7565536d1780a7da892ad2dcf35270f97ea3d6f/Documentation/README_Images/Interpolated_GRB060614.png)

## Citation
If you use this repository in a scientific publication, we would appreciate citations: 

**_FUTURE CITATION_**

---