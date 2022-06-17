# Epicasting: An Ensemble Wavelet Neural Network (EWNet) for Forecasting Epidemics

In this study, we propose a novel MODWT-based auto-regressive neural network model with predefined architecture specially designed for forecasting epidemic datasets. Our analysis considers publicly available real-world infectious disease datatsets namely - influenza, dengue, hepatitis B, and malaria from different regions to analyse the epicasting abiility of our proposed model.

Usage of the repository for the paper "Epicasting: An Ensemble Wavelet Neural Network (EWNet) for Forecasting Epidemics":

* In this repository, we present examples with 15 epidemic data. These datasets are collected from varied open-source platforms and published manuscripts. The ".xlsx" files of these datasets available in [Source](https://github.com/mad-stat/Epicasting/tree/main/Datasets) contains disease incidence data reported on a monthly or weekly basis in different regions.  


The "models.R" file contains the implementation of popularly-used degree distributions, namely Lomax, power-law, power-law with cutoff, Log-normal, and Exponential distributions. Furthermore, the file "models.R" also contains the implementations of our proposed "Generalized Lomax" family of distributions, namely GLM Type-I, GLM Type-II, GLM Type-III and GLM Type-IV models. The decsirptions of all these models are provided in the manuscript titled "Searching for a new probability distribution for modeling non-scale-free heavy-tailed real-world networks".

Once the implementation is done, the predicted outputs of Lomax, power-law, power-law with cutoff, Log-normal, Exponential, GLM Type-I, GLM Type-II, GLM Type-III and GLM Type-IV models are restored in dataname_output.csv file. For example, in case of ego-Twitter(In).csv data set; all the predicted values based on different probability models are presented in ego-Twitter(In)_outputs.csv file. This file is further used for the computation of different metrics for finding predictive accuracy of several models in the manuscript.

Using the outputs of dataname_output.csv file, we obtain the graphs (Plots of degree distributions along with different proabbility distributions) for our paper and the codes are given in figures_plots.m (MATLAB implementation file).

Reults obtained in the paper for all these networks data sets can directly be computed along with the graphs and figures using the implementation files and data sets (along with outputs) given in this repository for replicability and sake of reproducibility of our paper. The rest of the data sets can be obtained from this link: http://snap.stanford.edu/data/index.html and similarly the implementations will be alike as shown for these 10 data sets.

## References
<a id="1">[1]</a> 
Dijkstra, E. W. (1968). 
Go to statement considered harmful. 
Communications of the ACM, 11(3), 147-148.
