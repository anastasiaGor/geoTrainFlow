# geo_ML_workflow : Geospatial ML prediction workflow

This repository contains a workflow for supervised training on sets of geospatial data. The workflow consists of 4 main step: data preparation, training, prediction and diagnostics. The workflow is demonstrated on the example of prediction of subgrid temperature variance in the ocean for 2D surface and 3D interior datasets.

## Application
This type of workflow can be applied to any problem that can be formulated in the following way.
There is a dataset describing a distribution of some physical quantities $V_1, V_2, V_3 ,...$ in space on a discrete mesh $[x,y]$:
$$
V_n = V_n[x_i, y_i] 
$$
The objective is to predict an unknown quantity $Q$ and it is assumed that it is linked to the known quantities through some functional mapping $f$:
$$
Q[x_i, y_i] = f(V_1[x_i, y_i], V_2[x_i, y_i], V_3[x_i, y_i], ...) 
$$
The ML methods can be used to learni this functional mapping.
In order to formulate this problem for supervised learning, one needs training and validation dataset, where the input fields $V_1, V_2, V_3 ,...$ are combined with the 'ground truth' for the quantity $Q$ that needs to be predicted.

Particularity of the workflow consists in adapting this ML method to geospatial data, that has some particular issues such as non-uniform grids, presence of masked points and usage of `xArray` datasets.

## Case of subgrid temperature variance
One group of quantities that can predicted with such a workflow are subgrid-scale variances and fluxes. Subgrid-scale (SGS) temperature variance is defined as:
$$
\sigma^2_T = \langle T^2 \rangle - \langle T\rangle^2
$$
where the triangle brackets denote operator of coarsening from a higher to a lower resolution. 
