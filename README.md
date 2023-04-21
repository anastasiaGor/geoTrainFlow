# geo_ML_workflow : Geospatial ML prediction workflow

This repository contains a workflow for supervised training on sets of geospatial data. The workflow consists of 4 main step: data preparation, training, prediction and diagnostics. The workflow is demonstrated on the example of prediction of subgrid temperature variance in the ocean for 2D surface and 3D interior datasets.

# Application
This type of workflow can be applied to any problem that can be formulated in the following way.
There is a dataset describing a distribution of some physical quantities $V_1, V_2, V_3 ,...$ in space on a discrete mesh $[x,y]$:

$$ V_n = V_n[x_i, y_i] $$

The objective is to predict an unknown quantity $Q$ and it is assumed that it is linked to the known quantities through some functional mapping $f$:

$$ Q[x_i, y_i] = f(V_1[x_i, y_i], V_2[x_i, y_i], V_3[x_i, y_i], ...) $$

The ML methods can be used to learni this functional mapping.
In order to formulate this problem for supervised learning, one needs training and validation dataset, where the input fields $V_1, V_2, V_3 ,...$ are combined with the 'ground truth' for the quantity $Q$ that needs to be predicted.

Particularity of the workflow consists in adapting this ML method to geospatial data, that has some particular issues such as non-uniform grids, presence of masked points and usage of `xArray` datasets.

# Prediction of subgrid-scale temperature variance
One group of quantities that can predicted with such a workflow are subgrid-scale variances and fluxes. Subgrid-scale (SGS) temperature variance is defined as:

$$ \sigma^2_T = \langle T^2 \rangle - \langle T\rangle^2 $$

where the triangle brackets denote operator of coarsening from a higher to a lower resolution. In other words, the sibgrid temperature variance is a difference between square of temperature computed bedore applyting coarsening in its square after applying the coarsening operator. 

The SGS temperature variance described the fluctuations of temperature at scales smaller than the grid size. This quantity is used in some subgrid parametrization models, for example it is used to quantify the subgrid density error, which in its turn matters when computing the pressure or stratification vertical profiles [refs????].
In most cases, only low resolution data is available, and the SGS temperature variance cannot be computed directly by the definition, since it requires information of the square of temperature at high resolution. The works [refs???] suggest that some information about SGS variance can be inferred from the low-resolution temperature field, in particular, it is linked with the spatial gradients of the low-resolution temprature.
It indicates, that one can assume that there is functional maping linking the SGS temperature variance with the coarsened temperature field:

$$ \sigma^2_T \approx f(\langle T\rangle) $$

The idea is to use ML methods to find the functional mapping $f$. For this aim, we build a supervised learning workflow. 

# Supervised learning problem formulation
For the supervised learning one needs to create a database where the model inputs are combined with 'ground-truth' outputs. 
During the training the model predictions are compared with the true outputs and the error of the prediction is minimized. 
In case of subgrid-scale quatities, this dataset can be constructed by taking a high-resolution dataset and coarsening it to a lower resolution. In this case, the true SGS temperature variance can be computed in the stage of daraprerocessing directly according to the definition. 
Both inputs and outpus of the model are defined on the same low-resolution grid.

![Screenshot 2023-04-21 at 12 14 02](https://user-images.githubusercontent.com/6516711/233611296-811c3210-66c1-4c09-a53f-c4305f35f2ca.png)

# Used datasets 
The data used for the demonstration of the workflow is obtained from [eNATL60](https://github.com/ocean-next/eNATL60) simulation at resolution 1/60 degree.  6 datasets data extractions were used, corresponding to 3 regions (Gulf Stream, mid Atlantic and west Mediterranean) of size 10°x10° in 2 seasons (winter FMA and summer ASO). For the surface 2D data workflow 1-hour outputs were used, for the 3D data workflow - 1-day outputs. The extractions can be accessed through Pangeo catalog [ref???]. (Mention total data volume?)
The spatial data was coarsened from the high resolution 1/60 degree to 1/4 degree in horizontal directions. This resolution was chosen for the further implementation in NEMO ocean circulation model in a configuration with 1/4 degree resolution.

![Screenshot 2023-04-21 at 12 36 32](https://user-images.githubusercontent.com/6516711/233615546-b6cc96ef-beb3-41a4-9d83-700cd4ad10d5.png)

# Workflow
## 1. Data pre-treatment
First step of the workflow consist in preparing inputs, computing ground-truth outputs and coarsening to low-res grid. This step is performed with the use of `xArray` library with `Dask`. Coarsening is performed with the use of `xESMF` library. The prepared coarsened datasets are saved in `netcdf` format. The examples can be found in notebooks [add refs!!!]

## 2. Training
Demostration of all training stages can be found in notebooks [refs!!!]. It it coded with the use of `torch` and `pytorch Lightning`.
Training is launched with a `config` dictionary that contains all necessary information for data loader, model instantiation and training.

<img width="712" alt="Screenshot 2023-03-29 at 09 37 03" src="https://user-images.githubusercontent.com/6516711/233631316-8d83fb19-1b29-4c09-ad11-a33376810f3a.png">

### Data loader
The data loader creates batches for training/validation or testing. It assures the correct geometry and data content in samples. At this stage, the data is transformed from `xArray` format to `pytorch` tensor.

![Screenshot 2023-04-21 at 13 39 46](https://user-images.githubusercontent.com/6516711/233627056-e5cd92bf-2bf5-4d2c-a530-4f17fcc1fed1.png)

## Data split
Datasets for straining and validation are splitted in time: first 60% of snapshots in the datasets are used for training, and last 20% - for validation. The rest 20% of snapshots are skipped, this gap in was introduced to make validation data decorrelated from the training data.
The data used for validation during the training is also used for the after-training tests.

![Screenshot 2023-04-21 at 13 42 18](https://user-images.githubusercontent.com/6516711/233627297-54b2e908-35df-4f08-a6ef-a62e348932bb.png)

## Data geometry
All data is organized in tensors with presibed shape. For 2D data it is $[N,C,H,W]$, where $N$ is the number of samples in a batch, $C$ - number of channels (features), $H$ - image height, $W$ - image width. For 3D data the  shape is $[N,C,L,H,W]$, where another dimension $L$ - number of vertical levels is addded.

## Model architecture
In the present demostration, three models are trained and compared:
1. Linear regression - based on the paper [Stanley et al. 2021], baseline for comparison. Represents a linear NN with 1 neuron.
2. Fully connected NN on patches. 
3. CNN on the full images.

## Training 
- Use of the early stopping critera - when validation loss reaches its minimum and stops changing
- Training on all datasets, so the diffferent regions and datasets are mixed

## Testing
Once training is finished, the trained model is tested on each region and season separately. Some metrics are compared (MSE, correlation coefficient), as well as prediction snapshots are saved.

## 3. Prediction and its diagnostics
The trained model is loaded and some input data is fed to the model. Some diagnostics are evaluated on prediction, some metrics for prediction accuracy (visual fields, slices, hisrograms)

## 4. Model intercomparison, analysis
Includes prediction evaluation for few runs.

# Important details of implementation
1. Treatment of nans (coastline in ocean data)
- Nans are filled, all losses are evaluated with masks. 
- Eroded mask to remove points too close to the coastline from optimisation

2. Data normalization
Implemented approaches:
- Statistical normalization (by mean and std)
- Physical normalization (by a selected physical quantity)
- Normalization of loss -> needs for combined loss functions

3. Flexible choice of input/output features.
4. Unique data geometry for all models to enable straightforward intercomparison, since the same data loader is used.

# Environment and requirements
The workflow was developed and tested under Pangeo `torch-notebook` environment, the docker image can be found [here](https://github.com/pangeo-data/pangeo-docker-images/tree/master/pytorch-notebook).
It was run on 2i2c cloud.

#Acknowledgements

# Refenrences
