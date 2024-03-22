# Deep Learning Models for the Visual System

## Day 1: Data and LN Models

### Setup

- GPU machines for everyone
- Docker rights for everyone
- S3 access keys distributed (do not store them in git!)

### Get to know the data and import it ([Notebook](01_data.ipynb))

- Images & Response vector
- Program a PyTorch dataset
- What are data loaders and data samplers
- How to get data on the GPU

### Fitting your first linear model ([Notebook](02_linear_model.ipynb))

* How to build a model in PyTorch
* Setting up a training loop with loss and optimizers
* `model.eval()` and `model.train()`
* training error vs. validation error
* Visualizing the weights
* Problems with dimensionality

### Slightly improving the model

* Regularizers
* Batchnorm
* Downsampling the image
* Adding a nonlinearity
* Change from MSE to Poisson Loss

## Day 2: Deep Models and Gaussian Readout

### Modularization

* Clean up the code: training epoch, compute correlation, use normalizers provided by `neuralpredictors`

### First Deep Model

* The problem of dimensionality (again)
* Ways to deal with it: Factorized Readout, Gaussian Readout
* Implement Gaussian Readout
* The importance of a good initialization

### ML tricks

* Early stopping
* Training schedule (learning rate decrease)
* Sparse regularizer
* Hyperparameter selection (weights and biases)
* Optimize models over night

### How good is good
* Repeats 
* Oracle correlation


## Day 3: Visualization

### Linear Receptive Fields

* How to run classical experiments on the model
* How to get RFs with a gradient

### Maximally exiciting inputs

* How to compute MEIs
* Smoothing
* Ensembling and generalization tests

## Day 4: Further improvements

### Optional: Behavioral variables
* Eye movements
* Pupil dilation and behavior

### Project planning
* Sensorimotor cortex
* Subiculum
* Conni's data
* Superior Colliculus
* Astrocytes
* V1 Video
* Bring your own data