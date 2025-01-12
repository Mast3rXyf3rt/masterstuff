import wandb
import neural_predictors_library.models
from login import login_wandb

login_wandb()
# load data




# configure model -> use model module

# train on data -> use training modulr

# define some objectives for the datasets

# maybe define a main function, I kind of do not want to run any code in this package

# define a sweep configuration: this will have to be sepcific for the models - definitely will have boilerplate code if I don't think about it enough

# run sweep



"""
Stuff to think about:
Regularization is also something I want to optimize my hyperparams for.
What is best practice there?
Optimize for:
- Model architecture
- Hyperparams of the model
- Regularization:
    1. L1
    2. L2
    3. ...
"""