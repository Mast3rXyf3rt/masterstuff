import torch
import numpy as np
from neural_predictors_library.constants import on_response_bool, stim_list

"""
This module contains functions used for preprocessing.
They are in the order they should be called on the data.
1. The responses have to be added up over the time dimension.
    I wanted to train basic networks and not 
    - Recurrent Neural Networks
    - Long Short-Term Memory 
    - Temporal Convolutional Networks 
    Or a transformer based network.
2. Neural subsets can be chosen (different types present, recorded over multiple sessions (overlap of neurons, varying quality), filter for response "quality"-> I have done so by "eye", mght implement function later...)
3. Bad neurons (see oracle correlation in data_analysis) can be excluded.
4. Target needs to be normalized per neuron (range of response varies highly between neurons)
NB: The functions are not agnostic towards the order of processing steps!
"""

#This only works for my data, I did for some reason fail to import the stim_list from matlab (really weird file format) so I just copy-pasted it into a constant. The loader does return a stim_list, but it is an opaque object.

def preprocess_responses(responses:torch.Tensor, time_begin:int, time_end:int)->torch.Tensor:
    """
    Args:
    responses: [n_neurons, n_images, n_batches]
    time_begin, time_end: integers in [0, max(time_bin)] e.g. batch numbers where images appeared/disappeared from screen, can be chosen freely.

    -> [n_images, n_neurons] 
    """
    responses_p2 = responses.permute(1, 0, 2)
    # transposes to [n_images, n_neurons, n_timebins]
    responses_p3 = torch.sum(responses_p2[:,:,time_begin:time_end], dim=2)
    return responses_p3
    
#The dataset consists of different neural types and of neurons that were recorded in different sessions. This function allows to pick subsets, depending on: neural type, recording session, recognizable (hand-picked) on response,
def indices_of_neural_types(cell_type):
    set_of_types_indices = set()
    if "HD" in cell_type: set_of_types_indices.add(2)
    if "FS" in cell_type: set_of_types_indices.add(1)
    if "NC" in cell_type: set_of_types_indices.add(4)
    if "slow" in cell_type: set_of_types_indices.add(3)
    return set_of_types_indices

def neural_subset(responses: torch.Tensor, cell_type={"all"}, type_index=None, accepted_sessions={1,2,3}, session_index=None, response_quality={0,1,2})->torch.Tensor:
    """
    Args: 
    responses: preprocessed responses [n_images, n_neurons]
    cell_type: string in {"HD"=HD neurons,"FS"= FS neurons,"NC"=not classified neurons,"slow"=slow firing non-HD neurons}
    type_index: is returned by load as idx_cellType. Array with cell types of neurons.
    accepted_sessions= list of accepted sessions, default "all" (in my experiment it is 3 sessions) sessions.
    session_index: array with session index of neurons.
    response_quality: List with accepted grades of on response (in my case handpicked visually - for 165 neurons that was more efficient than implementing some automated grading...). 0: bad, 1:okay, 2: good.
    """
    # Kick out insufficiently bad neurons:
    responses=responses[:,on_response_bool in response_quality]
    type_index=type_index[:,on_response_bool in response_quality]
    session_index=session_index[:,on_response_bool in response_quality] or None
    # Reduce to specific neural types:
    if type_index is None and session_index is None:
        return responses
    elif "all" in cell_type or type_index is None:
        return responses[:, session_index in accepted_sessions]
    else:
        indices_types = indices_of_neural_types(cell_type)
        responses = responses[:, type_index in indices_types]
        session_index = session_index[:, type_index in indices_types] or None
        return responses[:,session_index in accepted_sessions] or responses

#Some neurons might have negative oracle correlations. It can be helpful to only train on neurons with a correlation bigger than some lower bound.

def exclude_bad_neurons(responses:torch.Tensor,images:torch.Tensor, test_boolean,device,lower_bound=0)->torch.Tensor:
    """
    Args:
    responses: preprocessed (time squeezed, subsets picked - has to be done first!) responses in shape [n_images, n_neurons]
    images: 
    """
    import neural_predictors_library.data_analysis.stimulus as stimulus
    from neural_predictors_library.data_analysis.oracle import oracle_prediction
    test_responses=responses[test_boolean==1]
    test_images=images[test_boolean==1]
    oracle_correlation_per_neuron= oracle_prediction(test_responses, test_images, device)
    responses=responses[:, oracle_correlation_per_neuron>lower_bound]
    return responses

#Normalization (output scale really varies on the set of neurons -> mean loss, correlation harder to interpret. Therefore, I use different normalizations (per neuron)). This is best to be done last in processing.

def minmax_scale_per_neuron(data: torch.Tensor) -> torch.Tensor:
    # Assuming data shape is (n_stimuli, n_neurons)
    min_vals, _ = data.min(dim=0, keepdim=True)
    max_vals, _ = data.max(dim=0, keepdim=True)
    return (data - min_vals) / (max_vals - min_vals + 1e-8)

def log_transform(data: torch.Tensor) -> torch.Tensor:
    return torch.log1p(data)  # log1p is log(1+x) to handle zero counts

def sqrt_transform(data: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(data)

def normalize_by_mean_rate(data: torch.Tensor) -> torch.Tensor:
    # Assuming data shape is (n_stimuli, n_neurons)
    mean_rates = data.mean(dim=0, keepdim=True)
    return data / (mean_rates + 1e-8)

def soft_normalize(data: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    # Assuming data shape is (n_stimuli, n_neurons)
    min_vals, _ = data.min(dim=0, keepdim=True)
    max_vals, _ = data.max(dim=0, keepdim=True)
    return alpha*(data - min_vals) / (max_vals - min_vals + 1)

def normalize_spike_counts(data: torch.Tensor, method: str = 'soft') -> torch.Tensor:
    if method == 'minmax':
        return minmax_scale_per_neuron(data)
    elif method == 'log':
        return log_transform(data)
    elif method == 'sqrt':
        return sqrt_transform(data)
    elif method == 'mean_rate':
        return normalize_by_mean_rate(data)
    elif method == 'soft':
        return soft_normalize(data, alpha=1.0)
    else:
        raise ValueError("Unknown normalization method")