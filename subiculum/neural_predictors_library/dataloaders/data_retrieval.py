import torch
import numpy as np
import scipy.io
from typing import Tuple
from torchvision import transforms
from neural_predictors_library.data_analysis.stimulus import check_for_repeated_stims
import os

def retrieve_mat_file(file_path):
    """
    Returns responses in shape [n_neurons, n_images, n_timebins]
    """
    data = scipy.io.loadmat(file_path)
    responses = data['responses']
    if 'stim_list' in data:
        stim_list = data['stim_list']
    else:
        stim_list=[]
    binsize = data['binsize']
    if 'idx_cellType' in data:
        idx_cellType = data['idx_cellType']
        if 'labels' in data:
            labels = data['labels']
            return responses, stim_list, binsize, idx_cellType, labels
        else:
            return responses, stim_list, binsize, idx_cellType
    else:
        if 'labels' in data:
            labels = data['labels']
            return responses, stim_list, binsize, labels
        else:
            return responses, stim_list, binsize

def retrieve_new_data(file_path: str, images_path: str,ids_path:str, device: torch.device):
    """
    Args:
    file_path: assumes that responses and further information on experimental data is stored in one .mat file
    images_path: assumes images to be stored in .npy file
    Returns:
    response_tensor: tensor of responses of shape [n_neurons, n_images, n_batches]
    images_tensor: tensor with images of shape [n_images, width, height]
    idx_cell_type: tensor with cell type of shape [n_neurons]
    id_session: tensor with the session id of the neurons of shape [n_neurons]
    test_boolean: tensor of booleans of shape [n_images]: 0 if image not repeated, 1 if images repeated -> will be used to divide into training/validation and test data
    """
    responses,_,_, idx_cellType,_ = retrieve_mat_file(file_path)
    responses=responses.astype(np.uint8)
    responses=torch.from_numpy(responses, dtype=torch.float32).to(device)

    images=np.load(images_path)
    transform=transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 72))
    ])
    images=transforms.ToTensor(transform(images)).to(device)
    
    idx_cellType=np.ndarray.flatten(idx_cellType)
    idx_cellType[idx_cellType==0]=4
    idx_cell_type=torch.from_numpy(idx_cellType).to(device)

    ids=scipy.io.loadmat(ids_path)
    ids=ids['rec']
    ids=ids[:,0].to(device)
    
    from neural_predictors_library.constants import stim_list 
    test_boolean=torch.from_numpy(check_for_repeated_stims(stim_list))
    
    return responses, images, idx_cell_type, ids, test_boolean

def retrieve_sensorium_data(root_dir, device):
    # Initialize lists to collect data
    responses_list = []
    images_list = []
    # Load the data
    for n in range(5994):
        image_path = os.path.join(root_dir, 'images', f'{n}.npy')
        response_path = os.path.join(root_dir, 'responses', f'{n}.npy')
        
        response_data = np.load(response_path)
        image_data = np.load(image_path)
        
        # Ensure image is grayscale
        if image_data.ndim == 3 and image_data.shape[0] == 3:
            image_data = np.mean(image_data, axis=0, keepdims=True)
        
        responses_list.append(response_data)
        images_list.append(image_data)
        
    # Convert lists to NumPy arrays
    sensorium_responses = torch.Tensor(responses_list).to(device)
    sensorium_images = np.array(images_list).astype('uint8').squeeze()
    return sensorium_responses, sensorium_images

def retrieve_old_data(images_path, responses_path, device):
    responses,_,_=retrieve_mat_file(responses_path)
    responses=responses.astype(np.uint8)
    responses=torch.from_numpy(responses, dtype=torch.float32).to(device)
    images=np.load(images_path)
    return responses, images

