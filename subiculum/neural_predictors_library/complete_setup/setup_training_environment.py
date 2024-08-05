import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data Retrieval
from neural_predictors_library.dataloaders import data_retrieval, dataset, loader, preprocessers
def setup_loader_and_model():
    # Data Retrieval
    new_data_path = '/project/subiculum/new_data/natural_images_awake_postSub.mat'
    new_data_images_path = '/project/subiculum/new_data/new_images.npy'
    ids_path='/project/subiculum/new_data/IDs.mat'
    responses, images, idx_cell_type, ids, test_boolean = data_retrieval.retrieve_new_data(new_data_path, new_data_images_path,ids_path, device )
    # Preprocessing
    
    # Create Dataloaders

    # Initialize model

