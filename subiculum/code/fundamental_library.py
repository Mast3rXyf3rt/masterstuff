import scipy.io
import torch
import h5py
import numpy as np
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, ToPILImage
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Define a function to load .mat files
def load_mat_file(file_path):
    data = scipy.io.loadmat(file_path)
    responses = data['responses']
    stim_list = data['stim_list']
    binsize = data['binsize']
    return responses, stim_list, binsize

def preprocess_responses(responses, time_begin, time_end):
    responses_p1 = torch.tensor(responses, dtype=torch.float32)
    responses_p2 = responses_p1.permute(1, 0, 2)
    responses_p3 = torch.sum(responses_p2[:,:,time_begin:time_end], dim=2)
    return responses_p3

class NeuralDataset(Dataset):
    def __init__(self, images, responses, transform=None):
        """
        Args:
            images (Tensor): Images tensor [N, C, H, W]
            responses (Tensor): Responses tensor [N, Features]
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.images = images
        self.responses = responses
        self.transform = transform or Compose([
            ToPILImage(),
            Resize((64, 64)),  # Resize images to 64x64
            ToTensor(),        # Converts numpy.ndarray (H x W) to a torch.FloatTensor of shape (C x H x W)
            Normalize(mean=[0.456], std=[0.224])  # Adjust mean and std for single-channel
        ])  

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        response = self.responses[idx]
        if self.transform:
            image = self.transform(image)
        
        return image, response

import hashlib

def create_data_loaders(dataset, train_ratio=0.6, val_ratio=0.2, batch_size=32):
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
    

def image_to_hash(image):
    return hashlib.md5(image.tobytes()).hexdigest()

def find_duplicate_images_hashing(images):
    hashes = {}
    duplicates = []
    for idx, image in enumerate(images):
        image_hash = image_to_hash(image)
        if image_hash in hashes:
            duplicates.append((hashes[image_hash], idx))
        else:
            hashes[image_hash] = idx
    return duplicates

def my_train_epoch(model, loader, optimizer, loss_fn,device):
    model.train()
    for images, responses in loader:
        images, responses = images.to(device), responses.to(device)  # Move data to the appropriate device
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, responses)
        loss.backward()
        optimizer.step()
    return loss

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    for images, responses in loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, responses)
        loss.backward()
        optimizer.step()
    return loss

def find_duplicates_hash(images):
    # Assuming 'images' is a list or array of images
    duplicates = find_duplicate_images_hashing(images)

    if duplicates:
        print(f"Found {len(duplicates)} duplicate pairs:")
        for i, j in duplicates:
            print(f"Image {i} is equal to Image {j}")
    else:
        print("No duplicates found.")

import torch
import torch.nn as nn
import warnings

import torch
import torch.nn.functional as F

# Define a custom transform to apply average pooling and upscaling
import torch.nn.functional as F
class CustomTransform:
    def __call__(self, img):
        # Ensure the image is a tensor
        if isinstance(img, np.ndarray):
            img = torch.tensor(img, dtype=torch.float32)
        elif isinstance(img, torch.Tensor):
            img = img.float()
        else:
            raise TypeError("Input should be a numpy array or torch tensor")

        # Ensure the image has only one channel
        if img.dim() == 2:
            img = img.unsqueeze(0)  # [H, W] -> [1, H, W]
        elif img.dim() == 3 and img.size(0) != 1:
            img = img.mean(dim=0, keepdim=True)  # Convert to grayscale

        # Apply average pooling to reduce 256 dimension to 128
        pooled_img = F.avg_pool2d(img, kernel_size=(1, 2))  # [1, 144, 128]

        # Interpolate to upscale to 64x64
        resized_img = F.interpolate(pooled_img.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False)
        
        return resized_img.squeeze(0)  # Remove batch dimension if added

def train_readout(model, train_loader, val_loader, num_epochs, optimizer, loss_fn, device):
    model.eval()  # Make sure the core is in eval mode
    model.readout.train()  # Only train the readout

    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_loss = float('-inf')
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)

    for epoch in range(num_epochs):
        model.readout.train()
        train_loss = 0.0
        for images, responses in train_loader:
            images, responses = images.to(device), responses.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, responses)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        val_loss = 0.0
        model.readout.eval()
        with torch.no_grad():
            for images, responses in val_loader:
                images, responses = images.to(device), responses.to(device)
                outputs = model(images)
                loss = loss_fn(outputs, responses)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        with torch.no_grad():
            val_corrs = get_correlations(model, val_loader, device)
            validation_correlation = val_corrs.mean()

        lr_scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{epochs}], validation correlation: {validation_correlation:.4f}, trainloss: {train_loss:.4f}')

        if validation_correlation > best_val_loss:
            best_val_loss = validation_correlation
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print('Early stopping triggered!')
                break




# def find_duplicate_images(images):
#     num_images = len(images)
#     duplicates = []
#     for i in range(num_images):
#         for j in range(i + 1, num_images):
#             if np.array_equal(images[i], images[j]):
#                 duplicates.append((i, j))
#     return duplicates

 #def find_duplicates_naive(images):
    # # Assuming 'images' is a list or array of images
    # duplicates = find_duplicate_images(images)

    # if duplicates:
    #     print(f"Found {len(duplicates)} duplicate pairs:")
    #     for i, j in duplicates:
    #         print(f"Image {i} is equal to Image {j}")
    # else:
    #     print("No duplicates found.")
















# import scipy.io
# import torch
# import h5py
# import numpy as np
# from torchvision.transforms import ToTensor, Normalize, Compose, Resize, ToPILImage
# import matplotlib.pyplot as plt
# from torch.utils.data import Dataset, DataLoader, random_split

# # Define a function to load .mat files
# # scipy.io.loadmat returns mat_dict (dictionary with variable names as keys, and loaded matrices as values).
# def load_mat_file(file_path):
#     data = scipy.io.loadmat(file_path)
#     responses = data['responses']
#     stim_list = data['stim_list']
#     binsize = data['binsize']
#     return responses, stim_list, binsize

# def load_images(file_path, start_index=0, num_images=None):
#     with h5py.File(file_path, 'r') as file:
#         all_images = file['all_Images']
#         # Here, I want to be able to load only a subset as running this on my CPU just takes too long if I load all of the images every time.
#         if num_images is not None:
#             end_index = start_index + num_images
#             image_subset = np.array(all_images[start_index:end_index,:,:])
#         else:
#             image_subset = np.array(all_images)  # Load all if no range given
#     return image_subset

# # def preprocess_images(images):
# #     transform = Compose([
# #         ToPILImage(),
# #         Resize((64, 64)),  # Resize images to 64x64
# #         ToTensor(),        # Converts numpy.ndarray (H x W) to a torch.FloatTensor of shape (C x H x W)
# #         Normalize(mean=[0.456], std=[0.224])  # Adjust mean and std for single-channel
# #     ])  
# #     # Convert images to float32 before applying transformations
# #     #images = images.astype(np.float32)
# #     images = torch.stack([transform(image) for image in images])
# #     return images

# # Do preprocessing of images once, so it is not done every time in training.

# def preprocess_responses(responses):
#     responses_p1 =torch.tensor(responses, dtype=torch.float32)
#     responses_p2 =responses_p1.permute(1,0,2)
#     responses_p3=torch.sum(responses_p2,dim=2)
#     return responses_p3

# # def divide_and_shuffle(images,responses,train_ratio,eval_ratio,test_ratio):
# #     num_loaded_images=images.shape[0]
# #     num_total_images = responses.shape[0]

# #     indices = np.arange(num_loaded_images)
# #     np.random.shuffle(indices)

# #     shuffled_images = images[indices]
# #     # Since we have responses for all images, we need to shuffle the responses using the same shuffled indices
# #     shuffled_responses = responses[indices,:, :]
# #     num_train = int(train_ratio * num_loaded_images)
# #     num_test = int(test_ratio * num_loaded_images)
# #     num_eval = num_loaded_images - num_train - num_test 
# #     # Split the shuffled data into training, testing, and evaluation sets
# #     images_train_data = shuffled_images[:num_train]
# #     responses_train_data = shuffled_responses[:num_train,:, :]
    
# #     images_test_data = shuffled_images[num_train:num_train + num_test]
# #     responses_test_data = shuffled_responses[num_train:num_train + num_test,:, :]

# #     images_eval_data = shuffled_images[num_train + num_test:]
# #     responses_eval_data = shuffled_responses[num_train + num_test:,:, :]
    
# #     return images_train_data, images_test_data, images_eval_data, responses_train_data, responses_test_data, responses_eval_data



# class NeuralDataset(Dataset):
#     def __init__(self, images, responses, transform=None):
#         """
#         Args:
#             images (Tensor): Images tensor [N, C, H, W]
#             responses (Tensor): Responses tensor [N, Features]
#             transform (callable, optional): Optional transform to be applied on an image.
#         """
#         self.images = images
#         self.responses = responses
#         self.transform = transform or Compose([
#             ToPILImage(),
#             Resize((64, 64)),  # Resize images to 64x64
#             ToTensor(),        # Converts numpy.ndarray (H x W) to a torch.FloatTensor of shape (C x H x W)
#             Normalize(mean=[0.456], std=[0.224])  # Adjust mean and std for single-channel
#         ])  

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         image = self.images[idx]
#         response = self.responses[idx]  # Index responses properly
#         if self.transform:
#             image = self.transform(image)
#         return image, response

# def CustomDataLoader(number_of_images, images_path, responses_path, train_ratio, val_ratio):
#     responsesm, _, _ = load_mat_file(responses_path)
#     images = load_images(images_path, 0, number_of_images)
#     responses = preprocess_responses(responsesm)
#     dataset = NeuralDataset(images, responses)
#     total_size = len(dataset)
#     train_size = int(train_ratio * total_size)
#     val_size = int(val_ratio * total_size)
#     test_size = total_size - train_size - val_size
#     train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
#     return train_loader, val_loader, test_loader






    
