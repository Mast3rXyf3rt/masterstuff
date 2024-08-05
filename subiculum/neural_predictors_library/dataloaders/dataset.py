from torchvision.transforms import ToTensor, Normalize, Compose, Resize, ToPILImage
from torch.utils.data import Dataset

#New data and sensorium

class NeuralDatasetAwake(Dataset):
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
            Resize((128, 72)),  # Resize images to 64x64
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
    
#Old data
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