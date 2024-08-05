from torch.utils.data import DataLoader, random_split
from neural_predictors_library.dataloaders.dataset import NeuralDataset, NeuralDatasetAwake

def new_loader(responses, images,test_boolean, batch_size):
    test_responses = responses[test_boolean == 1]
    training_validation_data = responses[test_boolean == 0]
    test_images=images[test_boolean==1]
    training_validation_images=images[test_boolean==0]
    data_set=NeuralDatasetAwake(training_validation_images,training_validation_data)
    val_ratio=0.2
    train_ratio=0.8
    total_size = len(data_set)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    train_dataset, val_dataset = random_split(data_set, [train_size, val_size])
    test_dataset=NeuralDatasetAwake(test_images,test_responses)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def sensorium_loader(responses, images, batch_size):
    train_ratio=0.8
    val_ratio=0.1
    dataset = NeuralDatasetAwake(images, responses)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader

def old_loader(responses,images,batch_size):
    train_ratio=0.8
    val_ratio=0.1
    dataset = NeuralDataset(images, responses)
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader