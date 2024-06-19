from subiculum.code.Neural_Lib_Flo import *
import wandb
api=wandb.Api()
run = api.run("seifertflo/V1 Training 04-06-2024/4shpdtsc")
config = run.config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def configure_model(config, n_neurons, device):
    model = ConvModel(layers=config.get("layers"), 
                      input_kern=config.get("input_kern"), 
                      hidden_kern=config.get("hidden_kern"), 
                      hidden_channels=config.get("hidden_channels"), 
                      spatial_scale = config.get("spatial_scale"), 
                      std_scale = config.get("std_scale"),
                      output_dim=n_neurons)
    return model.to(device)

model = configure_model(config, 13, device)

images_path = '/project/subiculum/data/images_uint8.npy'
responses_path = '/project/subiculum/data/V1_Data.mat'
train_loader, val_loader, test_loader = dataloader_from_mat(images_path, responses_path, 75, 125, 64)

train_and_eval(model, config.get("epochs"), train_loader, test_loader, val_loader, device)