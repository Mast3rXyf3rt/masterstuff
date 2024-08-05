import torch
import seaborn as sns
import numpy as np
from neural_predictors_library.correlation import get_correlations, corr
import matplotlib.pyplot as plt


def oracle_comparison(model, model_state_path, device, test_loader):
    """ 
    Args:
    model, model_state_path: model and trained state to be evaluated
    test_loader: Needs to contain repeated stimuli
    
    Computes oracle prediction on the test_loader (leave-one-out correlation) and compares this to the model's performance.
    """
    for x, y in test_loader:
        print(x.shape, y.shape)
        x = x.detach().cpu().numpy()
        print(np.abs(np.diff(x, axis=0)).max())
        break
    responses, oracle_predictor = [], []
    for _, y in test_loader:
        y = y.detach().cpu().numpy()
        responses.append(y)
        n = y.shape[0]
        trial_oracle = (n * np.mean(y, axis=0, keepdims=True) - y) / (n - 1)
        oracle_predictor.append(trial_oracle)
    responses = np.vstack(responses)
    oracle_predictor = np.vstack(oracle_predictor)
    oracle_correlation = corr(responses, oracle_predictor, dim=0)
    model= model
    state_dict=torch.load(model_state_path)
    model.load_state_dict(state_dict)
    model.to(device)
    with torch.no_grad():
        test_corrs = get_correlations(model, test_loader, device)
    sns.set_context('notebook', font_scale=1.5)
    print(test_corrs.shape)
    print(oracle_correlation.shape)
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(test_corrs, kde=False, ax = ax, color=sns.xkcd_rgb['denim blue'], label='Test')
    sns.histplot(oracle_correlation, kde=False, ax = ax, color='deeppink', label='Oracle')
    ax.legend(frameon=False)
    sns.set_context('notebook', font_scale=1.5)
    with sns.axes_style('ticks'):
        fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(oracle_correlation, test_corrs, s=3, color=sns.xkcd_rgb['cerulean'])
    ax.grid(True, linestyle='--', color='slategray')
    ax.plot([0, 1], [0, 1], color='black', linestyle='--')
    ax.set(
        xlabel='Oracle correlation',
        ylabel='Model correlation',
        xlim=[0, 1],
        ylim=[0, 1],
        aspect='equal',
    )

def oracle_prediction(test_responses,test_images, device):
    """
    Args:
    test_responses: [n_images = 500, n_neurons]
    test_images: [n_images= 500, width, height], images need to have been repeatedly shown m*5 for some m. Optimal as tensor.
    """
    n_neurons=test_responses.shape[1]
    if not torch.is_tensor(test_responses):
        responses_tensor = torch.tensor(test_responses, device=device)
        responses_tensor.view(100,5,n_neurons)
    else:
        responses_tensor=test_responses
        responses_tensor.view(100,5,n_neurons)
    if not torch.is_tensor(test_images):
        test_images=torch.tensor(test_images, device=device)
        test_images=test_images.view(100,5,test_images.shape[1],test_images.shape[2])
    if not torch.all(test_images[:,0:1]==test_images[:,1:]):
        raise ValueError("Mistake: computing with unequal images")
    oracle_prediction_tensor=torch.mean(responses_tensor[:,1:,:],dim=1).squeeze()
    predicted_response_tensor=responses_tensor[:,1,:].squeeze()
    corr_tensor=corr(oracle_prediction_tensor,predicted_response_tensor,dim=0)
    return corr_tensor


def oracle_bar_plot(test_responses,test_images,device,type=" "):   
    """
    Args:
    test_responses: [n_images = 500, n_neurons]
    test_images: [n_images= 500, n_neurons], images need to have been repeatedly shown m*5 for some m.
    type: String with neural type (only relevant for title in the plot).
    """
    corrs=oracle_prediction(test_responses,test_images,device)
    # Assuming your tensor is named 'corr_tensor' and might be on GPU
    corr_array = corrs.detach().cpu().numpy()

    # Create an array for the x-axis (neuron numbers)
    neuron_numbers = np.arange(1, len(corr_array) + 1)
    mean = "%.2f" % (100*torch.mean(corrs).item())
    # Create the bar plot
    plt.figure(figsize=(12, 6))
    plt.bar(neuron_numbers, corr_array, label=f"mean correlation = {mean}%")

    # Customize the plot
    plt.title(f'Correlation Values by{type}Neuron')
    plt.xlabel(f'Neurons ({len(corr_array)} total)')
    plt.ylabel('Correlation')
    plt.legend()
    
    # If you have many neurons, you might want to adjust the x-axis ticks
    plt.xticks(np.arange(0, len(corr_array) + 1, 20))  # Show every 20th neuron number

    # Add a horizontal line at y=0 for reference
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)

    # Optionally, add a color bar to represent correlation strength
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=corr_array.min(), vmax=corr_array.max())), 
                label='Correlation Strength')

    # Show the plot
    plt.tight_layout()
    plt.show()