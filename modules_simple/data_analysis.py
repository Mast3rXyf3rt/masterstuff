from modules_simple import Neural_Lib as nlb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import torch
import tqdm


def neural_histogram(responses, time_begin=0, time_end=199, bin_width=0.1, num_neurons=None):
    # Returns responses in shape [n_neurons, n_images, n_timebins]
    responses = nlb.preprocess_responses(responses, time_begin, time_end)
    # Shape right now is [n_images, n_neurons]

    n_neurons = num_neurons or responses.shape[1]

    # Determine the number of rows and columns for the subplots
    n_cols = 4  # or another value depending on how many subplots you want in each row
    n_rows = (n_neurons + n_cols - 1) // n_cols  # This ensures enough rows for all neurons

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
    axes = axes.flatten()  # Flatten in case of multiple rows and columns

    for n in range(n_neurons):
        neuron_responses = responses[:, n].numpy()
        # Determine the range of the data
        min_val = np.min(neuron_responses)
        max_val = np.max(neuron_responses)
        bins = np.arange(min_val, max_val + bin_width, bin_width)
        axes[n].hist(neuron_responses)
        axes[n].set_title(f'Neuron {n+1}')

    # Hide any unused subplots
    for i in range(n_neurons, n_rows * n_cols):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()



def plot_avrg_response(response_path, new=False, t_shown=50, t_end=120, n_neurons=None,m=0):
    """
    Takes responses and plots the average response of each individual neuron as neural activity per time.
    """
    #Returns responses in shape [n_neurons, n_images, n_timebins]
    # Load responses and preprocess them
    if new==False:
        responses, _, _ = nlb.load_mat_file(response_path)
    else:
        responses, _, _ ,_,_= nlb.load_mat_file(response_path)
    print(responses.shape)
    responses = np.transpose(responses, (1, 2, 0))
    print(responses.shape)
    # [n_images, n_timebins, n_neurons]
    # Look at mean data for neurons
    responses_mean = np.mean(responses, axis=0)
    print(responses_mean.shape)

    # Define the number of plots
    num_plots = n_neurons or responses.shape[2]
    neuron_indices = range(m*10,m*10+num_plots)
    num_bins=responses.shape[1]

    # Create a 3x5 grid of subplots
    fig, axs = plt.subplots(int(np.ceil(float(num_plots)/3.0)),3, figsize=(15, 9))
    axs = axs.flatten()  # Flatten the 2D array to 1D for easier iteration

    # Create a custom legend patch
    legend_patch = mpatches.Patch(color='gray', alpha=0.5, label='Image shown')

    # Plot each neuron data
    x = np.arange(0, num_bins*10, 10)
    for i, neuron_index in enumerate(neuron_indices):
        if i < len(neuron_indices):  # Ensure we don't go out of bounds
            neuron_mean = responses_mean[:, neuron_index]
            axs[i].plot(x, neuron_mean)
            axs[i].axvspan(500, 1200, color='gray', alpha=0.5)  # Add transparent gray tile
            axs[i].set_title(f'Neuron {neuron_index + 1}')
            axs[i].set_xlabel('Time (ms)')
            axs[i].set_ylabel('Response')
            # Add the legend to the first plot only to avoid repetition
            if i == 0:
                axs[i].legend(handles=[legend_patch])

    # Remove empty subplots (if any)
    for j in range(len(neuron_indices), len(axs)):
        fig.delaxes(axs[j])

    # Adjust layout
    plt.tight_layout()
    plt.show()

def oracle_prediction(test_responses,test_images, device):
    responses_tensor = torch.tensor(test_responses, device=device)
    oracle_prediction_tensor=torch.zeros(100,responses_tensor[0].shape[0])
    predicted_response_tensor=torch.zeros(100,responses_tensor[0].shape[0])
    for n in range(100):
        # Compute mean of four responses
        if not np.array_equal(test_images[5*n],test_images[5*n+1]) or torch.equal(responses_tensor[5*n],responses_tensor[5*n+1]):
            print(n)
            raise ValueError("Mistake: computing with not equal images, or exactly repeated response (unrealistic due to noise).")
        oracle_prediction_tensor[n] = responses_tensor[5*n : 5*n+4].mean(dim=0)
        predicted_response_tensor[n]=responses_tensor[5*n+4]
    corr_tensor=nlb.corr(oracle_prediction_tensor,predicted_response_tensor,dim=0)
    return corr_tensor

def oracle_bar_plot(test_responses,test_images,device,type=" "):   
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

def find_duplicate_images(images):
    num_images = len(images)
    total_num=0
    number_ims =0
    sub5=0
    sup5=0
    seen = set()
    for i in range(num_images):
        image_hash = hash(images[i].tobytes())
        if image_hash in seen:
            continue
        else:
            seen.add(image_hash)
        my_list=set()
        my_list.add(i)
        n=1
        for j in range(i + 1, num_images):
            if np.array_equal(images[i], images[j]):
                my_list.add(j)
                n += 1
        if n >1:
            total_num+=n
            number_ims+=1
            if n > 5:
                # print(f"{n-1} duplicates found for image {i} at")
                # print(my_list)
                sup5+=1
            else:
                sub5+=1
    print(number_ims,float(total_num)/float(number_ims))
    return total_num

def plot_im_activation(model, model_state, images, best_neurons,device,num_images=2500):
    model.load_state_dict(model_state)
    model=model.to(device)
    model.eval()
    """
    Sort the images by their activity for the best neurons.
    """
    fig, axs = plt.subplots(int(np.ceil(float(best_neurons.shape[0])/3.0)),3, figsize=(15, 9))
    axs = axs.flatten()  # Flatten the 2D array to 1D for easier iteration
    # print("Shape is")
    # print(model(images[0].unsqueeze(0)).shape)
    with torch.no_grad():
        for i,n in enumerate(best_neurons):
            predicted_activation = torch.tensor([model(im.unsqueeze(0)).squeeze()[n] for im in images])
            sorted_pa, sorted_indices = torch.sort(predicted_activation)
            #print(sorted_pa.shape)
            axs[i].plot(np.arange(0,num_images,1), sorted_pa[:num_images])
            axs[i].set_title(f'Activation per image for neuron {n}')
            axs[i].set_xlabel('Image ID')
            axs[i].set_ylabel('Predicted Neural Response')
            tick_pos = np.linspace(0,num_images,10,endpoint=False)
            tick_labels=sorted_indices[tick_pos].numpy()
            axs[i].set_xticks(tick_pos)
            axs[i].set_xticklabels(tick_labels)
    plt.tight_layout()
    plt.show

def plot_actual_activation(responses, images, best_neurons, time_window=(50, 120), num_images=2500):
    """
    Sort the images by their actual neural activity for the best neurons.
    
    Args:
        responses: Neural responses tensor [neuron_index, image_index, time]
        images: Image tensor (for reference only)
        best_neurons: Array of neuron indices to analyze
        time_window: Tuple of (start_time, end_time) in ms for response analysis
        num_images: Number of images to plot
    """
    # Convert time window to indices (assuming 10ms bins)
    t_start = time_window[0] // 10
    t_end = time_window[1] // 10
    
    fig, axs = plt.subplots(int(np.ceil(float(len(best_neurons))/3.0)), 3, figsize=(15, 9))
    axs = axs.flatten()  # Flatten the 2D array to 1D for easier iteration

    for i, n in enumerate(best_neurons):
        # Get mean activation over time window for this neuron
        actual_activation = torch.mean(responses[n, :, t_start:t_end], dim=1)
        sorted_act, sorted_indices = torch.sort(actual_activation)
        
        axs[i].plot(np.arange(0, num_images, 1), sorted_act[:num_images])
        axs[i].set_title(f'Actual activation per image for neuron {n}')
        axs[i].set_xlabel('Image ID')
        axs[i].set_ylabel('Neural Response')
        
        # Add ticks showing image indices
        tick_pos = np.linspace(0, num_images, 10, endpoint=False)
        tick_labels = sorted_indices[tick_pos.astype(int)].numpy()
        axs[i].set_xticks(tick_pos)
        axs[i].set_xticklabels(tick_labels)
        
        # Add mean and std as text
        mean_act = actual_activation.mean().item()
        std_act = actual_activation.std().item()
        axs[i].text(0.02, 0.98, f'Mean: {mean_act:.2f}\nStd: {std_act:.2f}', 
                   transform=axs[i].transAxes, 
                   verticalalignment='top')

    # Remove empty subplots if any
    for j in range(len(best_neurons), len(axs)):
        fig.delaxes(axs[j])
        
    plt.tight_layout()
    plt.show()

def plot_rf_and_top_images(model, model_state, images, best_neurons, device, width=128, height=72, num_images=2500):
    # Load model state
    if isinstance(model_state, str):
        # If model_state is a string (file path), load it
        state_dict = torch.load(model_state)
    elif isinstance(model_state, dict):
        # If model_state is already a state dict, use it directly
        state_dict = model_state
    else:
        raise ValueError("model_state must be either a file path or a state dict")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    fig, axs = plt.subplots(5, 4, figsize=(20, 25))

    for idx, neuron in enumerate(best_neurons):
        # Plot gradient RF
        x = torch.zeros(1, 1, width, height).to(device)
        x.requires_grad = True
        r = model(x)
        r[0, neuron].backward()
        rf = x.grad.cpu().numpy().squeeze()

        axs[idx, 0].imshow(rf.squeeze().transpose(), cmap='gray')
        axs[idx, 0].axis('off')
        axs[idx, 0].set_title(f'RF neuron {neuron}')

        # Get top 3 activating images
        with torch.no_grad():
            activations = torch.tensor([model(im.unsqueeze(0)).squeeze()[neuron] for im in images])
            _, top_indices = torch.topk(activations, 3)

        # Plot top 3 activating images
        for i in range(3):
            img = images[top_indices[i]].cpu().squeeze()
            axs[idx, i+1].imshow(img, cmap='gray')
            axs[idx, i+1].axis('off')
            axs[idx, i+1].set_title(f'Top {i+1} image')

    plt.tight_layout()
    plt.show()

def single_neuron_shuffle_analysis(test_responses, test_images, device, neuron_index, num_shuffles=1000):
    # Get the original correlation
    original_corr = oracle_prediction(test_responses, test_images, device)
    original_corr_neuron = original_corr[neuron_index].item()
    
    shuffled_corrs = torch.zeros(num_shuffles, device=device)
    
    for i in tqdm.notebook.trange(num_shuffles):
        # Create a copy of test_responses and shuffle the last response of each set
        shuffled_responses = test_responses.clone()
        shuffle_indices = torch.randperm(100)
        for j in range(100):
            shuffled_responses[5*j + 4] = test_responses[5*shuffle_indices[j] + 4]
        
        # Compute correlation for shuffled data
        shuffled_corr = oracle_prediction(shuffled_responses, test_images, device)
        shuffled_corrs[i] = shuffled_corr[neuron_index]
    
    # Move data to CPU for analysis and plotting
    shuffled_corrs = shuffled_corrs.cpu().numpy()
    
    # Compute statistics
    shuffled_mean = np.mean(shuffled_corrs)
    shuffled_std = np.std(shuffled_corrs)
    z_score = (original_corr_neuron - shuffled_mean) / shuffled_std
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.hist(shuffled_corrs, bins=50, alpha=0.7, color='skyblue', density=True)
    plt.axvline(original_corr_neuron, color='red', linestyle='dashed', linewidth=2, label='Original')
    plt.axvline(shuffled_mean, color='green', linestyle='dashed', linewidth=2, label='Shuffled Mean')
    
    plt.title(f'Shuffle Analysis for Neuron {neuron_index}')
    plt.xlabel('Correlation')
    plt.ylabel('Density')
    plt.legend()
    
    # Add text for original correlation and z-score
    plt.text(0.05, 0.95, f'Original: {original_corr_neuron:.3f}\nShuffled Mean: {shuffled_mean:.3f}\nZ-score: {z_score:.3f}', 
             transform=plt.gca().transAxes, verticalalignment='top')
    
    plt.tight_layout()
    plt.show()
    
    # Return summary statistics
    return {
        'neuron_index': neuron_index,
        'original_corr': original_corr_neuron,
        'shuffled_mean': shuffled_mean,
        'shuffled_std': shuffled_std,
        'z_score': z_score
    }

def plot_rf_and_activations(model, model_state, images, responses, best_neurons, device, width=128, height=72):
    """
    Args:
    - model
    - modelstate
    - images
    - responses: not preprocessed etc -> shape is [neuron_index, image_index, time: 0 -170ms] the last part is summed over for training data, but here we want to plot the responses over time
    Plot for each neuron:
    - Gradient RF
    - Top 3 predicted activating images (from model)
    - Top 3 actual activating images (from data)
    - Average over top5 images response profile per neuron 
    """
    # Load model state
    if isinstance(model_state, str):
        state_dict = torch.load(model_state)
    elif isinstance(model_state, dict):
        state_dict = model_state
    else:
        raise ValueError("model_state must be either a file path or a state dict")

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    responses = torch.tensor(responses.astype(np.float32)).to(device)

    # Create figure with subplots: 5 rows (neurons) x 8 columns 
    # (1 RF + 3 predicted images + 3 actual images + 1 response profile)
    fig, axs = plt.subplots(len(best_neurons), 8, figsize=(32, 5*len(best_neurons)))
    if len(best_neurons) == 1:
        axs = axs.reshape(1, -1)

    for idx, neuron in enumerate(best_neurons):
        # 1. Plot gradient RF
        x = torch.zeros(1, 1, width, height).to(device)
        x.requires_grad = True
        r = model(x)
        r[0, neuron].backward()
        rf = x.grad.cpu().numpy().squeeze()

        axs[idx, 0].imshow(rf.squeeze().transpose(), cmap='gray')
        axs[idx, 0].axis('off')
        axs[idx, 0].set_title(f'RF neuron {neuron}')

        # 2. Get and plot top 3 predicted activating images
        with torch.no_grad():
            activations = torch.tensor([model(im.unsqueeze(0)).squeeze()[neuron] for im in images])
            _, top_pred_indices = torch.topk(activations, 3)

            for i in range(3):
                img = images[top_pred_indices[i]].cpu().squeeze()
                axs[idx, i+1].imshow(img, cmap='gray')
                axs[idx, i+1].axis('off')
                axs[idx, i+1].set_title(f'Top {i+1} predicted')

        # 3. Get and plot top 3 actual activating images
        actual_responses = responses[neuron,:, :]
        print(actual_responses.shape)
        top_actual_indices = torch.topk(torch.sum(actual_responses[:,50:120],axis = 1), 3)[1]

        for i in range(3):
            img = images[top_actual_indices[i]].cpu().squeeze()
            axs[idx, i+4].imshow(img, cmap='gray')
            axs[idx, i+4].axis('off')
            axs[idx, i+4].set_title(f'Top {i+1} actual')

        # 4. Plot average response profile for top 5 images
        top_5_indices = torch.topk(torch.sum(actual_responses,axis = 1), 5)[1]
        top_5_responses_avg = torch.mean(actual_responses[top_5_indices,:], axis = 0)
        
        axs[idx, 7].plot(np.arange(0,170), top_5_responses_avg.cpu().numpy())
        axs[idx, 7].set_title(f'Avg Response of Neuron {neuron} over Top 5 Images')
        axs[idx, 7].set_xlabel('Time')
        axs[idx, 7].set_ylabel('Response')

    plt.tight_layout()
    plt.show()


# Usage example:
# plot_rf_and_activations(model, model_state, images, responses, best_neurons[:5], device)
