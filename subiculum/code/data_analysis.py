import Neural_Lib_Flo as nlb
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
import Neural_Lib_Flo as nlb
import torch


# def neural_histogram(path_responses, time_begin=0, time_end=199, num_bins=None, num_neurons=None):
#     responses,_,_ = nlb.load_mat_file(path_responses)
#     # Returns responses in shape [n_neurons, n_images, n_timebins]
#     responses = nlb.preprocess_responses(responses, time_begin, time_end)
#     # Shape right now is [n_images, n_neurons]
    
#     n_neurons =num_neurons or responses.shape[1]
    
#     # Determine the number of rows and columns for the subplots
#     n_cols = 4  # or another value depending on how many subplots you want in each row
#     n_rows = (n_neurons + n_cols - 1) // n_cols  # This ensures enough rows for all neurons
    
#     fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
#     axes = axes.flatten()  # Flatten in case of multiple rows and columns

#     for n in range(n_neurons):
#         neuron_responses = responses[:, n].numpy()
#         axes[n].hist(neuron_responses, bins=num_bins or 2*int(max(neuron_responses)))
#         axes[n].set_title(f'Neuron {n+1}')
    
#     # Hide any unused subplots
#     for i in range(n_neurons, n_rows * n_cols):
#         axes[i].axis('off')

#     plt.tight_layout()
#     plt.show()

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