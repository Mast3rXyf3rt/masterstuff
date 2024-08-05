import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


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