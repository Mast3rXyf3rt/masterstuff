import numpy as np
def check_for_repeated_stims(stim_list):
    '''
    Takes stim_list as list of strings and returns tensor with booleans.
    '''
    # Count the occurrences of each string
    occurrences = {}
    for item in stim_list:
        if item in occurrences:
            occurrences[item] += 1
        else:
            occurrences[item] = 1
    # Create the stim_boolean array based on the counts
    stim_boolean = np.array([1 if occurrences[item] > 1 else 0 for item in stim_list])
    # Convert stim_boolean to a PyTorch tensor
    return stim_boolean