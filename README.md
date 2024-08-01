# This Repo can be used to predict neural responses to image stimuli.
This library was specifically written for data in mat files. As I, in the beginning, shared memory with other users and frequently ran into runtime errors I chose to convert the images files to npy files locally. Therefore, the loader expect only all files except the image files in mat format.
You will, basically, find three different models + 2 readout options to train. A simple CNN, a depth separable CNN, an even simpler linear model and the two readouts (explained in the code).
