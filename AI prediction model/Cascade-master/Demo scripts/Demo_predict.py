

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to predict spiking activity from calcium imaging data

The function "load_neurons_x_time()" loads the input data as a matrix. It can
be modified to load npy-files, mat-files or any other standard format.

The line "spike_prob = cascade.predict( model_name, traces )" performs the
predictions. As input, it uses the loaded calcium recordings ('traces') and
the pretrained model ('model_name'). The output is a matrix with the inferred spike rates.

"""



"""

Import python packages

"""

import os, sys
import matplotlib.pyplot as plt

folder_pathCascade = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO",
    "Code",
    "Cascade-master"
)

if not os.path.isdir(folder_pathCascade):
 raise FileNotFoundError(f"{folder_pathCascade!r} n'existe pas")

# 2) Add it to Python's import search path (prioritaire)
if folder_pathCascade not in sys.path:
 sys.path.insert(0, folder_pathCascade)

# 3) Change your working dir to that same folder
os.chdir(folder_pathCascade)
print('Current working directory: {}'.format( os.getcwd() ))

from cascade2p import checks
checks.check_packages()

import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth

"""

Define function to load dF/F traces from disk

"""


def load_neurons_x_time(file_path):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""
    
    if file_path.endswith('.mat'):
       traces = sio.loadmat(file_path)['dF_traces']
    elif file_path.endswith('.npy'):
        traces = np.load(file_path, allow_pickle=True)
        # if saved data was a dictionary packed into a numpy array (MATLAB style): unpack
        if traces.shape == ():
            traces = traces.item()['dF_traces']
    else:
        raise Exception('This function only supports .mat or .npy files.')
    
    # do here transposing or percent to numeric calculation if necessary
    #traces = traces.T
    #traces = traces / 100
    if traces.ndim == 1:
        traces = traces[np.newaxis, :]

    # vérif’ rapide
    if traces.ndim != 2:
        raise ValueError(f"dff doit être 2D, got ndim={traces.ndim}")
    return traces




"""

Load dF/F traces, define frame rate and plot example traces

"""


example_file = r'C:\Users\Utilisateur\OneDrive - Cégep de Shawinigan\Bureau\Stage CERVO\Code\ArrayGcamp.npy'

frame_rate = 100 # in Hz

traces = load_neurons_x_time( example_file )
print('Number of neurons in dataset:', traces.shape[0])
print('Number of timepoints in dataset:', traces.shape[1])


noise_levels = plot_noise_level_distribution(traces,frame_rate)
print(type(noise_levels))


#np.random.seed(3952)
neuron_indices = np.random.randint(traces.shape[0], size=10)
plot_dFF_traces(traces,neuron_indices,frame_rate)


"""

Load list of available models

"""

cascade.download_model( 'update_models',verbose = 1)

yaml_file = open('Pretrained_models/available_models.yaml')
X = yaml.load(yaml_file)
list_of_models = list(X.keys())





"""

Select pretrained model and apply to dF/F data

"""

model_name = 'Global_EXC_30Hz_smoothing50ms'
cascade.download_model( model_name,verbose = 1)

spike_prob = cascade.predict( model_name, traces )


"""

Save predictions to disk

"""


folder = os.path.dirname(example_file)
save_path = os.path.join(folder, 'full_prediction_3'+os.path.basename(example_file))

# save as numpy file
np.save(save_path, spike_prob)
#sio.savemat(save_path+'.mat', {'spike_prob':spike_prob})

"""

Plot example predictions

"""

neuron_indices = np.random.randint(traces.shape[0], size=10)
plot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob)
plt.show()


"""

Plot noise-matched examples from the ground truth

"""

#median_noise = np.round(np.median(noise_levels))
nb_traces = 2
duration = 40 # seconds
#plot_noise_matched_ground_truth( model_name, median_noise, frame_rate, nb_traces, duration )