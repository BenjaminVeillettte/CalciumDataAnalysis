"""
This script is used to predict spiking activity from calcium imaging data using two different algorithms: Cascade and OASIS.
It reads the data from the given csv and then applies the prediction algorithms.
"""



import os 
import numpy as np
import sys
from sys import path
import pandas as pd
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

#Path to read (Optionnal since its only to predict so you can put them directly into another script)
folder_path = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO", 
    "Data", 
)
file_data = f"m_neuron4_stim_867.csv"
file_path = os.path.join(folder_path, file_data)


def ReadData (file_path):
   """
   Read the data from the CSV file and return the relevant columns.
    Args:
    file_path (str): The path to the CSV file containing the data.
    Returns:
    tuple: A tuple containing the time, ephys, gcamp, and spikes data as numpy arrays.
   """
   df = pd.read_csv(
      file_path,
      sep=",",          
      decimal=".",      
      header=0
    )
   Order = ['Time', '1 Spikes', 'gcamp', 'spikes']
   df = df[Order]

   time = df["Time"].to_numpy()
   ephys = df["1 Spikes"].to_numpy()
   gcamp = df["gcamp"].to_numpy()
   spikes = df["spikes"].to_numpy()

   return time, ephys, gcamp, spikes


# Where the Oasis folder located 
folder_pathOasis = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO",
    "CalciumDataAnalysis",
    "AI prediction model",
    "OASIS-master"
)

# Adding the Oasis folder to the Python path
path.append(folder_pathOasis)

# Importing the deconvolve function from the oasis package
from oasis.functions import deconvolve


# Where the Cascade folder located
folder_pathCascade = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO",
    "CalciumDataAnalysis",
    "AI prediction model",
    "Cascade-master"
)


# Adding the Cascade folder to the Python path
if not os.path.isdir(folder_pathCascade):
 raise FileNotFoundError(f"{folder_pathCascade!r} n'existe pas")

# 2) Add it to Python's import search path (prioritaire)
if folder_pathCascade not in sys.path:
 sys.path.insert(0, folder_pathCascade)

# 3) Change your working dir to that same folder
os.chdir(folder_pathCascade)
print('Current working directory: {}'.format( os.getcwd() ))

# Importing Cascade functions
from cascade2p import checks, cascade

checks.check_packages()

# Base path to save results
base = os.path.join(
  os.path.expanduser("~"),
  "OneDrive - Cégep de Shawinigan",
  "Bureau",
  "Stage CERVO",
  "Code"
)


frame_rate = 100 # in Hz

def predictionCascade(base, gcamp):
   """
   Predict spiking activity from calcium imaging data using the Cascade algorithm.
   Args:
   base (str): The base path where the results will be saved.
   gcamp (numpy.ndarray): The calcium imaging data as a 1D numpy array.
   Returns:
   numpy.ndarray: The predicted spiking activity as a 2D numpy array.
   """
   # Prepare the data for the Cascade model
   datatoCascade = gcamp

   pathCascade = os.path.join(base, "ArrayCascade.npy")

   np.save(pathCascade, datatoCascade)
   
   def load_neurons_x_time(file_path):
       """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

       if file_path.endswith('.npy'):
         traces = np.load(file_path, allow_pickle=True)
        # if saved data was a dictionary packed into a numpy array (MATLAB style): unpack
         if traces.shape == ():
            traces = traces.item()['dF_traces']
       else:
           raise Exception('This function only supports .mat or .npy files.')
       if traces.ndim == 1:
         traces = traces[np.newaxis, :]

    # vérif’ rapide
       if traces.ndim != 2:
         raise ValueError(f"dff doit être 2D, got ndim={traces.ndim}")
       return traces
   traces = load_neurons_x_time(pathCascade)

   #Checking the lenght of the recording
   print('Number of neurons in dataset:', traces.shape[0])
   print('Number of timepoints in dataset:', traces.shape[1])

   cascade.download_model( 'update_models',verbose = 1)
   yaml_file = open('Pretrained_models/available_models.yaml')
   X = yaml.load(yaml_file)
   list_of_models = list(X.keys())
   model_name = 'Global_EXC_30Hz_smoothing50ms'
   cascade.download_model( model_name,verbose = 1)

   spike_prob = cascade.predict( model_name, traces )
   return spike_prob


def predictionOasis(base, time, gcamp):
    """
    Predict spiking activity from calcium imaging data using the OASIS algorithm.
    Args:
    base (str): The base path where the results will be saved.
    time (numpy.ndarray): The time vector as a 1D numpy array.
    gcamp (numpy.ndarray): The calcium imaging data as a 1D numpy array.
    Returns:
    numpy.ndarray: The predicted spiking activity as a 1D numpy array.
    """
   # Prepare the data for the OASIS model
    csv_name = "DonneesOASIS.csv"
    csv_path = os.path.join(base, csv_name)

    df = pd.DataFrame({
       "Time": time,
       "3 PMT1": gcamp,
       })
   
    df.to_csv(csv_path, index=False)
   
    data = pd.read_csv(csv_path)  
    y = data['3 PMT1'].values
    c, s, b, g, lam = deconvolve(y, penalty=1)
    return s 