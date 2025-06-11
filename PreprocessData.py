"""    
This script first processes calcium imaging data by applying a low-pass filter, correcting for bleaching, and calculating the ΔF/F and Z-score.
It then computes the signal-to-noise ratio (SNR) and normalizes noise levels. The processed data is saved to a new CSV file, and plots of the results are generated
to make sure the processing was successful.
"""


import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.stats import zscore
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


input_folder = os.path.join(
    os.path.expanduser("~"),
    "OneDrive - Cégep de Shawinigan",
    "Bureau",
    "Stage CERVO",
    "Data2",
)

output_folder = os.path.join(input_folder, "processed")
os.makedirs(output_folder, exist_ok=True)


RemovedTime = 5 * 100  
frame_rate = 100
Wn = (2 * 10) / frame_rate
window = 30 * frame_rate  


def troncage(array, RemovedTime):
    """
    Remove points in the array to 
    avoid the first and last desired 
    seconds of data to remove recording artifacts.
    
    Parameters:
    array (np.ndarray): The input array to be truncated.
    RemovedTime (int): The number of points to remove from the start and end.
    Returns:
    np.ndarray: The truncated array.
    """ 
    return array[RemovedTime : len(array) - RemovedTime]

def LowPass(signal, time_array):
    """
    Applying a low-pass filter to the signal to remove noise above 10 Hz.
    Parameters:
    signal (np.ndarray): The input signal to be filtered.
    Returns:
    np.ndarray: The filtered signal.
    """
    mask = np.isfinite(signal) & np.isfinite(time_array)
    calcium_clean = signal[mask]
    time_clean = time_array[mask]

    b, a = butter(5, 10, btype="low", analog=False, output="ba")
    return filtfilt(b, a, calcium_clean), time_clean

def Double_exponential(t, const, amp_fast, amp_slow, tau_slow, tau_multiplier):
    '''Compute a double exponential function with constant offset.
    Parameters:
    t       : Time vector in seconds.
    const   : Amplitude of the constant offset. 
    amp_fast: Amplitude of the fast component.  
    amp_slow: Amplitude of the slow component.  
    tau_slow: Time constant of slow component in seconds.
    tau_multiplier: Time constant of fast component relative to slow. 
    '''
    tau_fast = tau_slow*tau_multiplier
    return const+amp_slow*np.exp(-t/tau_slow)+amp_fast*np.exp(-t/tau_fast)

def Rising_exponential(t, const, amp, tau):
    '''Compute a rising exponential function with constant offset.
    Parameters:
    t     : Time vector in seconds.
    const : Amplitude of the constant offset. 
    amp   : Amplitude of the rising component.  
    tau   : Time constant of the rising component in seconds.
    '''
    return const + amp * (1 - np.exp(-t / tau))

def bleaching_correct(calcium, time_array):
    """
    Correction for bleaching in fiber photometry using a double exponential.
    If the recording have a baseline that is going up, a rising exponential is used.
    Parameters:         
    calcium (np.ndarray): The calcium signal to be corrected.
    time_array (np.ndarray): The time vector corresponding to the calcium signal.
    Returns:
    np.ndarray: The corrected calcium signal after applying the bleaching correction.
    np.ndarray: The fitted curve used for correction.
    """

    #Removing Nan values and giving them the same lenghth
    mask = np.isfinite(calcium) & np.isfinite(time_array)
    calcium_clean = calcium[mask]
    time_clean = time_array[mask]

    #Setting initial and bounds for fitting
    max_sig = np.max(calcium_clean)
    inital_params = [max_sig/2, max_sig/4, max_sig/4, 3600, 0.1] 
    bounds = ([0      , 0      , 0      , 600  , 0],
          [max_sig, max_sig, max_sig, 36000, 1])
    
    #Setting the end plateau for rising exponential defined as the 5th percentile of the last 1000 points(10 last seconds)
    plateau  = np.percentile(calcium_clean[-1000:], 5)
    popt, _ = curve_fit(Double_exponential, time_clean, calcium_clean, p0=inital_params, bounds=bounds, maxfev=1000)
    fitted = Double_exponential(time_clean, *popt)

    popt2, _ = curve_fit(Rising_exponential, time_clean, calcium_clean, p0=[plateau, max_sig/4, 3600], bounds=([0, 0, 600], [max_sig, max_sig, 36000]), maxfev=1000)
    fitted2 = Rising_exponential(time_clean, *popt2)

    r2DownExp = r2_score(calcium_clean, fitted)
    r2RiseExp = r2_score(calcium_clean, fitted2)

    if r2DownExp > r2RiseExp:
        print("Using double exponential fit for bleaching correction.")
        return calcium_clean - fitted, fitted
    else:
        print("Using rising exponential fit for bleaching correction.")
        return calcium_clean - fitted2, fitted2

def bleaching_correct2(calcium):
    """
    Another method to correct for bleaching in fiber photometry using a high-pass filter.
    Parameters:
    calcium (np.ndarray): The calcium signal to be corrected.
    Returns:
    np.ndarray: The corrected calcium signal after applying the high-pass filter.
    """    
    #Note the baseline is harder to find with this method
    b,a = butter(2, 0.005, btype='high', fs=frame_rate)
    CalciumHigh = filtfilt(b, a, calcium, padtype='even')
    return CalciumHigh

def ScalingDFF(signal, baseline):
    """
    Normalizing the calcium trace to make it into ΔF/F.
    The methode used here is to divided the signal by the baseline found with the curve fitting and multiply by 100 to get a percentage.
    I also correct the baseline to be always positive by making the minimum value 0.
    Parameters:
    signal (np.ndarray): The calcium signal to be normalized.
    baseline (np.ndarray): The baseline signal used for normalization.
    Returns:
    np.ndarray: The normalized calcium signal in ΔF/F (%).
    """
    dff = (signal/baseline)*100

    mini = np.nanmin(dff)

    dff = dff + abs(mini) if mini < 0 else dff
    return dff

def sliding(data, win, step=200):
    """
    Creation of a sliding window to segment the data into smaller parts.
    That way we can get an appromixation note based on the whole recording but only using a small part of it.
    It is more precise than using the whole recording.
    Parameters:
    data (np.ndarray): The input data to be segmented.
    win (int): The size of the sliding window.
    step (int): The step size for the sliding window.
    Returns:
    list: A list of segments created from the input data.
    """ 
    #Note if the window = steps it will not overlap anymore.
    segments = []
    L = len(data)
    for i in range(0, L - win + 1, step):
        segments.append(data[i : i + win])
    if (L - win) % step != 0:
        segments.append(data[-win :])
    return segments


def noise_std_from_diff(signal):
    """
    Calculate the noise level from the standard deviation of the differences in the signal.
    This method is based on the Median Absolute Deviation (MAD) of the differences.
    Parameters:
    signal (np.ndarray): The input signal from which to calculate the noise level.
    Returns:
    float: The calculated noise level across the recording.
    """
    diffs = np.diff(signal)
    mad_diff = np.median(np.abs(diffs - np.median(diffs)))
    return mad_diff / (0.6745 * np.sqrt(2))

def calcPercentile(segments, perc):
    """
    Calculating the percentile of the segments created from the sliding function.
    It will be used to deteremine the SNR by defining a threshold for the signal at 95%.
    Parameters:
    segments (list): A list of segments created from the sliding function.
    perc (int): The percentile to calculate.
    Returns:
    float: The calculated percentile value across the segments."""

    values = [np.percentile(seg, perc) for seg in segments]
    return np.nanmean(values)

#Ajout calcul SNR en dépassant 2 sigma selon cote Z et bruit = std deviation (MAD) fois 1,42...
def PeakFinder(Zscore):
    """
    Finding the peak to calculate the SNR from the Z-score.
    This function identifies peaks in the Z-score data that are above a threshold (2 standard deviations).
    When it is bewlow, it will take maximum of the peak defined and average of all the peaks found.
    Parameters:
    Zscore (np.ndarray): The Z-score data from which to find peaks.
    Returns:
    float: The average peak value found in the Z-score data.
    """

    Peak = []
    PeakMean = []
    for value in Zscore:
        if value >= 2:
            Peak.append(value)
        if value < 2:
            if len(Peak) > 1:
                Maximum = np.max(Peak)
                PeakMean.append(Maximum)
                Peak = []
    if len(Peak) > 1:
        Maximum = np.max(Peak)
        PeakMean.append(Maximum)
    Moyenne = np.nanmean(PeakMean)
    return Moyenne

#Main loop to read all the csv files in the input folder
pattern = os.path.join(input_folder, "*.csv")
for file_path in glob.glob(pattern):
 
    df = pd.read_csv(file_path, sep=",", decimal=".", header=0)
    
    #Taking the required columns to process 
    required_cols = ["Time", "1 Spikes", "gcamp", "spikes"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' introuvable dans {file_path}.")
        else:
            df = df[required_cols]

    #Making sure they are array
    time_arr   = df["Time"].to_numpy()
    ephys_arr  = df["1 Spikes"].to_numpy()
    gcamp_arr  = df["gcamp"].to_numpy()
    spikes_arr = df["spikes"].to_numpy()

    #Removing artefacts
    time_tr   = troncage(time_arr, RemovedTime)
    ephys_tr  = troncage(ephys_arr, RemovedTime)
    gcamp_tr  = troncage(gcamp_arr, RemovedTime)
    spikes_tr = troncage(spikes_arr, RemovedTime)


    gcamp_filt, time_correct = LowPass(gcamp_tr, time_tr)

    gcamp_corr, curve = bleaching_correct(gcamp_filt, time_correct)

    Calcium_Dff = ScalingDFF(gcamp_corr, curve)
    Calcium_Zscore = zscore(gcamp_corr, nan_policy="omit")

    #Calucation of noise and SNR with Df/f and Z-score. No filtering is applied to the raw gcamp trace to get the real noise level.
    #SNR is defined as ration of signal peak to noise.
    Gcamp_corriger, baseline = bleaching_correct(gcamp_tr, time_tr)
    Fscaled = ScalingDFF(Gcamp_corriger, baseline)
    Calcium_NoProccess_Z = zscore(Gcamp_corriger, nan_policy="omit")
    PeakZ = PeakFinder(Calcium_NoProccess_Z)
    NoiseZ = noise_std_from_diff(Calcium_NoProccess_Z)
    SNRZ = float(PeakZ) / NoiseZ
    
    Noise = noise_std_from_diff(Fscaled)
    segments = sliding(Fscaled, window)
    SNR95 = calcPercentile(segments, 95) / Noise
    Value95 = calcPercentile(segments, 95)

    #Calculating the normalized noise level
    #This is the median of the absolute difference between each point divided by the square root of the frame rate and is given in percentage.
    Normalized_noise_levels = (np.nanmedian(np.abs(np.diff(Fscaled, axis=-1)), axis=-1) / np.sqrt(frame_rate)) * 100

    print(f"Normalized noise : {Normalized_noise_levels}")
    print(f"SNR with DFF : {SNR95}")
    print(f"SNR with Z score : {SNRZ}")

    a = ({
        "Time"             : time_tr,
        "1 Spikes"         : ephys_tr,
        "spikes"           : spikes_tr,
        "gcamp"            : gcamp_tr,
        "gcamp_filtered"   : gcamp_filt,
        "gcamp_corrected"  : gcamp_corr,
        "Calcium_Dff"      : Calcium_Dff,
        "Calcium_Zscore"   : Calcium_Zscore,
        "Normalized Noise Level" : Normalized_noise_levels * np.ones_like(time_tr),
        "SNR95%"           : SNR95 * np.ones_like(time_tr),
        "SNR with Z score" : SNRZ * np.ones_like(time_tr),
        "Baseline"         : baseline,
    })

    df_out = pd.DataFrame.from_dict(a, orient="index")
    df_out = df_out.transpose()

    #Saving the processed data to a new CSV file and making sure the output folder worked. I also made this since array can have different lengths.
   
    fname = os.path.basename(file_path)
    out_filename = f"{fname}"           
    out_path    = os.path.join(output_folder, out_filename)
    df_out.to_csv(out_path, index=False)

    print(f"→ '{fname}' traité et enregistré sous '{out_path}'.")


    #Plotting the results to visualize the processing steps and ensure correctness 
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    ax1.plot(time_tr, gcamp_tr, color="red")
    ax1.plot(time_correct, curve, color="blue")
    ax1.set_title("Raw calcium trace")
    ax1.set_ylabel("Fluorescence brute")
    ax1.set_xlim()
    
    
    ax2.plot(time_correct, gcamp_corr, color="blue")
    ax2.set_title("After filter 10 Hz & correction bleaching")
    ax2.set_ylabel("Fluorescence corrigée")
    ax2.set_xlim()

    ax3.plot(time_correct, Calcium_Dff, color="green")
    ax3.axhline(Value95, color="lightgrey", linestyle="--", label="Seuil SNR95 %")
    ax3.set_title("ΔF/F (%)")
    ax3.set_ylabel("ΔF/F (%)")
    ax3.legend(loc="best")
    
    ax4.plot(time_correct, Calcium_Zscore, color="purple")
    ax4.set_title("Z-score")
    ax4.set_ylabel("Z-score (σ)")

    
    plt.tight_layout()
    plt.show()