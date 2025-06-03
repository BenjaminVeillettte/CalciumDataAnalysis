import os
import glob
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit
from scipy.stats import zscore
import matplotlib.pyplot as plt


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
    return array[RemovedTime : len(array) - RemovedTime]

def LowPass(signal):
    b, a = butter(5, Wn, btype="low", analog=False, output="ba")
    return filtfilt(b, a, signal)

def Double_exponential(t, a, b, c, d, e):
    return a * np.exp(-b * t) + c * np.exp(-d * t) + e

def bleaching_correct(calcium, time_array):
    popt, _ = curve_fit(Double_exponential, time_array, calcium)
    fitted = Double_exponential(time_array, *popt)
    return calcium - fitted + np.nanmedian(calcium)

def sliding(data, win, step=200):
    segments = []
    L = len(data)
    for i in range(0, L - win + 1, step):
        segments.append(data[i : i + win])
    if (L - win) % step != 0:
        segments.append(data[-win :])
    return segments

def ScalingDFF(signal):
    baselines = []
    segments = sliding(signal, window)
    for seg in segments:
        baselines.append(np.percentile(seg, 5))
    baseline_mean = np.nanmean(baselines)
    dff = ((signal / baseline_mean) - 1) * 100
    return dff, baseline_mean

def noise_std_from_diff(signal):
    diffs = np.diff(signal)
    mad_diff = np.median(np.abs(diffs - np.median(diffs)))
    return mad_diff / (0.6745 * np.sqrt(2))

def calcPercentile(segments, perc):

    values = [np.percentile(seg, perc) for seg in segments]
    return np.nanmean(values)

#Ajout calcul SNR en dépassant 2 sigma selon cote Z et bruit = std deviation (MAD) fois 1,42...
def PeakFinder(Zscore):
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


pattern = os.path.join(input_folder, "*.csv")
for file_path in glob.glob(pattern):
 
    df = pd.read_csv(file_path, sep=",", decimal=".", header=0)
    
 
    required_cols = ["Time", "1 Spikes", "gcamp", "spikes"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' introuvable dans {file_path}.")
        else:
            df = df[required_cols]

    time_arr   = df["Time"].to_numpy()
    ephys_arr  = df["1 Spikes"].to_numpy()
    gcamp_arr  = df["gcamp"].to_numpy()
    spikes_arr = df["spikes"].to_numpy()

    time_tr   = troncage(time_arr, RemovedTime)
    ephys_tr  = troncage(ephys_arr, RemovedTime)
    gcamp_tr  = troncage(gcamp_arr, RemovedTime)
    spikes_tr = troncage(spikes_arr, RemovedTime)


    gcamp_filt = LowPass(gcamp_tr)


    gcamp_corr = bleaching_correct(gcamp_filt, time_tr)

    Calcium_Dff, Baseline = ScalingDFF(gcamp_corr)
    Calcium_Zscore = zscore(gcamp_corr, nan_policy="omit")

    Gcamp_corriger = bleaching_correct(gcamp_tr, time_tr)
    Fscaled, Baseline2 = ScalingDFF(Gcamp_corriger)
    Calcium_NoProccess_Z = zscore(Gcamp_corriger, nan_policy="omit")
    PeakZ = PeakFinder(Calcium_NoProccess_Z)
    NoiseZ = noise_std_from_diff(Calcium_NoProccess_Z)
    SNRZ = float(PeakZ) / NoiseZ
    
    Noise = noise_std_from_diff(Fscaled)
    segments = sliding(Fscaled, window)
    SNR95 = calcPercentile(segments, 95) / Noise
    Value95 = calcPercentile(segments, 95)

    Normalized_noise_levels = (np.nanmedian(np.abs(np.diff(Fscaled, axis=-1)), axis=-1) / np.sqrt(frame_rate)) * 100

    print(f"Normalized noise : {Normalized_noise_levels}")
    print(f"SNR with DFF : {SNR95}")
    print(f"SNR with Z score : {SNRZ}")

    df_out = pd.DataFrame({
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
        "Baseline"         : Baseline * np.ones_like(time_tr),
    })
    # Si vous souhaitez inclure les spikes tronqués :
    # df_out["spikes_tr"] = spikes_tr

   
    fname = os.path.basename(file_path)
    out_filename = f"processed_{fname}"           
    out_path    = os.path.join(output_folder, out_filename)
    df_out.to_csv(out_path, index=False)

    print(f"→ '{fname}' traité et enregistré sous '{out_path}'.")


    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    ax1, ax2, ax3, ax4 = axes.flatten()
    
    ax1.plot(time_tr, gcamp_tr, color="red")
    ax1.set_title("Trace Calcium brute")
    ax1.set_ylabel("Fluorescence brute")
    
    ax2.plot(time_tr, gcamp_corr, color="blue")
    ax2.set_title("Après filtrage 10 Hz & correction bleaching")
    ax2.set_ylabel("Fluorescence corrigée")
    
    ax3.plot(time_tr, Calcium_Dff, color="green")
    ax3.axhline(Value95, color="lightgrey", linestyle="--", label="Seuil SNR95 %")
    ax3.axhline(Baseline, color="dimgrey", linestyle="--", label="Baseline")
    ax3.set_title("ΔF/F (%)")
    ax3.set_ylabel("ΔF/F (%)")
    ax3.legend(loc="best")
    
    ax4.plot(time_tr, Calcium_Zscore, color="purple")
    ax4.set_title("Z-score")
    ax4.set_ylabel("Z-score (σ)")

    
    plt.tight_layout()
    plt.show()
