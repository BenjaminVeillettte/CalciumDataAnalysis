import numpy as np 
import matplotlib.pyplot as plt
import os 
import sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from PredictionAlgorithme import predictionCascade, predictionOasis, ReadData

folder_path = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO", 
    "Data", 
)
file_data = f"m_neuron4_stim_867.csv"
file_path = os.path.join(folder_path, file_data)

time, ephys, gcamp, spikes = ReadData(file_path)


base = os.path.join(
  os.path.expanduser("~"),
  "OneDrive - Cégep de Shawinigan",
  "Bureau",
  "Stage CERVO",
  "Code"
)

frame_rate = 100 # in Hz
n = 50  #changer incrément temps (100 = 1s)

PredCascade = predictionCascade(base, gcamp)
PredCascade = PredCascade.squeeze()
PredOasis = predictionOasis(base, time, gcamp)

#Smoothing des données en les moyennant autour dans une plage de valeur
def calc_moyenne(Array, n):
    Moyenne = []
    Nb = len(Array)/n
    for i in range(1, round(Nb)+1):
     if i == round(Nb):
        calc2 = np.nanmean(Array[(i-1)*n:len(Array)])
        Moyenne.append(calc2)
     else:
        Calc = np.nanmean(Array[(i-1)*n:i*n])
        Moyenne.append(Calc)
    return Moyenne

MoyenneTemps = calc_moyenne(time, n)
MoyennePredCascade = calc_moyenne(PredCascade, n)
MoyennePredOasis = calc_moyenne(PredOasis, n)
MoyenneSpike = calc_moyenne(spikes, n)
MoyenneGcamp = calc_moyenne(gcamp, n)

window = 20 * frame_rate

def sliding(data, window, step=200):
    segments = []
    L = len(data)
    for i in range(0, L - window + 1, step):
        seg = data[i : i + window]
        seg = calc_moyenne(seg, n)
        segments.append(seg)

    if (L - window) % step != 0:
     seg = data[-window:]
     seg = calc_moyenne(seg, n)
     segments.append(seg)
    return segments



SegmentCascade = sliding(PredCascade, window)
SegmentOasis = sliding(PredOasis, window)
SegmentSpike = sliding(spikes, window)
SegmentCalcium = sliding(gcamp, window)
SegmentTemps = sliding(time, window)

SegmentSpikeRate = [[(100 / n) * x for x in sous_liste] for sous_liste in SegmentSpike]
#print(SegmentSpikeRate)


def calc_R(Spikerate, Seg):
   listeR = []
   for a, b in zip(Spikerate, Seg):
      y1 = np.array(a)
      y2 = np.array(b)

      y1_min = np.nanmin(y1)
      y1_max = np.nanmax(y1)

      scaler = MinMaxScaler(feature_range=(y1_min, y1_max))
      y2_scaled = scaler.fit_transform(y2.reshape(-1, 1)).flatten()
      mask = np.isfinite(y1) & np.isfinite(y2_scaled)
      a_clean = y1[mask]
      b_clean = y2_scaled[mask]

      R = np.corrcoef(a_clean, b_clean)[0,1]
      listeR.append(R)
   MoyenneR = np.nanmean(listeR)
   return MoyenneR

def scaleReference(ArrayBase, ArrayToscale):
    y1 = np.array(ArrayBase)
    y2 = np.array(ArrayToscale)

    y1_min = np.nanmin(y1)
    y1_max = np.nanmax(y1)

    scaler = MinMaxScaler(feature_range=(y1_min, y1_max))
    y2_scaled = scaler.fit_transform(y2.reshape(-1, 1)).flatten()
    return y2_scaled








y1 = np.array(MoyenneSpike)
y2 = np.array(MoyennePredCascade)
y3 = np.array(MoyennePredOasis)
y4 = np.array(MoyenneGcamp)

y1 = y1*(100/n) #Pour mettre en spike/sec je divise mon unité de 1sec de 100 par le nombre de séparation
y1_min = np.nanmin(y1)
y1_max = np.nanmax(y1)

scaler = MinMaxScaler(feature_range=(y1_min, y1_max))
y2_scaled = scaler.fit_transform(y2.reshape(-1, 1)).flatten()
y3_scaled = scaler.fit_transform(y3.reshape(-1, 1)).flatten()


def calc_Std(Array):
   std_value = np.zeros(len(Array))
   for i in range(len(Array)):
      value = Array[i]
      std = np.nanstd(value)
      std_value[i] = std
   return std_value

for i in range(50, 202, 50):
   MoyennePredCascade = calc_moyenne(PredCascade, i)
   MoyennePredOasis = calc_moyenne(PredOasis, i)
   MoyenneSpike = calc_moyenne(spikes, i)
   MoyenneTemps = calc_moyenne(time, i)
   MoyenneSpike = np.array(MoyenneSpike)
   MoyenneSpikerate = MoyenneSpike * (100/i)

   CascadeScale = scaleReference(MoyenneSpikerate, MoyennePredCascade)
   OasisScale = scaleReference(MoyenneSpikerate, MoyennePredOasis)

   std_Cascade = calc_Std(CascadeScale)
   std_Oasis = calc_Std(OasisScale)
   std_Spikes = calc_Std(MoyenneSpikerate)

   increment = i / 100 
   r1 = calc_R(SegmentSpikeRate, SegmentCascade)
   r2 = calc_R(SegmentSpikeRate, SegmentOasis)



   fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
# top plot
   ax1.errorbar(MoyenneTemps, OasisScale, color="dimgrey", label=f"Oasis",
                 xerr=std_Oasis, marker="o", linestyle="", elinewidth=2, capsize=4, markersize=3)
   ax1.errorbar(MoyenneTemps, MoyenneSpikerate, color="red",label="Spikes rate",
                xerr=std_Spikes, marker="o", linestyle="", elinewidth=2, capsize=4, markersize=3)
   ax1.plot(MoyenneTemps, MoyenneSpikerate, color="red", alpha=0.4)
   ax1.plot(MoyenneTemps, OasisScale, color="grey", alpha=0.4)
   ax1.set_title(f"Comparaison of Oasis algorithm based off groundtruth data, {increment: .1f} [s]")
   ax1.legend(loc="best")
   ax1.set_xlabel("Time [s]")
   ax1.set_ylabel("Spikes rate")

# bottom plot
   ax2.errorbar(MoyenneTemps, CascadeScale, color="dimgrey", label=f"Cascade",
                xerr=std_Cascade, marker="o", linestyle="", elinewidth=2, capsize=4, markersize=3)
   ax2.errorbar(MoyenneTemps, MoyenneSpikerate, color="red",label="Spikes rate",
                xerr=std_Spikes, marker="o", linestyle="", elinewidth=2, capsize=4, markersize=3)
   ax2.plot(MoyenneTemps, MoyenneSpikerate, color="red", alpha=0.4)
   ax2.plot(MoyenneTemps, CascadeScale, color="grey", alpha=0.4)
   ax2.set_title(f"Comparaison of Cascade algorithm based off groundtruth data. {increment: .1f} [s]")
   ax2.legend(loc="best")
   ax2.set_xlabel("Time [s]")
   ax2.set_ylabel("Spike rate [spikes/s]")


   plt.xlim()
   plt.legend()
   plt.tight_layout()
   plt.show()


