import numpy as np 
import matplotlib.pyplot as plt
import os 
from sys import path
import sys
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import MinMaxScaler
import ruamel.yaml as yaml
import random
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

PredCascade = predictionCascade(base, gcamp)
PredCascade = PredCascade.squeeze()
PredOasis = predictionOasis(base, time, gcamp)
#print(PredOasis)
n = 50  #changer incrément temps (100 = 1s)
#0.01 0.05 0.1 0.25 0.5 0.75 1 sec
# 1    5    10   25   50  75 100

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



GaussianCascade = gaussian_filter1d(PredCascade, sigma=6.125)
GaussianOasis = gaussian_filter1d(PredOasis, sigma=6.125)
GaussianSpikes = gaussian_filter1d(spikes, sigma=6.125)
GaussianGcamp = gaussian_filter1d(gcamp, sigma=6.125)

window = 3 * frame_rate

def sliding(data, window, step=300):
    segments = []
    L = len(data)
    i = 0
    while i + window <= L:
        DonneSmooth = calc_moyenne(data[i : i + window], n)
        segments.append(DonneSmooth)
        i += step
    if i < L:
        segments.append(calc_moyenne(data[i : L], n))
    return segments


SegmentCascade = sliding(PredCascade, window)
SegmentOasis = sliding(PredOasis, window)
SegmentSpike = sliding(spikes, window)
SegmentCalcium = sliding(gcamp, window)
SegmentTemps = sliding(time, window)

MoyenneTemps = calc_moyenne(time, n)
MoyennePredCascade = calc_moyenne(PredCascade, n)
MoyennePredOasis = calc_moyenne(PredOasis, n)
MoyenneSpike = calc_moyenne(spikes, n)


print(len(SegmentTemps))

SegmentCascadeGauss = sliding(GaussianCascade, window)
SegmentOasisGauss = sliding(GaussianOasis, window)
SegmentSpikeGauss = sliding(GaussianSpikes, window)
SegmentCalciumGauss = sliding(GaussianGcamp, window)

SegmentSpikeRate = [[(100 / n) * x for x in sous_liste] for sous_liste in SegmentSpike]
SegmentSpikeRateGauss = [[(100 / n) * x for x in sous_liste] for sous_liste in SegmentSpikeGauss]
#print(SegmentSpikeRate)

def scaleReference(ArrayBase, ArrayToscale):
    y1 = np.array(ArrayBase, dtype=float)
    y2 = np.array(ArrayToscale, dtype=float)

    y1_min = np.nanmin(y1)
    y1_max = np.nanmax(y1)

    scaler = MinMaxScaler(feature_range=(y1_min, y1_max))
    y2_scaled = scaler.fit_transform(y2.reshape(-1, 1)).flatten()
    return y2_scaled

def calc_R(Spikerate, Seg):
   listeR = []
   for a, b in zip(Spikerate, Seg):
      y1 = np.array(a, dtype=float)
      y2 = np.array(b, dtype=float)

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
   return np.array(listeR), MoyenneR

WindowR1, MoyenneR1 = calc_R(SegmentSpikeRate, SegmentCascade)
WindowR2, MoyenneR2 = calc_R(SegmentSpikeRate, SegmentOasis)

y1 = np.array(MoyenneSpike)
y2 = np.array(MoyennePredCascade)
y3 = np.array(MoyennePredOasis)


y1 = y1*(100/n) #Pour mettre en spike/sec je divise mon unité de 1sec de 100 par le nombre de séparation
y1_min = np.nanmin(y1)
y1_max = np.nanmax(y1)

scaler = MinMaxScaler(feature_range=(y1_min, y1_max))
y2_scaled = scaler.fit_transform(y2.reshape(-1, 1)).flatten()
y3_scaled = scaler.fit_transform(y3.reshape(-1, 1)).flatten()

GaussSpikeRate = GaussianSpikes * (100/n)

GaussCascadeScaled = scaleReference(GaussSpikeRate, GaussianCascade)
GaussOasisScaled = scaleReference(GaussSpikeRate, GaussianOasis)
GaussWindowR1, GaussR1 = calc_R(SegmentSpikeRateGauss,SegmentCascadeGauss)
GaussWindowR2, GaussR2 = calc_R(SegmentSpikeRateGauss,SegmentOasisGauss)

heatmap_data1 = WindowR1[np.newaxis, :]
heatmap_data2 = WindowR2[np.newaxis, :]
heatmap_data3 = GaussWindowR1[np.newaxis, :]
heatmap_data4 = GaussWindowR2[np.newaxis, :]
fig, axs = plt.subplots(3, 2, sharex=True, figsize=(8, 6))
axs.shape == (3,2)

# On récupère d’abord la colonne de gauche (indice 0), puis la colonne de droite (indice 1).
ax1, ax2, ax3 = axs[:, 0]  # [axs[0,0], axs[1,0], axs[2,0]]
ax4, ax5, ax6 = axs[:, 1]

t_minMoyen = MoyenneTemps[0]    
t_maxMoyen = MoyenneTemps[-1]

t_min = time.iloc[0]    
t_max = time.iloc[-1]

ax1.plot(MoyenneTemps, y2_scaled, color="dimgrey", label=f"Predicted activity : Cascade (R = {MoyenneR1:.6f})")
ax1.plot(MoyenneTemps, y1, color="red",label="Spikes rate", alpha=0.6)
ax1.plot(MoyenneTemps, y3_scaled, color="lightgrey", label=f"Predicted activity : Oasis (R = {MoyenneR2:.6f})")
ax1.set_ylabel("Spiking activity")
ax1.set_title("Average in sliding window")

ax2.imshow(heatmap_data1, aspect='auto', cmap='autumn_r', extent=[t_minMoyen, t_maxMoyen, 0, 1], origin='lower',vmin=0,vmax=1)
ax2.set_title("Heatmap Cascade")

ax3.imshow(heatmap_data2, aspect='auto', cmap='autumn_r', extent=[t_minMoyen, t_maxMoyen, 0, 1], origin='lower',vmin=0,vmax=1)
ax3.set_title("Heatmap Oasis")

ax4.plot(time, GaussCascadeScaled, color="dimgrey", label=f"Predicted activity : Cascade (R = {GaussR1:.6f})")
ax4.plot(time, GaussSpikeRate, color="red",label="Spikes rate", alpha=0.6)
ax4.plot(time, GaussOasisScaled, color="lightgrey", label=f"Predicted activity : Oasis (R = {GaussR2:.6f})")
ax4.set_ylabel("Spiking activity")
ax4.set_title("Average with gaussian filter")

im = ax5.imshow(heatmap_data3, aspect='auto', cmap='autumn_r', extent=[t_min, t_max, 0, 1], origin='lower',vmin=0,vmax=1)
ax5.set_title("Heatmap Cascade")

im2 = ax6.imshow(heatmap_data4, aspect='auto', cmap='autumn_r', extent=[t_min, t_max, 0, 1], origin='lower',vmin=0,vmax=1)
ax6.set_title("Heatmap Oasis")

fig.colorbar(im2, ax=ax6, orientation='vertical', fraction=0.046, pad=0.04)
fig.colorbar(im, ax=ax5, orientation='vertical', fraction=0.046, pad=0.04)

x1 = random.randint(0, len(GaussCascadeScaled) - 2000)
x2 = x1 + 2000



plt.legend()
plt.tight_layout()
plt.show()

print("R de Cascade =", MoyenneR1)
print("R de Oasis =", MoyenneR2)

print("R de Cascade Gauss =", GaussR1)
print("R de Oasis Gauss =", GaussR2)