#Pour faire rouler le code ce que j'ai fait pour commencer:
#Création environnement avec conda : conda create -n Cascade python=3.7 tensorflow==2.3 keras==2.3.1 h5py numpy scipy matplotlib seaborn ruamel.yaml spyder
# J'ai mis le dossier de code dans mon dossier qui contient Cascade vue que pour une raison j'ignore le path marchait pas autrement
#Ensuite pour Oasis : pip install oasis-deconv si conda install -c conda-forge oasis-deconv marche pas que j'avais fait messemble
# Pour le scaler : conda install scikit-learn
#À changer aussi pour les path sur ton ordi

import numpy as np 
import matplotlib.pyplot as plt
import os 
import sys
from sys import path
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


datatoCascade = gcamp
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



r1 = calc_R(SegmentSpikeRate, SegmentCascade)
print("R de Cascade =", r1)
r2 = calc_R(SegmentSpikeRate, SegmentOasis)
print("R de OASIS =", r2)


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


fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# top plot
ax1.plot(MoyenneTemps, y4, color="green")
ax1.set_title("Calcium data")
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("dF/F")

# bottom plot
ax2.plot(MoyenneTemps, y2_scaled, color="dimgrey", label=f"Predicted activity : Cascade (R = {r1:.6f})")
ax2.plot(MoyenneTemps, y1, color="red",label="Spikes rate")
ax2.plot(MoyenneTemps, y3_scaled, color="lightgrey", label=f"Predicted activity : Oasis (R = {r2:.6f})")
ax2.set_title("Comparaison of OASIS and Cascade algorithm based off groundtruth data")
ax2.legend(loc="best")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Spike rate [spikes/s]")

plt.xlim()
plt.legend()


df = pd.DataFrame({"Time": MoyenneTemps, "Gcamp": y4,"Spike rate":y1, "Cascade":y2_scaled, "Oasis":y3_scaled})
df.to_csv('DataScaled.csv',index=False)

df2 = pd.DataFrame({"Time": MoyenneTemps, "Cascade": y2, "Oasis": y3})
df2.to_csv('DataNoscale.csv',index=False)

increments = np.arange(1, 201, 5)
listeR1 = []
listeR2 = []
listeDev1 = []
listeDev2 = []
listeDev3 = []
for n in increments:
   MoyennePredCascade = calc_moyenne(PredCascade, n)
   MoyennePredOasis = calc_moyenne(PredOasis, n)
   MoyenneSpike = calc_moyenne(spikes, n)

   SegmentCascade = sliding(PredCascade, window)
   SegmentOasis = sliding(PredOasis, window)
   SegmentSpike = sliding(spikes, window)
   SegmentSpikeRate = [[(100 / n) * x for x in sous_liste] for sous_liste in SegmentSpike]

   MoyenneSpike = np.array(MoyenneSpike)
   MoyenneSpikerate = MoyenneSpike * (100/n)
   CascadeScale = scaleReference(MoyenneSpikerate, MoyennePredCascade)
   DeviationCascade = np.nanstd(CascadeScale)
   listeDev1.append(DeviationCascade)
   OasisScale = scaleReference(MoyenneSpikerate, MoyennePredOasis)
   DeviationOasis = np.nanstd(OasisScale)
   listeDev2.append(DeviationOasis)
   DeviationSpikes = np.nanstd(MoyenneSpikerate)
   listeDev3.append(DeviationSpikes)

   r1 = calc_R(SegmentSpikeRate, SegmentCascade)
   listeR1.append(r1)
   r2 = calc_R(SegmentSpikeRate, SegmentOasis)
   listeR2.append(r2)

#print(listeR1, listeR2)
increments = np.array(increments)
temps = increments/100 #Ramener sur une échelle de 1 seconde
plt.figure(figsize=(8, 4))
plt.plot(temps, listeR1, label="R de Cascade", color="red")
plt.plot(temps, listeR2, label="R de oasis")
plt.title("Précision en fonction incrément temps de lissage")
plt.xlabel("Incréments temps [s]")
plt.ylabel("Coefficient R")
plt.legend()

plt.show()


df3 = pd.DataFrame({"Incrément": temps, " R de Cascade": listeR1, "R d'Oasis": listeR2})
df3.to_csv('PrécisionIntervalleVariable.csv',index=False)

df4 = pd.DataFrame({"Incrément": temps, " Déviation de Cascade": listeDev1, "Déviation d'Oasis": listeDev2, "Déviation spike rate": listeDev3})
df4.to_csv("IncertitudesMesures.csv", index=False)


