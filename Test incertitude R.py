import numpy as np 
from scipy import stats
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

def sliding(data, window, n, step=200):
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



SegmentCascade = sliding(PredCascade, window, n)
SegmentOasis = sliding(PredOasis, window, n)
SegmentSpike = sliding(spikes, window, n)
SegmentCalcium = sliding(gcamp, window, n)
SegmentTemps = sliding(time, window, n)

SegmentSpikeRate = [[(100 / n) * x for x in sous_liste] for sous_liste in SegmentSpike]
#print(SegmentSpikeRate)


def calc_R(Spikerate, Seg, addMoyenne = True):
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
   if addMoyenne == True:
        MoyenneR = np.nanmean(listeR)
        return MoyenneR
   else :
      return listeR

def scaleReference(ArrayBase, ArrayToscale):
    y1 = np.array(ArrayBase)
    y2 = np.array(ArrayToscale)

    y1_min = np.nanmin(y1)
    y1_max = np.nanmax(y1)

    scaler = MinMaxScaler(feature_range=(y1_min, y1_max))
    y2_scaled = scaler.fit_transform(y2.reshape(-1, 1)).flatten()
    return y2_scaled



def calc_Std(Array):
   std_value = np.zeros(len(Array))
   for i in range(len(Array)):
      value = Array[i]
      std = np.nanstd(value)
      std_value[i] = std
   return std_value



listeS1 = []
listeS2 = []
listeRm1 = []
listeRm2 = []

increments = np.arange(1, 201, 5)


for i in increments:
   SegmentCascade = sliding(PredCascade, window, i)
   SegmentOasis = sliding(PredOasis, window, i)
   SegmentSpike = sliding(spikes, window, i)
   SegmentSpikeRate = [[(100 / i) * x for x in sous_liste] for sous_liste in SegmentSpike]

   r1 = calc_R(SegmentSpikeRate, SegmentCascade, addMoyenne=False)
   r2 = calc_R(SegmentSpikeRate, SegmentOasis,addMoyenne=False)
   std_Cascade = stats.sem(r1, ddof=1) 
   std_Oasis = stats.sem(r2, ddof=1)

   listeS1.append(std_Cascade)
   listeS2.append(std_Oasis)

   Rm1 = calc_R(SegmentSpikeRate, SegmentCascade, addMoyenne=True)
   listeRm1.append(Rm1)
   Rm2 = calc_R(SegmentSpikeRate, SegmentOasis, addMoyenne=True)
   listeRm2.append(Rm2)



increments = np.array(increments)
temps = increments/100

def incertitudeGraph(donne, incertitude):
   High = []
   Low = []
   for a, b in zip(donne, incertitude):
      LigneHaut = a + b
      LigneBas = a - b
      High.append(LigneHaut)
      Low.append(LigneBas)
   return High, Low

SupCascade, infCascade = incertitudeGraph(listeRm1, listeS1)
SupOasis, infOasis = incertitudeGraph(listeRm2, listeS2)

SupCascade = np.array(SupCascade)
infCascade = np.array(infCascade)
SupOasis = np.array(SupOasis)
infOasis = np.array(infOasis)

df3 = pd.DataFrame({"Incrément": temps, " R de Cascade": listeRm1, "Borne sup": SupCascade, "Borne inf":infCascade})
df3.to_csv('PrécisionRdeCascade.csv',index=False)
df4 = pd.DataFrame({"Incrément": temps, " R d'Oasis": listeRm2, "Borne sup": SupOasis, "Borne inf":infOasis})
df4.to_csv('PrécisionRdeOasis.csv',index=False)

plt.fill_between(temps, infCascade, SupCascade, alpha=0.2, color="red")
plt.fill_between(temps, infOasis, SupOasis, alpha=0.2, color="blue")
plt.plot(temps, listeRm1, label="R de Cascade", color="red")
plt.plot(temps, listeRm2, label="R de oasis", color="blue")
plt.title("Précision en fonction incrément temps de lissage")
plt.xlabel("Incréments temps [s]")
plt.ylabel("Coefficient R")
plt.legend()

plt.show()