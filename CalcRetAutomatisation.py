import numpy as np 
from scipy.stats import sem
import matplotlib.pyplot as plt
import os 
import sys
import glob
from sys import path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

folder_liste = ["Gcamp6", "Gcamp7", "Gcamp8"]

input_folder = os.path.join(
    os.path.expanduser("~"),
    "OneDrive - Cégep de Shawinigan",
    "Bureau",
    "Stage CERVO",
    "Analysis"
)

folder_pathOasis = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO",
    "Code",
    "OASIS"
)

# Ajouter le chemin pour les modules oasis
path.append(folder_pathOasis)
from oasis.functions import deconvolve


# 2) Chemin vers Cascade-master (contenant le sous-dossier cascade2p/)
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


from cascade2p import checks, cascade

checks.check_packages()

frame_rate = 100 # in Hz
window = 20 * frame_rate

ArrangeGen = np.arange(0.01, 1.02, 0.20)
Folder_outname = []
FolderAverage_out = []

results = { "Cascade": {}, "Oasis":   {}}

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

def sliding(data, window, n, step=500):
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


    def load_neurons_x_time(file_path):
        if file_path.endswith('.npy'):
            traces = np.load(file_path, allow_pickle=True)
        if traces.shape == ():
            traces = traces.item()['dF_traces']
        else:
            raise Exception('This function only supports .mat or .npy files.')
        if traces.ndim == 1:
            traces = traces[np.newaxis, :]
        return traces
           
    traces = load_neurons_x_time(pathCascade)
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

def incertitudeGraph(donne, incertitude):
   High = []
   Low = []
   for a, b in zip(donne, incertitude):
      LigneHaut = a + b
      LigneBas = a - b
      High.append(LigneHaut)
      Low.append(LigneBas)
   return High, Low

def ajoutdico(IncrémentDico, Array2, Array3, Array4, dicoR, dicoUp, dicoDown):
    for temps, R , BornUp, BornDo in zip(IncrémentDico, Array2, Array3, Array4):
        dicoR[(temps)].append(R)
        dicoUp[(temps)].append(BornUp)
        dicoDown[(temps)].append(BornDo)
    return dicoR, dicoUp, dicoDown

def MoyenneValue(dico, liste):
    for i in np.arange(0.01, 1.02, 0.20):
        i = np.round(i, 2)
        Moyenne = np.nanmean(dico[i])
        liste.append(Moyenne)
    return liste

for folder in folder_liste:
    full_path = os.path.join(input_folder, folder)
    if full_path not in sys.path:
        sys.path.append(full_path)
    pattern = os.path.join(full_path, "*.csv")

    output_name = f"{folder}_calculated"
    output_folder = os.path.join(input_folder, output_name)
    os.makedirs(output_folder, exist_ok=True)
    Folder_outname.append(output_name)

    for file_path1 in glob.glob(pattern):
        df = pd.read_csv(file_path1, sep=",", decimal=".", header=0)
        required_cols = ["Time", "1 Spikes", "Calcium_Dff", "spikes"]

        for col in required_cols:
            if col not in df.columns:
             raise KeyError(f"Colonne '{col}' introuvable dans {file_path1}.")
            else:
             df = df[required_cols]
        
        time = df["Time"].to_numpy()
        ephys = df["1 Spikes"].to_numpy()
        gcamp = df["Calcium_Dff"].to_numpy()
        spikes = df["spikes"].to_numpy()


        base = os.path.join(
           os.path.expanduser("~"),
           "OneDrive - Cégep de Shawinigan",
           "Bureau",
           "Stage CERVO",
           "Code"
        )
        mask = np.isfinite(gcamp) & np.isfinite(time)
        gcamp_clean = gcamp[mask]
        time_clean = time[mask]

        datatoCascade = gcamp_clean
        
        pathCascade = os.path.join(base, "ArrayCascade.npy")
        np.save(pathCascade, datatoCascade)

        csv_name = "DonneesOASIS.csv"
        csv_path = os.path.join(base, csv_name)
        df = pd.DataFrame({"Time": time_clean,"3 PMT1": gcamp_clean,})
        df.to_csv(csv_path, index=False)
        
        def predictionCascade():
           def load_neurons_x_time(file_path):
              traces = np.load(file_path, allow_pickle=True)
              if traces.ndim == 1:
                 traces = traces[np.newaxis, :]
              return traces
           
           traces = load_neurons_x_time(pathCascade)
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

        PredCascade = predictionCascade()
        PredCascade = PredCascade.squeeze()

        def predictionOasis():
           data = pd.read_csv(csv_path)  
           y = data['3 PMT1'].values
           c, s, b, g, lam = deconvolve(y, penalty=1)
           return s 
        
        PredOasis = predictionOasis()

        increments = np.arange(1, 102, 20)
        listeR1 = []
        listeR2 = []
        listeDev1 = []
        listeDev2 = []

        for i in increments:
           SegmentCascade = sliding(PredCascade, window, i)
           SegmentOasis = sliding(PredOasis, window, i)
           SegmentSpike = sliding(spikes, window, i)
           SegmentSpikeRate = [[(100 / i) * x for x in sous_liste] for sous_liste in SegmentSpike]
           
           r1 = calc_R(SegmentSpikeRate, SegmentCascade, addMoyenne=False)
           r2 = calc_R(SegmentSpikeRate, SegmentOasis,addMoyenne=False)
           std_Cascade = sem(r1, ddof=1, nan_policy="omit") 
           std_Oasis = sem(r2, ddof=1, nan_policy="omit")
           
           listeDev1.append(std_Cascade)
           listeDev2.append(std_Oasis)
           
           Rm1 = calc_R(SegmentSpikeRate, SegmentCascade, addMoyenne=True)
           listeR1.append(Rm1)
           Rm2 = calc_R(SegmentSpikeRate, SegmentOasis, addMoyenne=True)
           listeR2.append(Rm2)
        
        temps = increments/100

        SupCascade, infCascade = incertitudeGraph(listeR1, listeDev1)
        SupOasis, infOasis = incertitudeGraph(listeR2, listeDev2)
        
        SupCascade = np.array(SupCascade)
        infCascade = np.array(infCascade)
        SupOasis = np.array(SupOasis)
        infOasis = np.array(infOasis)

        df_out = pd.DataFrame({"Incrément": temps, "R Cascade": listeR1, "Borne supCascade": SupCascade, "Borne infCascade":infCascade, 
                               "R Oasis": listeR2, "Borne supOasis": SupOasis, "Borne infOasis":infOasis})
        fname = os.path.basename(file_path1)
        out_filename = f"processed_{fname}"           
        out_path = os.path.join(output_folder, out_filename)
        df_out.to_csv(out_path, index=False)



for folder2 in Folder_outname:
    full_path = os.path.join(input_folder, folder2)
    if full_path not in sys.path:
        sys.path.append(full_path)
    pattern2 = os.path.join(full_path, "*.csv")
    output_name = f"{folder2}_processed"
    output_folder = os.path.join(input_folder, output_name)
    os.makedirs(output_folder, exist_ok=True)

    FolderAverage_out.append(output_name)
    increments = np.arange(0.01, 1.02, 0.20)
    increments = np.round(increments, 2)
    dicoRCascade    = {i: [] for i in increments}
    dicoUpCascade   = {i: [] for i in increments}
    dicoDownCascade = {i: [] for i in increments}
       
    dicoROasis      = {i: [] for i in increments}
    dicoUpOasis     = {i: [] for i in increments}
    dicoDownOasis   = {i: [] for i in increments}

    for file_path2 in glob.glob(pattern2):
       df = pd.read_csv(file_path2)
       Smooth = df["Incrément"].to_numpy()
       Smooth = np.round(Smooth, 2)

       RCascade = df["R Cascade"].to_numpy()
       SupCascade = df["Borne supCascade"].to_numpy()
       infCascade = df["Borne infCascade"].to_numpy()

       ROasis = df["R Oasis"].to_numpy()
       SupOasis = df["Borne supOasis"].to_numpy()
       infOasis = df["Borne infOasis"].to_numpy()

       dicoRCascade, dicoUpCascade, dicoDownCascade = ajoutdico(Smooth, RCascade, SupCascade, infCascade, dicoRCascade, dicoUpCascade, dicoDownCascade)
       dicoROasis, dicoUpOasis, dicoDownOasis = ajoutdico(Smooth, ROasis, SupOasis, infOasis, dicoROasis, dicoUpOasis, dicoDownOasis)

       listeRCascade = []
       listeROasis = []

       ErrUpCascade = []
       ErrDownCascade = []

       ErrUpOasis = []
       ErrdownOasis = []

       RmCascade = MoyenneValue(dicoRCascade, listeRCascade)
       RmOasis = MoyenneValue(dicoROasis, listeROasis)
       
       StdUpCascade = MoyenneValue(dicoUpCascade, ErrUpCascade)
       StdupOasis = MoyenneValue(dicoUpOasis, ErrUpOasis)

       StdDownCascade = MoyenneValue(dicoDownCascade, ErrDownCascade)
       StdDownOasis = MoyenneValue(dicoDownOasis, ErrdownOasis)

    a = ({"Incrément": increments, "R Cascade": RmCascade, "Borne supC": StdUpCascade, "Borne infC": StdDownCascade,
                               "R Oasis": RmOasis, "Borne supO": StdupOasis, "Borne infO": StdDownOasis})
    df_out2 = pd.DataFrame.from_dict(a, orient="index")
    df_out2 = df_out2.transpose()
    fname = os.path.basename(folder2)
    out_filename = f"Moyenner_{fname}.csv"
    out_path = os.path.join(output_folder, out_filename)
    df_out2.to_csv(out_path, index=False)


for finalFolder in FolderAverage_out:
    full_path3 = os.path.join(input_folder, finalFolder)
    if full_path3 not in sys.path:
        sys.path.append(full_path3)

    pattern3 = os.path.join(full_path3, "*.csv")
    for file_path3 in glob.glob(pattern3):
       df3 = pd.read_csv(file_path3)

       results["Cascade"][finalFolder] = {
          "R":   df3["R Cascade"].to_numpy(),
          "sup": df3["Borne supC"].to_numpy(),
          "inf": df3["Borne infC"].to_numpy()
         }
       results["Oasis"][finalFolder] = {
          "R":   df3["R Oasis"].to_numpy(),
          "sup": df3["Borne supO"].to_numpy(),
          "inf": df3["Borne infO"].to_numpy()
         }

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# top plot
ax1.plot(increments, results["Cascade"][FolderAverage_out[0]]["R"], color="blue", label=f"Gcamp6")
ax1.fill_between(increments, results["Cascade"][FolderAverage_out[0]]["inf"], results["Cascade"][FolderAverage_out[0]]["sup"], alpha=0.2, color="blue")
ax1.plot(increments,results["Cascade"][FolderAverage_out[1]]["R"], color="red",label="Gcamp7")
ax1.fill_between(increments, results["Cascade"][FolderAverage_out[1]]["inf"], results["Cascade"][FolderAverage_out[1]]["sup"], alpha=0.2, color="red")
ax1.plot(increments, results["Cascade"][FolderAverage_out[2]]["R"], color="green", label=f"Gcamp8")
ax1.fill_between(increments, results["Cascade"][FolderAverage_out[2]]["inf"], results["Cascade"][FolderAverage_out[2]]["sup"], alpha=0.2, color="green")
ax1.set_title("Cascade prediction on 3 files")
ax1.set_xlabel("Time increments [s]")
ax1.set_ylabel("Pearson correlation coefficient (R)")
ax1.legend(loc="best")


# bottom plot
ax2.plot(increments, results["Oasis"][FolderAverage_out[0]]["R"], color="blue", label=f"Gcamp6")
ax2.fill_between(increments, results["Oasis"][FolderAverage_out[0]]["inf"], results["Oasis"][FolderAverage_out[0]]["sup"], alpha=0.2, color="blue")
ax2.plot(increments,results["Oasis"][FolderAverage_out[1]]["R"], color="red",label="Gcamp7")
ax2.fill_between(increments, results["Oasis"][FolderAverage_out[1]]["inf"], results["Oasis"][FolderAverage_out[1]]["sup"], alpha=0.2, color="red")
ax2.plot(increments, results["Oasis"][FolderAverage_out[2]]["R"], color="green", label=f"Gcamp8")
ax2.fill_between(increments, results["Oasis"][FolderAverage_out[2]]["inf"], results["Oasis"][FolderAverage_out[2]]["sup"], alpha=0.2, color="green")
ax2.set_title("Oasis prediction on 3 files")
ax2.legend(loc="best")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Pearson correlation coefficient (R)")


plt.xlim()
plt.legend()
plt.tight_layout()
plt.show()