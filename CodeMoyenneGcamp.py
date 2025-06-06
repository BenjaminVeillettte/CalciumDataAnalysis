import numpy as np 
import matplotlib.pyplot as plt
import os 
import pandas as pd

folder_path = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO", 
    "Analysis",
    "Philippe",
    "Gcamp8"
)




increments = np.arange(0.01, 2.01, 0.05)
increments = np.round(increments, 2)
dicoRCascade    = {i: [] for i in increments}
dicoUpCascade   = {i: [] for i in increments}
dicoDownCascade = {i: [] for i in increments}

dicoROasis      = {i: [] for i in increments}
dicoUpOasis     = {i: [] for i in increments}
dicoDownOasis   = {i: [] for i in increments}


def ajoutdico(IncrémentDico, Array2, Array3, Array4, dicoR, dicoUp, dicoDown):
    for temps, R , BornUp, BornDo in zip(IncrémentDico, Array2, Array3, Array4):
        dicoR[(temps)].append(R)
        dicoUp[(temps)].append(BornUp)
        dicoDown[(temps)].append(BornDo)
    return dicoR, dicoUp, dicoDown


for i in range(1, 4):
    file_name = f"PrécisionRdeCascade{i}.csv"
    file_path = os.path.join(folder_path, file_name)

    # Lecture en DataFrame pandas à partir de data_start_index
    df = pd.read_csv(
        file_path,
    )

    #Incrément, R de Cascade,Borne sup,Borne inf
    Smooth = df["Incrément"].to_numpy()
    Smooth = np.round(Smooth, 2)
    RCascade = df[" R de Cascade"].to_numpy()
    SupCascade = df["Borne sup"].to_numpy()
    infCascade = df["Borne inf"].to_numpy()

    file_name2 = f"PrécisionRdeOasis{i}.csv"
    file_path2 = os.path.join(folder_path, file_name2)
    # Lecture en DataFrame pandas à partir de data_start_index
    df2 = pd.read_csv(
        file_path2,
    )
    Smooth = df2["Incrément"].to_numpy()
    Smooth = np.round(Smooth, 2)
    ROasis = df2[" R d'Oasis"].to_numpy()
    SupOasis = df2["Borne sup"].to_numpy()
    infOasis = df2["Borne inf"].to_numpy()


    dicoRCascade, dicoUpCascade, dicoDownCascade = ajoutdico(Smooth, RCascade, SupCascade, infCascade, dicoRCascade, dicoUpCascade, dicoDownCascade)
    dicoROasis, dicoUpOasis, dicoDownOasis = ajoutdico(Smooth, ROasis, SupOasis, infOasis, dicoROasis, dicoUpOasis, dicoDownOasis)


listeRCascade = []
listeROasis = []

ErrUpCascade = []
ErrDownCascade = []

ErrUpOasis = []
ErrdownOasis = []

def MoyenneValue(dico, liste):
    for i in np.arange(0.01, 2.01, 0.05):
        i = np.round(i, 2)
        Moyenne = np.nanmean(dico[i])
        liste.append(Moyenne)
    return liste

RmCascade = MoyenneValue(dicoRCascade, listeRCascade)
RmOasis = MoyenneValue(dicoROasis, listeROasis)

StdUpCascade = MoyenneValue(dicoUpCascade, ErrUpCascade)
StdupOasis = MoyenneValue(dicoUpOasis, ErrUpOasis)

StdDownCascade = MoyenneValue(dicoDownCascade, ErrDownCascade)
StdDownOasis = MoyenneValue(dicoDownOasis, ErrdownOasis)

plt.fill_between(increments, StdDownCascade, StdUpCascade, alpha=0.2, color="red")
plt.fill_between(increments, StdDownOasis, StdupOasis, alpha=0.2, color="blue")
plt.plot(increments, RmCascade, label="R de Cascade", color="red")
plt.plot(increments, RmOasis, label="R de oasis", color="blue")
plt.title("Précision en moyenne sur Gcamp6")
plt.xlabel("Incréments temps [s]")
plt.ylabel("Coefficient R")
plt.legend()

plt.show()

df3 = pd.DataFrame({"Incrément": increments, " R de Cascade": RmCascade, "Borne sup": StdUpCascade, "Borne inf": StdDownCascade})
df3.to_csv('PrécisionCascadeGcamp.csv',index=False)

df4 = pd.DataFrame({"Incrément": increments, " R de Cascade": RmOasis, "Borne sup": StdupOasis, "Borne inf":StdDownOasis})
df4.to_csv('PrécisionOasisGcamp.csv',index=False)