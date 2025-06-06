import numpy as np 
import matplotlib.pyplot as plt
import os 
import pandas as pd

folder_path = os.path.join(
    os.path.expanduser("~"), 
    "OneDrive - Cégep de Shawinigan", 
    "Bureau", 
    "Stage CERVO", 
    "Data",
)




replicates  = [6, 7, 8]           
increments  = np.arange(0.01, 2.01, 0.05)               

# 2. Préparation des conteneurs
results = {
    "Cascade": {},
    "Oasis":   {}
}

# 3. Chargement des données
for i in replicates:
    # --- Cascade ---
    fc = f"PrécisionCascadeGcamp{i}.csv"
    dfc = pd.read_csv(os.path.join(folder_path, fc))
    inc = np.round(dfc["Incrément"].to_numpy(), 2)

    results["Cascade"][i] = {
        "R":   dfc[" R de Cascade"].to_numpy(),
        "sup": dfc["Borne sup"].to_numpy(),
        "inf": dfc["Borne inf"].to_numpy()
    }

    # --- Oasis ---
    fo = f"PrécisionOasisGcamp{i}.csv"
    dfo = pd.read_csv(os.path.join(folder_path, fo))
    results["Oasis"][i] = {
        "R":   dfo[" R de Cascade"].to_numpy(),
        "sup": dfo["Borne sup"].to_numpy(),
        "inf": dfo["Borne inf"].to_numpy()
    }




fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# top plot
ax1.plot(increments, results["Cascade"][6]["R"], color="blue", label=f"Gcamp6")
ax1.fill_between(increments, results["Cascade"][6]["inf"], results["Cascade"][6]["sup"], alpha=0.2, color="blue")
ax1.plot(increments,results["Cascade"][7]["R"], color="red",label="Gcamp7")
ax1.fill_between(increments, results["Cascade"][7]["inf"], results["Cascade"][7]["sup"], alpha=0.2, color="red")
ax1.plot(increments, results["Cascade"][8]["R"], color="green", label=f"Gcamp8")
ax1.fill_between(increments, results["Cascade"][8]["inf"], results["Cascade"][8]["sup"], alpha=0.2, color="green")
ax1.set_title("Cascade prediction on 3 files")
ax1.set_xlabel("Time increments [s]")
ax1.set_ylabel("Pearson correlation coefficient (R)")
ax1.legend(loc="best")



# bottom plot
ax2.plot(increments, results["Oasis"][6]["R"], color="blue", label=f"Gcamp6")
ax2.fill_between(increments, results["Oasis"][6]["inf"], results["Oasis"][6]["sup"], alpha=0.2, color="blue")
ax2.plot(increments,results["Oasis"][7]["R"], color="red",label="Gcamp7")
ax2.fill_between(increments, results["Oasis"][7]["inf"], results["Oasis"][7]["sup"], alpha=0.2, color="red")
ax2.plot(increments, results["Oasis"][8]["R"], color="green", label=f"Gcamp8")
ax2.fill_between(increments, results["Oasis"][8]["inf"], results["Oasis"][8]["sup"], alpha=0.2, color="green")
ax2.set_title("Oasis prediction on 3 files")
ax2.legend(loc="best")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Pearson correlation coefficient (R)")


plt.xlim()
plt.legend()
plt.tight_layout()
plt.show()
