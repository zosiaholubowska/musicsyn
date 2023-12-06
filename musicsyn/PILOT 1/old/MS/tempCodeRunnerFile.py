import slab
import pandas as pd
import json
import numpy as np
import os

os.chdir("/Users/zofiaholubowska/Documents/PhD/experiment/musicsyn/Results/MS/")
data = slab.ResultsFile.read_file("MS_2023-10-12-12-43-26.txt")


timestamps = [float(list(d.keys())[0]) for d in data]
frequencies = [list(d.values())[0] for d in data]
responses = np.zeros_like(frequencies)


df = pd.DataFrame(
    {"Timestamp": timestamps, "Frequency": frequencies, "Responses": responses}
)

print(df["Frequency"].dtype)

answers = df[df["Frequency"].astype(str).str.isalpha()]
ans = answers["Timestamp"].tolist()

freq = df[~df["Frequency"].astype(str).str.isalpha()]

freq = freq.reset_index(drop=True)

print(len(freq))