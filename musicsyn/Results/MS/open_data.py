import slab
import pandas
import json
import numpy
import os


def subject_data(subject, file):
    os.chdir(
        f"/Users/zofiaholubowska/Documents/PhD/experiment/musicsyn/Results/{subject}/"
    )
    data = slab.ResultsFile.read_file(file)

    timestamps = [float(list(d.keys())[0]) for d in data]
    frequencies = [list(d.values())[0] for d in data]
    responses = numpy.zeros_like(frequencies)

    df = pandas.DataFrame(
        {"Timestamp": timestamps, "Frequency": frequencies, "Responses": responses}
    )

    answers = df[df["Frequency"].astype(str).str.isalpha()]
    ans = answers["Timestamp"].tolist()

    freq = df[~df["Frequency"].astype(str).str.isalpha()]

    freq = freq.reset_index(drop=True)

    for row in freq.index[:-1]:
        start = freq.loc[row][0]
        end = freq.loc[row + 1][0]
        if row < len(freq):
            if any(start <= x < end for x in ans if isinstance(x, (int, float))):
                freq.at[row, "Responses"] = 1
            else:
                freq.at[row, "Responses"] = 0

    start_last = freq.iloc[-1]["Timestamp"]
    if any(
        start_last <= x < (start_last + 1.5) for x in ans if isinstance(x, (int, float))
    ):
        freq.at[len(freq) - 1, "Responses"] = 1
    else:
        freq.at[len(freq) - 1, "Responses"] = 0

    seq = pandas.read_csv(f"{subject}_seq.csv")

    final = seq.join(freq["Responses"])
    final = final.join(freq["Frequency"])
    final = final.drop(columns=["idx"])
    print(final)
