import slab
import pandas
import numpy
import os

def read_data(subject, file_name):
    """
    This is to create a dataframe with results and the sequence
    """
    path = os.getcwd()
    data = slab.ResultsFile.read_file(path + f"/Results/{subject}/{file_name}")

    stimulus = data[0]['0'][:-4]

    timestamps = [float(list(d.keys())[0]) for d in data]
    frequencies = [list(d.values())[0] for d in data]

    df = pandas.DataFrame({'time': timestamps, 'frequencies': frequencies})
    df1 = df.iloc[1:]
    df1["channel"] = 0
    df1["answer"] = 0
    df1["prec_time"] = 0

    for idx in df1.index:
        if (df1["frequencies"][idx] == 17.5) or (df1["frequencies"][idx] == -17.5):
            df1["channel"][idx + 1] = df1["frequencies"][idx]
        elif (df1["frequencies"][idx] == "p"):
            df1["answer"][idx - 1] = 1
            df1["prec_time"][idx - 1] = df1["time"][idx]
        elif (df1["frequencies"][idx] == 0.0):
            df1["channel"][idx + 1] = 1

    df_filtered = df1[~df['frequencies'].isin([1, 0.0, 23, 17.5, -17.5, 'p'])]
    df_filtered.reset_index(drop=True, inplace=True)  # reset the index

    df_filtered["stimulus"] = stimulus
    df_filtered["subject"] = subject

    seq = pandas.read_csv(
        path + f"/Results/{subject}/{subject}_seq_{stimulus}.csv"
    )

    df_filtered['visual_cue'] = 0
    df_filtered['boundary'] = 0
    df_filtered['loc_change'] = 0
    df_filtered['changable_notes'] = 0

    df_filtered['visual_cue'] = seq['cue']
    df_filtered['boundary'] = seq['boundary']
    df_filtered['loc_change'] = seq['sequence']
    df_filtered['changable_notes'] = seq['changable_notes']

    for idx in df_filtered.index:
        if df_filtered["channel"][idx] == 0:
            df_filtered["channel"][idx] = df_filtered["channel"][idx - 1]

    df_filtered.to_csv(
        path + f"/Results/{subject}/{subject}_data_{stimulus}.csv",
    )
