import slab
import pandas
import numpy
import os

path = os.getcwd()
def subject_data(subject, file):
    """
    This is to create a dataframe with results and the sequence
    """

    file_name = file.name

    data = slab.ResultsFile.read_file(
        path + f"/musicsyn/Results/{subject}/{file_name}"
    )

    stimulus = data[0]['0'][:-4]

    initial_channel = data[1]['0']

    timestamps = [float(list(d.keys())[0]) for d in data]
    frequencies = [list(d.values())[0] for d in data]

    df = pandas.DataFrame({'time': timestamps, 'frequencies': frequencies})
    df1 = df.iloc[1:]
    df1["channel"] = 0
    df1["answer"] = 0
    df1["prec_time"] = 0


    for idx in df1.index:
        if (df1["frequencies"][idx] == 1) or (df1["frequencies"][idx] == 9) or (df1["frequencies"][idx] == 18):
            df1["channel"][idx+1] = df1["frequencies"][idx]
        elif (df1["frequencies"][idx] == "p"):
            df1["answer"][idx - 1] = 1
            df1["prec_time"][idx - 1] = df1["time"][idx]

    df_filtered = df1[~df['frequencies'].isin([1, 9, 18, 'p'])]
    df_filtered.reset_index(drop=True, inplace=True) #reset the index

    df_filtered["stimulus"] = stimulus
    df_filtered["subject"] = subject

    seq = pandas.read_csv(
          path + f"/musicsyn/Results/{subject}/{subject}_seq_{stimulus}.csv"
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
            df_filtered["channel"][idx] = df_filtered["channel"][idx-1]

    df_filtered.to_csv(
        path + f"/musicsyn/Results/{subject}/{subject}_data_{stimulus}_test.csv",
    )
