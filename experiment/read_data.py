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
    condition = data[1]['1']

    timestamps = [float(list(d.keys())[0]) for d in data]
    frequencies = [list(d.values())[0] for d in data]

    df = pandas.DataFrame({'time': timestamps, 'frequencies': frequencies})
    df1 = df.iloc[2:]
    df1.loc[:, "channel"] = 0
    df1.loc[:, "answer"] = 0
    df1.loc[:, "prec_time"] = 0

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
    df_filtered["condition"] = condition

    # Find the index of the year
    year_index = file_name.find('202')

    # Extracting date and hour
    date = file_name[year_index: year_index+10]
    hour = file_name[year_index+11 : -4]

    # Add date and hour columns to the DataFrame
    df_filtered['date'] = date
    df_filtered['hour'] = hour

    seq = pandas.read_csv(
        path + f"/Results/{subject}/{subject}_seq_{stimulus}_{condition}.csv"
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
        path + f"/Results/{subject}/{subject}_data_{stimulus}_{condition}.csv",
    )

    df_dprime = df_filtered.copy()


    for index, row in df_dprime.iterrows():
        if row['visual_cue'] == 1:
            # Check for response in the window of 1.1 seconds
            window_condition = (df_dprime['time'] >= row['time']) & (df_dprime['time'] <= row['time'] + 1.1)
            found_index = df_dprime.index[window_condition & (df_dprime['answer'] == 1)].tolist()

            if any((df_dprime.loc[window_condition, 'answer'] == 1)):
                # If there is a response in the window, set the Response value for the current row to 1
                df_dprime.at[index, 'answer'] = 1
                df_dprime.at[index, 'prec_time'] = df_dprime['prec_time'][found_index[0]]

    df_dprime = df_dprime.loc[df_dprime["visual_cue"]==1]
    df_dprime['signal_theory'] = ''

    df_dprime.loc[(df_dprime['loc_change'] == 1) & (df_dprime['answer'] == 1), 'signal_theory'] = 'hit'
    df_dprime.loc[(df_dprime['loc_change'] == 1) & (df_dprime['answer'] == 0), 'signal_theory'] = 'miss'
    df_dprime.loc[(df_dprime['loc_change'] == 0) & (df_dprime['answer'] == 0), 'signal_theory'] = 'corr'
    df_dprime.loc[(df_dprime['loc_change'] == 0) & (df_dprime['answer'] == 1), 'signal_theory'] = 'fa'

    df_dprime_grouped = df_dprime.groupby(['subject','signal_theory']).size().unstack(fill_value=0)

    if 'fa' not in df_dprime_grouped.columns:
        df_dprime_grouped['fa'] = 0

    if 'hit' not in df_dprime_grouped.columns:
        df_dprime_grouped['hit'] = 0

    if 'miss' not in df_dprime_grouped.columns:
        df_dprime_grouped['miss'] = 0

    if 'corr' not in df_dprime_grouped.columns:
        df_dprime_grouped['corr'] = 0

    df_dprime_grouped.reset_index(inplace=True)

    df_dprime_grouped.at[0, "hit_rate"] = df_dprime_grouped.at[0, "hit"] / (df_dprime_grouped.at[0, "hit"] + df_dprime_grouped.at[0, "miss"])
    df_dprime_grouped.at[0, "false_alarm_rate"] = df_dprime_grouped.at[0, "fa"] / (
                df_dprime_grouped.at[0, "fa"] + df_dprime_grouped.at[0, "corr"])
    hr = df_dprime_grouped.at[0, "hit_rate"]
    far = df_dprime_grouped.at[0, "false_alarm_rate"]
    dp = hr-far
    print(f"Current hit rate: {hr}")
    print(f"Current d-prime: {dp}")


