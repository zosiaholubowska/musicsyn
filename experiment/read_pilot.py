import slab
import pandas
import os

path = os.getcwd()
subjects = [f for f in os.listdir(f"{path}/Results") if f.startswith("p")]

#results_df = pandas.DataFrame(columns=['time', 'frequencies', 'channel', 'answer', 'prec_time', 'stimulus', 'subject', 'visual_cue','boundary', 'loc_change', 'changable_notes'])
#results_df = pandas.read_csv("results_df.csv")
def subject_data(subject, file):
    """
    This is to create a dataframe with results and the sequence
    """
    #results_df = pandas.read_csv("results_df.csv")

    data = slab.ResultsFile.read_file(
        path + f"/Results/{subject}/{file}"
    )

    stimulus = data[0]['0'][:-4]

    timestamps = [float(list(d.keys())[0]) for d in data]
    frequencies = [list(d.values())[0] for d in data]

    df = pandas.DataFrame({'time': timestamps, 'frequencies': frequencies})
    df1 = df.iloc[1:]
    df1["channel"] = 0
    df1["answer"] = 0
    df1["prec_time"] = 0

     #for idx in df1.index:
     #   if type(df1["frequencies"][idx]) == str:
     #        if (type(df1["frequencies"][idx-1]) != str) and (type(df1["frequencies"][idx]) == str) and (type(df1["frequencies"][idx+1]) == str):
     #            df1["answer"][idx] = 1

     #df1 = df1.drop(df1[(df1["answer"] == 0) & ((df1["frequencies"] == 'p'))].index)
     #time1 = 0
     #time2 = 0
     #for idx in df1.index:

     #    if df1["frequencies"][idx] == 'p':
     #        time2 = df1["time"][idx]
     #        if (time2 - time1 < 0.2):
     #            df1["answer"][idx] = 100

     #    time1 = time2

     #df1 = df1.drop(df1[(df1["answer"] == 100) & ((df1["frequencies"] == 'p')) ].index)
    for idx in df1.index:
        if (df1["frequencies"][idx] == 17.5) or (df1["frequencies"][idx] == -17.5):
            df1["channel"][idx+1] = df1["frequencies"][idx]
        elif (df1["frequencies"][idx] == "p"):
            df1["answer"][idx - 1] = 1
            df1["prec_time"][idx - 1] = df1["time"][idx]
        elif (df1["frequencies"][idx] == 0.0):
            df1["channel"][idx + 1] = 1

    df_filtered = df1[~df['frequencies'].isin([1, 0.0, 23, 17.5, -17.5, 'p'])]
    df_filtered.reset_index(drop=True, inplace=True) #reset the index

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
            df_filtered["channel"][idx] = df_filtered["channel"][idx-1]

    #temp = pandas.concat([results_df, df_filtered], axis=0)
    #temp.to_csv("results_df.csv")

    melody = pandas.read_csv(
        f"/Users/zofiaholubowska/Documents/PhD/3_experiment/musicsyn/stimuli/{stimulus}.csv"
    )

    df_filtered = pandas.concat([df_filtered, melody['duration']], axis=1)

    df_filtered.to_csv(
        path + f"/Results/{subject}/{subject}_data_{stimulus}.csv",
    )

for subject in subjects:
    folder_path = f"{path}/Results/{subject}"

    # Get all file names in the folder
    file_names = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    for file in file_names:
        print(file)
        subject_data(subject, file)



folder_path = f"{path}/Results/{subject}"

# Get all file names in the folder
file_names = [f for f in os.listdir(folder_path) if '_data_' in f]


for file in file_names:
    print(file)
    data = pandas.read_csv(
        f"{path}/Results/p05/{file}"
    )

    stimulus = data.iloc[0]['stimulus']

    melody = pandas.read_csv(
        f"/Users/zofiaholubowska/Documents/PhD/3_experiment/musicsyn/stimuli/{stimulus}.csv"
    )

    data = pandas.concat([data, melody['duration']], axis=1)

    data.to_csv(
        path + f"/Results/{subject}/{subject}_data_{stimulus}.csv",
    )


