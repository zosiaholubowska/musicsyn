import slab
import pandas
import numpy
import os

path = os.getcwd()
def subject_data(subject, file, melody_file):
    """
    This is to create a dataframe with results and the sequence
    """

    file_name = file.name
    data = slab.ResultsFile.read_file(
          path + f"/Results/{subject}/{file_name}"
        )

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



    seq = pandas.read_csv(
      path + f"/Results/{subject}/{subject}_seq_{melody_file}"
    )

    data = seq.join(freq["Responses"])
    data = data.join(freq["Frequency"])
    data = data.join(freq["Timestamp"])



    data = data.drop(columns=["idx"])
    data = data.loc[:, ~data.columns.str.contains("^Unnamed")]
    data = data[["Frequency", "Timestamp", "boundary", "sequence", "cue", "Responses", ]]
    data = data.rename(
        columns={
            "boundary": "Boundary",
            "sequence": "Location_change",
            "cue": "Visual_cue",
        }
    )

    data.to_csv(
      path +  f"/Results/{subject}/{subject}_data_{melody_file}",
    )
