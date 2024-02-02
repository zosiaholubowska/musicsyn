import slab
import pandas
import numpy


def subject_data(subject, file, melody_file):
    """
    This is to create a dataframe with results and the sequence


    file_name = file.name
    RCX_files = slab.ResultsFile.read_file(f'C:\projects\musicsyn\musicsyn\Results\{subject}\{file_name}')

    timestamps = [float(list(d.keys())[0]) for d in RCX_files]
    frequencies = [list(d.values())[0] for d in RCX_files]
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



    seq = pandas.read_csv(f'C:\projects\musicsyn\musicsyn\Results/{subject}/{subject}_seq_{melody_file}')

    RCX_files = seq.join(freq["Responses"])
    RCX_files = RCX_files.join(freq["Frequency"])
    RCX_files = RCX_files.join(freq["Timestamp"])



    RCX_files = RCX_files.drop(columns=["idx"])
    RCX_files = RCX_files.loc[:, ~RCX_files.columns.str.contains("^Unnamed")]
    RCX_files = RCX_files[["Frequency", "Timestamp", "boundary", "sequence", "cue", "Responses", ]]
    RCX_files = RCX_files.rename(
        columns={
            "boundary": "Boundary",
            "sequence": "Location_change",
            "cue": "Visual_cue",
        }
    )

    RCX_files.to_csv(f'C:\projects\musicsyn\musicsyn\Results\{subject}\{subject}_data_{melody_file}')

    """

    print('Good luck!')
