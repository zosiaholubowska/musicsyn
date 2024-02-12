import pandas
import numpy as np
import os

"""
We can use the file of one of the stimuli to test the function
"""
"""
file = ("stim_min_3_a.csv")
os.chdir('C://projects//musicsyn/stimuli')

def read_melody(file):
    score_data = pandas.read_csv(file, sep=";")  # open the csv file with notes
    onsets = score_data.onset_sec.to_list()  # list of onsets of consecutive notes
    frequencies = score_data.freq.to_list()  # frequencies of consecutive notes
    durations = score_data.duration.to_list()  # note durations
    boundaries = score_data.boundary.to_list()  # 0 or 1 indication if the note is the beginning of a new phrase
    changable_notes = score_data.changable_note.to_list() #if at note is possible to change direction
    onsets.append(onsets[-1] + durations[-1] + 0.1)  # I add a dummy note here
    durations.append(0.1)  # I add a dummy note here
    return onsets, frequencies, durations, boundaries, changable_notes


onsets, frequencies, durations, boundaries, changable_notes = read_melody(file)

"""
path = 'C:\\projects\\musicsyn'


def balanced_sequence(boundaries, changable_notes, subject, melody_file, p, condition):
    """
    Here we define a df, to which we will append all the sequence values

    """
    boundaries_df = pandas.DataFrame(np.column_stack([boundaries, changable_notes]),
                                     columns=['boundary', 'changable_notes'])
    boundaries_df['idx'] = range(len(boundaries_df))



    n_boundaries = sum(boundaries)  # number of boundaries in stimulus
    n_changable = sum(changable_notes)
    # number of changable notes in stimulus
    n_changes = round(p * n_changable)  # 20% of notes has to have a location/cue change

    # This part is to calculate the sequence for location changes
    temp_arr = np.array([0] * round((0.4 * n_boundaries)) + [1] * round((0.6 * n_boundaries)))
    np.random.shuffle(temp_arr)  # we give 60% chance to location change, when there is a phrase boundary
    seq_boundaries = temp_arr.tolist()

    n_boundaries_loc = sum(seq_boundaries)
    p_nbound = (n_changes - n_boundaries_loc) / n_changable

    temp_arr = np.array([0] * round(((1 - p_nbound) * (n_changable - n_boundaries))) + [1] * round(
        p_nbound * (n_changable - n_boundaries)))
    np.random.shuffle(temp_arr)  # we give 60% chance to location change, when there is a phrase boundary
    seq_noboundaries = temp_arr.tolist()

    no_boundaries = boundaries_df.loc[boundaries_df['boundary'] == 0]

    changable_no_boundaries = no_boundaries.loc[no_boundaries['changable_notes'] == 1]

    if len(seq_noboundaries) > len(changable_no_boundaries):
        seq_noboundaries = seq_noboundaries[:len(changable_no_boundaries)]
    elif len(seq_noboundaries) < len(changable_no_boundaries):
        seq_noboundaries += [1] + [0] * (len(changable_no_boundaries) - 1)

    # Ensure the length is exactly the same
    seq_noboundaries = seq_noboundaries[:len(changable_no_boundaries)]

    changable_no_boundaries.insert(1, 'sequence', seq_noboundaries)

    unchangable_no_boundaries = no_boundaries.loc[no_boundaries['changable_notes'] == 0]
    unchangable_no_boundaries['sequence'] = 0

    sequence = [unchangable_no_boundaries, changable_no_boundaries]  # we append two sequences into one df
    sequence = pandas.concat(sequence)
    temp_no_seq = sequence.sort_values(by='idx', ascending=True)

    yes_boundaries = boundaries_df.loc[boundaries_df['boundary'] == 1]

    if len(seq_boundaries) > len(yes_boundaries):
        seq_boundaries = seq_boundaries[:len(yes_boundaries)]
    elif len(seq_boundaries) < len(yes_boundaries):
        seq_boundaries.append(1)

    yes_boundaries.insert(1, 'sequence', seq_boundaries)

    sequence = [yes_boundaries, temp_no_seq]  # we append two sequences into one df
    sequence = pandas.concat(sequence)
    temp_seq = sequence.sort_values(by='idx', ascending=True)

    # This part is to compute the prompt for the response - visual cue

    yes_boundaries_change = temp_seq.loc[(temp_seq['sequence'] == 1) & (temp_seq['boundary'] == 1)]

    temp_arr = np.array(
        [0] * round((0.5 * len(yes_boundaries_change))) + [1] * round((0.5 * len(yes_boundaries_change))))
    np.random.shuffle(temp_arr)  # we give 60% chance to location change, when there is a phrase boundary
    seq_boundaries_change_cues = temp_arr.tolist()

    if len(seq_boundaries_change_cues) > len(yes_boundaries_change):
        seq_boundaries_change_cues = seq_boundaries_change_cues[:len(yes_boundaries_change)]
    elif len(seq_boundaries_change_cues) < len(yes_boundaries_change):
        seq_boundaries_change_cues.append(1)
    yes_boundaries_change.insert(1, 'cue', seq_boundaries_change_cues)

    yes_boundaries_nochange = yes_boundaries.loc[temp_seq['sequence'] == 0]

    temp_arr = np.array(
        [0] * round((0.5 * len(yes_boundaries_nochange))) + [1] * round((0.5 * len(yes_boundaries_nochange))))
    np.random.shuffle(temp_arr)  # we give 60% chance to location change, when there is a phrase boundary
    seq_boundaries_nochange_cues = temp_arr.tolist()

    if len(seq_boundaries_nochange_cues) > len(yes_boundaries_nochange):
        seq_boundaries_nochange_cues = seq_boundaries_nochange_cues[:len(yes_boundaries_nochange)]
    elif len(seq_boundaries_nochange_cues) < len(yes_boundaries_nochange):
        seq_boundaries_nochange_cues.append(1)
    yes_boundaries_nochange.insert(1, 'cue', seq_boundaries_nochange_cues)

    yes_boundaries = [yes_boundaries_change, yes_boundaries_nochange]  # we append two sequences into one df
    yes_boundaries = pandas.concat(yes_boundaries)
    yes_boundaries = yes_boundaries.sort_values(by='idx', ascending=True)

    n_boundary_cues = sum(yes_boundaries["cue"])

    p_nbound_cues = (n_changes - n_boundary_cues) / n_changable

    no_boundaries_unchangable = temp_seq.loc[(temp_seq['boundary'] == 0) & (temp_seq['changable_notes'] == 0)]
    no_boundaries_unchangable['cue'] = 0
    no_boundaries_changable = temp_seq.loc[(temp_seq['boundary'] == 0) & (temp_seq['changable_notes'] == 1)]

    no_boundaries_change = no_boundaries_changable.loc[no_boundaries_changable['sequence'] == 1]

    temp_arr = np.array([0] * round(((1 - p_nbound_cues - 0.5) * len(no_boundaries_change))) + [1] * round(
        ((p_nbound_cues + 0.5) * len(no_boundaries_change))))
    np.random.shuffle(temp_arr)  # we give 60% chance to location change, when there is a phrase boundary
    seq_boundaries_change_cues = temp_arr.tolist()

    if len(seq_boundaries_change_cues) > len(no_boundaries_change):
        seq_boundaries_change_cues = seq_boundaries_change_cues[:len(no_boundaries_change)]
    no_boundaries_change.insert(1, 'cue', seq_boundaries_change_cues)

    no_boundaries_nochange = no_boundaries_changable.loc[no_boundaries_changable['sequence'] == 0]

    temp_arr = np.array([0] * round(((1 - p_nbound_cues + 0.1) * len(no_boundaries_nochange))) + [1] * round(
        ((p_nbound_cues - 0.1) * len(no_boundaries_nochange))))
    np.random.shuffle(temp_arr)  # we give 60% chance to location change, when there is a phrase boundary
    seq_boundaries_nochange_cues = temp_arr.tolist()

    if len(seq_boundaries_nochange_cues) > len(no_boundaries_nochange):
        seq_boundaries_nochange_cues = seq_boundaries_nochange_cues[:len(no_boundaries_nochange)]
    no_boundaries_nochange.insert(1, 'cue', seq_boundaries_nochange_cues)

    no_boundaries = [no_boundaries_change, no_boundaries_nochange]  # we append two sequences into one df
    no_boundaries = pandas.concat(no_boundaries)
    no_boundaries = no_boundaries.sort_values(by='idx', ascending=True)

    sequence = [no_boundaries, yes_boundaries]  # we append two sequences into one df
    sequence = pandas.concat(sequence)
    semi_final = sequence.sort_values(by='idx', ascending=True)
    semi_final = semi_final.reset_index(drop=True)

    # control for consecutive ones in cue
    for i in range(len(semi_final) - 4):
        if (semi_final['cue'][i] == 1) and (semi_final['cue'][i + 1] == 1):
            semi_final['cue'][i + 1] = 0
            semi_final['cue'][i + 4] = 1

    # control for consecutive ones in sequence
    for i in range(len(semi_final) - 4):
        if (semi_final['sequence'][i] == 1) and (semi_final['sequence'][i + 1] == 1):
            semi_final['sequence'][i + 1] = 0
            semi_final['sequence'][i + 4] = 1

    sequence = [semi_final, no_boundaries_unchangable]
    sequence = pandas.concat(sequence)
    final = sequence.sort_values(by='idx', ascending=True)
    final = final.reset_index(drop=True)

    final.to_csv(
        path + f"/experiment/Results/{subject}/{subject}_seq_{melody_file[:-4]}_{condition}.csv",
    )
    # print(final.to_string())
    print("Total visual cues:")
    print(sum(final["cue"]))
    print("Total location changes:")
    print(sum(final["sequence"]))
    print("Boundary with change:")
    print(final[(final['boundary'] == 1) & (final['sequence'] == 1)].shape[0])
    print(final[(final['boundary'] == 1) & (final['sequence'] == 1) & (final['cue'] == 1)].shape[0])
    print("Boundary with no change:")
    print(final[(final['boundary'] == 1) & (final['sequence'] == 0)].shape[0])
    print(final[(final['boundary'] == 1) & (final['sequence'] == 0) & (final['cue'] == 1)].shape[0])
    print("No boundary with change:")
    print(final[(final['boundary'] == 0) & (final['sequence'] == 1)].shape[0])
    print(final[(final['boundary'] == 0) & (final['sequence'] == 1) & (final['cue'] == 1)].shape[0])
    print("No boundary with no change:")
    print(final[(final['boundary'] == 0) & (final['changable_notes'] == 1) & (final['sequence'] == 0)].shape[0])
    print(final[(final['boundary'] == 0) & (final['changable_notes'] == 1) & (final['sequence'] == 0) & (
            final['cue'] == 1)].shape[0])
    return final

# balanced_sequence(boundaries, changable_notes, "test", file, cond)
