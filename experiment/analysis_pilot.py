import pandas
import numpy
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import  ttest_rel

def create_df():
    path = os.getcwd()
    subjects = [f for f in os.listdir(f"{path}/Results") if f.startswith(("sub", "part"))]
    subjects_eeg = [i for i in subjects if "eeg" in i ]
    subjects = list(set(subjects) - set(subjects_eeg))
    old = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05']
    subjects = list(set(subjects) - set(old))
    df = pandas.DataFrame()

    for subject in subjects:
        folder_path = f"{path}/Results/{subject}"

        file_names = [f for f in os.listdir(folder_path) if "_data_" in f]

        for file in file_names:
            data = pandas.read_csv(f"{folder_path}/{file}")
            for index, row in data.iterrows():
                # Check if Visual_cue is 1
                if row['visual_cue'] == 1:
                    # Check for response in the window of 1.1 seconds
                    window_condition = (data['time'] >= row['time']) & (data['time'] <= row['time'] + 1.1)
                    found_index = data.index[window_condition & (data['answer'] == 1)].tolist()

                    if any((data.loc[window_condition, 'answer'] == 1)):
                        # If there is a response in the window, set the Response value for the current row to 1
                        data.at[index, 'answer'] = 1
                        data.at[index, 'prec_time'] = data['prec_time'][found_index[0]]

            df = pandas.concat([df, data])

    df2 = df.iloc[:, 1:]

    df2["time_difference"] = df2["prec_time"] - df2["time"]
    df2.loc[df2['stimulus'].str.startswith('test'), 'block'] = 'training'
    df2.loc[df2['stimulus'].str.startswith('stim'), 'block'] = 'experiment'

    df2.loc[df2['block'].str.startswith('train'), 'condition'] = 'train'

    df2_sorted = df2.sort_values(by=['subject', 'hour'])

    # Enumerating condition_number and trial_number within each group
    df2_sorted['condition_number'] = df2_sorted.groupby(['subject', 'block', 'condition']).cumcount() + 1
    df2_sorted['trial_number'] = df2_sorted.groupby(['subject', 'block', 'condition', 'hour']).cumcount() + 1
    df2_sorted['condition_n'] = ''
    df2_sorted['trial_n'] = ''
    df2_sorted = df2_sorted.reset_index()
    df2_sorted = df2_sorted.iloc[:, 1:]

    df_raw = pandas.DataFrame()

    for subject in subjects:
        temp_df = df2_sorted[df2_sorted["subject"] == subject]

        i=0
        for index, row in temp_df.iterrows():
            if row['condition_number'] == 1:
                i += 1
                temp_df.loc[index, 'condition_n'] = i

            else:
                temp_df.loc[index, 'condition_n'] = temp_df.loc[index - 1, 'condition_n']
        i = 0
        for index, row in temp_df.iterrows():
            if row['trial_number'] == 1:
                i += 1
                temp_df.loc[index, 'trial_n'] = i

            else:
                temp_df.loc[index, 'trial_n'] = temp_df.loc[index - 1, 'trial_n']

        df_raw = pandas.concat([df_raw, temp_df])
    df_raw = df_raw.reset_index()
    df_raw = df_raw.iloc[:, 1:]
    #worst_melodies = ['stim_min_4', 'stim_min_5', 'stim_min_6', 'stim_maj_4', 'stim_maj_5', 'stim_maj_6']
    #df_raw = df_raw[df_raw["stimulus"] == 'stim_maj_2']
    df_raw.to_csv(f'{path}/Results/results_raw.csv', index=False)

    #### PRINT DISTRIBUTION OF SIGNAL THEORY PER CONDITION

    condition_1 = df_raw[df_raw['boundary'] == 1]
    condition_0 = df_raw[df_raw['boundary'] == 0]

    # Creating contingency table for condition 1
    table_condition_1 = pandas.crosstab(index=condition_1['loc_change'], columns=condition_1['visual_cue'])

    # Creating contingency table for condition 0
    table_condition_0 = pandas.crosstab(index=condition_0['loc_change'], columns=condition_0['visual_cue'])

    print("Condition 'At boundary'")
    print(table_condition_1)

    print("\nCondition 'No boundary':")
    print(table_condition_0)


    #### FILTER FOR CUED INSTANCES

    vc_data = df_raw[df_raw['visual_cue'] == 1].copy()


    vc_data['signal_theory'] = ''
    vc_data['block'] = ''

    vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 1), 'signal_theory'] = 'hit'
    vc_data.loc[(vc_data['loc_change'] == 1) & (vc_data['answer'] == 0), 'signal_theory'] = 'miss'
    vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 0), 'signal_theory'] = 'corr'
    vc_data.loc[(vc_data['loc_change'] == 0) & (vc_data['answer'] == 1), 'signal_theory'] = 'fa'

    vc_data = vc_data[(vc_data['time_difference'] > 0.2) | (vc_data['answer'] != 1)]

    vc_data.loc[vc_data['stimulus'].str.startswith('test'), 'block'] = 'training'
    vc_data.loc[vc_data['stimulus'].str.startswith('stim'), 'block'] = 'experiment'

    #### SAVE RAW RESULTS AS .CSV

    vc_data.to_csv(f'{path}/Results/results.csv')


    #### CALCULATING D-PRIME

    at_boundary = vc_data[vc_data['boundary'] == 1]
    no_boundary = vc_data[vc_data['boundary'] == 0]
    at_boundary_grouped = at_boundary.groupby(['subject', 'block','condition', 'signal_theory']).size().unstack(fill_value=0)

    if 'fa' not in at_boundary_grouped.columns:
        at_boundary_grouped['fa'] = 0

    if 'hit' not in at_boundary_grouped.columns:
        at_boundary_grouped['hit'] = 0
    at_boundary_grouped.reset_index(inplace=True)

    no_boundary_grouped = no_boundary.groupby(['subject', 'block', 'condition', 'signal_theory']).size().unstack(fill_value=0)

    if 'fa' not in no_boundary_grouped.columns:
        no_boundary_grouped['fa'] = 0

    if 'hit' not in no_boundary_grouped.columns:
        no_boundary_grouped['hit'] = 0
    no_boundary_grouped.reset_index(inplace=True)

    at_boundary_grouped["state"] = "boundary"
    no_boundary_grouped["state"] = "no_boundary"

    main = pandas.concat([at_boundary_grouped, no_boundary_grouped])
    main = main.reset_index(drop=True)


    # CALCULATE HIT RATE AND FALSE RATE

    for i in range(len(main)):
        main.at[i, "hit_rate"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "miss"])
        main.at[i, "false_rate"] = main.at[i, "fa"] / (main.at[i, "fa"] + main.at[i, "corr"])

    main['false_rate'] = main['false_rate'].fillna(0)

    main["z_hit_rate"] = ""
    main["z_false_rate"] = ""
    main["d_prime"] = ""
    score_columns = ["hit_rate", "false_rate"]

    for col in score_columns:
        temp = main[col]
        temp_targ = main[f"z_{col}"]
        mean = numpy.mean(temp)
        sd = numpy.std(temp)

        for idx in temp.index:
            temp_targ[idx] = (temp[idx] - mean) / sd

        main[f"z_{col}"] = temp_targ

    for idx in main.index:
        main["d_prime"][idx] = main["z_hit_rate"][idx] - main["z_false_rate"][idx]

    for i in range(len(main)):
        main.at[i, "precision"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "fa"])
        main.at[i, "recall"] = main.at[i, "hit"] / (main.at[i, "hit"] + main.at[i, "miss"])
        main.at[i, "ff1"] = 2 / ((1 / main.at[i, "recall"]) + (1 / main.at[i, "precision"]))

    #### SAVE D-PRIME and F-SCORE INTO CSV
    main.to_csv(f'{path}/Results/results_postprocessed.csv', index=False)


def plot_group(condition, block):

    path = os.getcwd()
    df = pandas.read_csv(path + f"/Results/results_postprocessed.csv")
    df = df[df['block'].str.match(f'{block}')]
    main = df[df['condition'].str.match(f'{condition}')]
    vc_data = pandas.read_csv(path + f"/Results/results.csv")
    vc_data = vc_data[vc_data['condition'].str.match(f'{condition}')]
    vc_data = vc_data[vc_data['block'].str.match(f'{block}')]


    #### RESHAPE THE DF FOR THE PAIR-WISE COMPARISON
    pairwise_d_prime = main.pivot(index='subject', columns='state', values='d_prime')
    pairwise_ff1 = main.pivot(index='subject', columns='state', values='ff1')
    pairwise_hit_rate = main.pivot(index='subject', columns='state', values='hit_rate')
    pairwise_false_rate = main.pivot(index='subject', columns='state', values='false_rate')


    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#

    palette = sns.color_palette(['#a4e0f5'], len(main['subject'].unique()))

    def plot_and_test(ax, data, data_plot, x, y, title, a, b, c, d, e):
        sns.lineplot(ax=ax, x=x, y=y, data=data_plot, hue='subject', palette=palette, marker='o', legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['boundary', 'no_boundary'])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(title)
        ax.set_yticks(numpy.arange(a, b, c))
        ax.set_ylim(d, e)

        t_stat, p_value = ttest_rel(data.iloc[:, 0], data.iloc[:, 1], nan_policy='omit')
        ax.text(0.5, 0.9, f't(7)= {t_stat:.4f} \np = {p_value:.4f}', transform=ax.transAxes, ha='center')

    def plot_avg(ax, data_plot, x, y, title, a, b, c, d, e):
        sns.lineplot(ax=ax, x=x, y=y, data=data_plot, color='#008EBF', marker='o', legend=False)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['at boundary', 'within unit'])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(title)
        ax.set_yticks(numpy.arange(a, b, c))
        ax.set_ylim(d, e)

    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Comparison between two conditions - {condition}')
    main['ff1'] = pandas.to_numeric(main['ff1'], errors='coerce')
    main['d_prime'] = pandas.to_numeric(main['d_prime'], errors='coerce')

    # Hit rate plot
    plot_and_test(axes[0, 0], pairwise_hit_rate, main, 'state', 'hit_rate', 'Hit Rate', 0.5, 1.01, 0.1, 0.45, 1.1)
    plot_avg(axes[0, 0], main, 'state', 'hit_rate', 'Hit Rate', 0.5, 1.01, 0.1, 0.45, 1.1)

    # False alarm rate plot
    plot_and_test(axes[0, 1], pairwise_false_rate, main, 'state', 'false_rate', 'False Alarm Rate', 0.1, 1.01, 0.1, 0.01, 1.1)
    plot_avg(axes[0, 1], main, 'state', 'false_rate', 'False Alarm Rate', 0.1, 1.01, 0.1, 0.01, 1.1)

    # D-prime plot
    plot_and_test(axes[1, 1], pairwise_d_prime, main, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)
    plot_avg(axes[1, 1], main, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)

    # F-score plot
    plot_and_test(axes[1, 0], pairwise_ff1, main, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)
    plot_avg(axes[1, 0], main, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{path}/plots/{condition}_{block}_results.png', dpi=300)
    plt.show()

    def plot_conditions(ax, data_plot, x, y, title, a, b, c, d, e):
        sns.lineplot(ax=ax, x=x, y=y, data=data_plot, hue='condition', palette='tab10', marker='o')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['boundary', 'no_boundary'])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(title)
        ax.set_yticks(numpy.arange(a, b, c))
        ax.set_ylim(d, e)

    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'preliminary results')
    main['ff1'] = pandas.to_numeric(main['ff1'], errors='coerce')
    main['d_prime'] = pandas.to_numeric(main['d_prime'], errors='coerce')

    # Hit rate plot
    plot_conditions(axes[0, 0], df, 'state', 'hit_rate', 'Hit Rate', 0.5, 1.01, 0.1, 0.45, 1.1)

    # False alarm rate plot
    plot_conditions(axes[0, 1], df, 'state', 'false_rate', 'False Alarm Rate', 0.2, 1.01, 0.1, 0.15, 1.1)

    # D-prime plot
    plot_conditions(axes[1, 1], df, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)

    # F-score plot
    plot_conditions(axes[1, 0], df, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{path}/plots/condition_comparison.png', dpi=300)
    plt.show()

######## SINGLE PARTICIPANT
def plot_single(subject, condition, block):
    path = os.getcwd()
    df = pandas.read_csv(path + f"/Results/results_postprocessed.csv")
    main = df[df['condition'].str.match(f'{condition}')]
    main = main[main['block'].str.match(f'{block}')]
    vc_df = pandas.read_csv(path + f"/Results/results.csv")
    vc_data = vc_df[vc_df['condition'].str.match(f'{condition}')]
    vc_data = vc_data[vc_data['block'].str.match(f'{block}')]

    #### PARAMETERS
    sub_main = df[(df['subject'] == subject)]
    sub_main = sub_main[(sub_main['block'] == block)]


    #-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    palette = sns.color_palette(['#a4e0f5'], len(main['subject'].unique()))

    def plot_sub(ax, data_plot, x, y, title, a, b, c, d, e):
        sns.lineplot(ax=ax, x=x, y=y, data=data_plot, hue='condition', palette='tab10', marker='o')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['boundary', 'no_boundary'])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(title)
        ax.set_yticks(numpy.arange(a, b, c))
        ax.set_ylim(d, e)

    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{subject} - preliminary results')
    main['ff1'] = pandas.to_numeric(main['ff1'], errors='coerce')
    main['d_prime'] = pandas.to_numeric(main['d_prime'], errors='coerce')

    # Hit rate plot
    plot_sub(axes[0, 0], sub_main, 'state', 'hit_rate', 'Hit Rate', 0.5, 1.01, 0.1, 0.45, 1.1)

    # False alarm rate plot
    plot_sub(axes[0, 1], sub_main, 'state', 'false_rate', 'False Alarm Rate', 0.2, 1.01, 0.1, 0.15, 1.1)

    # D-prime plot
    plot_sub(axes[1, 1], sub_main, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)

    # F-score plot
    plot_sub(axes[1, 0], sub_main, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{path}/plots/{subject}_condition_results.png', dpi=300)
    plt.show()

    sub_main = main[(main['subject'] == subject)]


    # -#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-#-

    palette = sns.color_palette(['#a4e0f5'], len(main['subject'].unique()))

    def plot_sub(ax, data_plot, x, y, title, a, b, c, d, e):
        sns.lineplot(ax=ax, x=x, y=y, data=data_plot, marker='o')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['boundary', 'no_boundary'])
        ax.set_xlim(-0.2, 1.2)
        ax.set_title(title)
        ax.set_yticks(numpy.arange(a, b, c))
        ax.set_ylim(d, e)

    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{subject} - preliminary results')
    main['ff1'] = pandas.to_numeric(main['ff1'], errors='coerce')
    main['d_prime'] = pandas.to_numeric(main['d_prime'], errors='coerce')

    # Hit rate plot
    plot_sub(axes[0, 0], sub_main, 'state', 'hit_rate', 'Hit Rate', 0.5, 1.01, 0.1, 0.45, 1.1)

    # False alarm rate plot
    plot_sub(axes[0, 1], sub_main, 'state', 'false_rate', 'False Alarm Rate', 0.2, 1.01, 0.1, 0.15, 1.1)

    # D-prime plot
    plot_sub(axes[1, 1], sub_main, 'state', 'd_prime', 'D-prime', -4, 3, 1, -4.1, 2.5)

    # F-score plot
    plot_sub(axes[1, 0], sub_main, 'state', 'ff1', 'F-score', 0.5, 1.01, 0.1, 0.45, 1.1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'{path}/plots/{subject}_main_results.png', dpi=300)
    plt.show()