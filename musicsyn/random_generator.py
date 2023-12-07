import csv
import random


def generate_data():
    duration = 0.9
    freq_range = [440 * 2 ** (n / 12) for n in range(-12, 13)]
    data = []

    loc_change_sequence = random.sample([1] * 16 + [0] * 44, 60)

    if loc_change_sequence[0] == 1:
        loc_change_sequence[0] = 0
        loc_change_sequence[1] = 1
    if loc_change_sequence[-1] == 1:
        loc_change_sequence[-1] = 0
        loc_change_sequence[-2] = 1

    # Ensure no two consecutive 1s
    for i in range(2, 58, 2):
        if (loc_change_sequence[i] == 1) and (loc_change_sequence[i + 1] == 1):
            loc_change_sequence[i + 1] = 0
            loc_change_sequence[i + 2] = 1

    boundaries = [0, 0, 0, 1] * int(60/4)
    changable_notes = [0,1,0,1,1,0,1,1] * int(60/8)


    for onset_sec, loc_change, boundary, changable in zip(range(60), loc_change_sequence, boundaries, changable_notes):
        freq = random.choice(freq_range)
        offset = onset_sec + duration
        data.append([onset_sec, duration, freq, offset, loc_change, boundary, changable])

    return data

def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['onset_sec', 'duration', 'freq', 'offset', 'loc_change', 'boundary', 'changable_note'])
        csv_writer.writerows(data)


# Generate and save 5 sets of random sequences
for i in range(5):
    data = generate_data()
    filename = f'test{i + 1}.csv'
    save_to_csv(data, filename)
    print(f'Saved {filename}')
