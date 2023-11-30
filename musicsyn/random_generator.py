import csv
import random


def generate_data():
    duration = 0.9
    freq_range = [440 * 2 ** (n / 12) for n in range(-12, 13)]  # Equal temperament frequencies
    data = []

    # Generate random sequence of 0 and 1 (20% 1s, excluding first and last two rows)

    loc_change_sequence = random.sample([1] * 8 + [0] * 52, 60)


    # Ensure no two consecutive 1s
    for i in range(2, 58, 2):
        if loc_change_sequence[i] == 1:
            loc_change_sequence[i + 1] = 0

    # Change a 1 to 0 if the next row also contains a 1
    for i in range(58):  # Adjusted range to avoid IndexError
        if loc_change_sequence[i] == 1 and loc_change_sequence[i + 1] == 1:
            loc_change_sequence[i] = 0

    random.shuffle(loc_change_sequence)

    for onset_sec, loc_change in zip(range(60), loc_change_sequence):
        freq = random.choice(freq_range)
        offset = onset_sec + duration
        data.append([onset_sec, duration, freq, offset, loc_change])

    return data


def save_to_csv(data, filename):
    with open(filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['onset_sec', 'duration', 'freq', 'offset', 'loc_change'])
        csv_writer.writerows(data)


# Generate and save 5 sets of random sequences
for i in range(5):
    data = generate_data()
    filename = f'output_{i + 1}.csv'
    save_to_csv(data, filename)
    print(f'Saved {filename}')
