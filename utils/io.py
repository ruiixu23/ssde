import os
import pickle


def load_data(directory, filename):
    with open(os.path.join(directory, filename), 'rb') as infile:
        return pickle.load(infile)


def save_data(directory, filename, data):
    if os.path.exists(directory) is False:
        os.makedirs(directory)

    with open(os.path.join(directory, filename), 'wb') as outfile:
        pickle.dump(data, outfile)


def load_all_data(directory, filename, num_repetitions):
    return [
        load_data(directory, filename.format(i + 1))
        for i in range(num_repetitions)
    ]
