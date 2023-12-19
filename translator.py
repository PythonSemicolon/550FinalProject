import os
import json
import numpy as np
import copy
import random
from params import win_length, num_samples, LANGUAGE
from stylometric_analyzer import StylometricAnalyzer

def load_author_data_from_dir(source_dir, author):

    # For every author, retrieve the positive samples
    positive_samples = os.listdir(source_dir + author + "/positive/")
    if "README.txt" in positive_samples:
        positive_samples.remove("README.txt")
    training_samples_p = positive_samples[:3]
    dev_samples_p = positive_samples[3:4]
    test_samples_p = positive_samples[4:5]

    # For every author, retrieve the negative samples
    negative_samples = os.listdir(source_dir + author + "/negative/")
    if "README.txt" in negative_samples:
        negative_samples.remove("README.txt")
    training_samples_n = negative_samples[:3]
    dev_samples_n = negative_samples[3:4]
    test_samples_n = negative_samples[4:5]

    # Store the samples with their respective labels
    return {
        'positive': 
            [
                training_samples_p,
                dev_samples_p,
                test_samples_p
            ],
        'negative': 
            [
                training_samples_n,
                dev_samples_n,
                test_samples_n
            ],
    }


def get_fragments(path_to_work, win_length):
    # Obtain num_samples fragments
    fragments = []
    for _ in range(num_samples):
        with open(path_to_work, 'rb') as file:

            # Retrieve win_length size string from random point
            byte_position = random.randint(0, os.path.getsize(path_to_work) - 1 - win_length)
            file.seek(byte_position) 
            fragment_bytes = file.read(win_length) 

            # Decode bytes to string
            fragment = fragment_bytes.decode('utf-8', errors='ignore')
            fragments.append(fragment)

    return fragments


def vectorize_by_section(source_dir, data, author, set, split_name, is_positive):

    subdir = '/positive/' if is_positive else '/negative/'
    label = 1 if is_positive else 0

    for doc in set:
        X = []
        path_to_doc = source_dir + author + subdir + doc
        # Get fragments
        fragments = get_fragments(path_to_doc, win_length) 
        # For every fragment
        for fragment in fragments:
            fragment.replace('\0', '').replace('\r', '')
            # Stylometric Analysis
            analyzer = StylometricAnalyzer(fragment, LANGUAGE[author])
            # Get feature vector
            feature_vector = analyzer.get_feature_vector()
            X.append(feature_vector)

            total_syllable_fail_count[0] += analyzer.count_syllable_fail

        # Normalize X_train
        X = np.array(X)
        X_norm = X / np.linalg.norm(X, axis=0)
        X_norm = X_norm.tolist()

        if f'x_{split_name}' in data[author]:
            data[author][f'x_{split_name}'].extend(X_norm)
            data[author][f'y_{split_name}'].extend([label] * len(X_norm))
        else:
            data[author][f'x_{split_name}'] = copy.deepcopy(X_norm)
            data[author][f'y_{split_name}'] = [label] * len(X_norm)


def vectorize(source_dir, data, author, train, dev, test, is_positive):
    
    vectorize_by_section(source_dir, data, author, train, "train", is_positive)
    vectorize_by_section(source_dir, data, author, dev, "dev", is_positive)
    vectorize_by_section(source_dir, data, author, test, "test", is_positive)


def preprocess_data(source_dir):
    """
    Loads the data from the directories, labels it accordingly
    and carries out stylometric analysis.

    Creates dictionary with Xy_train, Xy_dev and Xy_test for each translator/author.
    {
    'borges':
        {
            'x_train': [[...], ..., [...]]
            'y_train': [...]
            'x_dev': [[...], ..., [...]]
            'y_dev': [...]
            'x_test': [[...], ..., [...]]
            'y_test': [...]
        }
    'rajt-kovaleva':
        {
            'x_train': [[...], ..., [...]]
            'y_train': [...]
            'x_dev': [[...], ..., [...]]
            'y_dev': [...]
            'x_test': [[...], ..., [...]]
            'y_test': [...]
        }
    'park-jung-so'
        {
            'x_train': [[...], ..., [...]]
            'y_train': [...]
            'x_dev': [[...], ..., [...]]
            'y_dev': [...]
            'x_test': [[...], ..., [...]]
            'y_test': [...]
        }
    }
    Returns the dictionary.
    """
    # Get the list of directories in the directory
    authors = os.listdir(source_dir)

    works = dict.fromkeys(authors)

    for author in authors:
        # Retrieve samples for an author
        works[author] = load_author_data_from_dir(source_dir, author) # works[author] := {'positive': [...], 'negative': [...]}
    
    # Initialize data structures
    struct = {'x_train': [], 'y_train': [], 'x_dev': [], 'y_dev': [], 'x_test': [], 'y_test': []}
    data = dict.fromkeys(authors, None)
    for author in authors:
        data[author] = copy.deepcopy(struct)

    # For every author
    for author in authors:
        # For the positive works
        (train, dev, test) = works[author]['positive']
        vectorize(source_dir, data, author, train, dev, test, is_positive=True) # Vectorize samples from all texts

        # For the negative works
        (train, dev, test) = works[author]['negative']
        vectorize(source_dir, data, author, train, dev, test, is_positive=False) # Vectorize samples from all texts
    
    return data


def main():

    global total_syllable_fail_count
    total_syllable_fail_count = [0]

    # Load data from files
    translators_data = preprocess_data('data/translators/')

    print("Total syllable fail count: ", total_syllable_fail_count) # hoping that it doesn't fail more than 100-200 times.

    # Save data to files
    with open('data/translators.json', 'w') as file:
        json.dump(translators_data, file, indent=2)

     

if __name__ == '__main__':
    main()