import csv
import json
import numpy as np
import pandas as pd
import gensim.downloader

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

raw_data_fd = open('raw_data.json')
raw_data = json.load(raw_data_fd)

word_to_vec = gensim.downloader.load("glove-wiki-gigaword-100")


# Feature Creation Fn's
def example(row):
    print(row.name)
    return row


def question_matching(row):
    keywords = {
        'q0': set(['load', 'dataset', 'csv', 'file']),
        'q1': set(['shape', 'summary', 'head', 'map', 'missing', 'label']),
        'q2': set(['shuffle', 'seperate', 'split', 'training', '80', '20']),
        'q3': set(['correlation', 'feature', 'selection', 'hypothetical']),
        'q4': set(['hyperparameter', 'tune', 'gridsearchcv']),
        'q5': set(['retrain', 'hyperparameter', 'decision', 'tree', 'plot']),
        'q6': set(['predict', 'classification', 'accuracy', 'confusion', 'matrix']),
        'q7': set(['information', 'gain', 'entropy', 'formula'])
    }
    name = row.name
    prompt_answer_pairs = raw_data.get(name)

    question_dict = {'q0': 0, 'q1': 0, 'q2': 0, 'q3': 0, 'q4': 0, 'q5': 0, 'q6': 0, 'q7': 0}
    for pair in prompt_answer_pairs:
        prompt_set = set(pair[0].split())
        match_counts = {key: 0 for key in keywords}

        for question_key, keywords_set in keywords.items():
            match_counts[question_key] += len(prompt_set.intersection(keywords_set))

        question_label = max(match_counts, key=match_counts.get)
        question_dict[question_label] += 1

    for i in range(0, 8):
        row[f'question_match_{i}'] = question_dict[f'q{i}']

    return row


def length_and_count(row):
    for key, prompt_answer_pairs in raw_data.items():
        prompt_sum_of_words = 0
        answer_sum_of_words = 0
        pair_count = len(prompt_answer_pairs)
        for prompt, answer in prompt_answer_pairs:
            prompt_sum_of_words += len(prompt)
            answer_sum_of_words += len(answer)

    row['pair_count'] = pair_count
    row['avg_prompt_length'] = prompt_sum_of_words / pair_count
    row['avg_answer_length'] = answer_sum_of_words / pair_count

    return row


def vectorized_prompts(row):
    key = row.name
    prompt_answer_pairs = raw_data[key]
    prompt_vector = np.zeros(word_to_vec.vector_size)

    for each_pair in prompt_answer_pairs:
        text = each_pair[0]
        words = text.split()
        word_vectors = []

        for word in words:
            if word in word_to_vec:
                word_vectors.append(word_to_vec[word])

        if word_vectors:  # Calculate the average of word vectors along the columns (axis=0)
            prompt_vector = np.mean(word_vectors, axis=0)

    for i, val in enumerate(prompt_vector):
        row[f"prompt_vector_{i}"] = prompt_vector[i]
    return row


# Row processing
def our_super_great_row_processor(row):
    row = question_matching(row)
    row = length_and_count(row)
    row = vectorized_prompts(row)
    print(row.name, "processed")
    return row


columns = [f"prompt_vector_{i}" for i in range(100)]
columns += [f"question_match_{i}" for i in range(8)]
columns += ["pair_count", "avg_prompt_length", "avg_answer_length", "grade"]

dataframe = pd.DataFrame(index=raw_data.keys(), columns=columns)
dataframe.apply(our_super_great_row_processor, axis=1)

grades_fd = open("./materials/scores.csv")
grades_csv_reader = csv.reader(grades_fd)

for i, row in enumerate(grades_csv_reader):
    if i > 0:
        key = row[1].strip()
        grade = float(row[2].strip())
        dataframe.at[key, 'grade'] = grade

print(dataframe)
