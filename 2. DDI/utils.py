from itertools import groupby

# helper functions for crf-learner and crf-classifier

def parse_string(line):
    split_data = line[:-1].split('\t')
    sentence_id = split_data[0]
    e1_id = split_data[1]
    e2_id = split_data[2]
    interaction = split_data[3]
    feats = split_data[4:]
    features = dict()
    for f in feats:
        features[f] = True
    
    return sentence_id, e1_id, e2_id, features, interaction

def read_feature_file(filepath):
    '''
    Task:
        Given the path to the file containing tokenized sentences, read it and return the necessary data structures
    Input:
        filepath: Path to the data
    Output:
        tokens_by_sentence: list of tuples: (sentence_id, list_of_tokens). Each tuple represents a sentence,
            where in the list_of_tokens each token is represented by a tuple (word, offset_from, offset_to)
        features: list of lists of features per sentence
        tags: list of lists of B-I-O tags per sentence
    '''
    
    f = open(filepath, 'r')
    lines = f.readlines()

    data = []

    for line in lines:
        if len(line) > 1:
            sentence_id, e1, e2, features, interaction = parse_string(line)
            data.append((features, interaction))
    return data

def read_test_feature_file(filepath):
    
    f = open(filepath, 'r')
    lines = f.readlines()

    data = []

    for line in lines:
        if len(line) > 1:
            sentence_id, e1, e2, features, interaction = parse_string(line)
            data.append((sentence_id, e1, e2, features))
    return data

