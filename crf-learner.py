# Import 
import argparse
import sklearn
import pycrfsuite
from itertools import groupby
from operator import itemgetter

def sentence2features(sentence):
    classes = []
    features = []
    tokens = []
    for token in sentence:
        split_line = token[:-1].split('\t')
        sentence_id = split_line[0]
        feature_dict = {
            "form": split_line[5],
            "formlower": split_line[6]

        }
        #TODO read all features from text files
        features.append(split_line[5:])
        classes.append(split_line[4])
        tokens.append((split_line[1], split_line[2], split_line[3]))
    return features, classes, tokens

def read_feature_file(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    features = []
    classes = []
    metadata=[] # sentence ids, token form, offsets - for later reconstruction
    sentences = set()
    #print(lines)
    sentences = groupby(lines, lambda l: l.split('\t')[0])
    for sid, sentence in sentences:
        s_features, s_classes, s_tokens = sentence2features(sentence)
        features.append(s_features)
        classes.append(s_classes)
        metadata.append((sid, s_tokens))
           
    return metadata, features, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file_path", type=str, help = "Path to data")
    args = parser.parse_args()
    train_metadata, X_train, y_train = read_feature_file(args.train_file_path)

    train_result_file = 'conll2002-esp.crfsuite'
    trainer = pycrfsuite.Trainer(verbose = False)

    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 1.0,  # coefficient for L1 penalty
        'c2': 1e-3,  # coefficient for L2 penalty
        'max_iterations': 50,   # stop earlier

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train(train_result_file)
