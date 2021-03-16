# Import 
import argparse
import sklearn
import pycrfsuite

from operator import itemgetter

from utils import read_feature_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model file")
    parser.add_argument("train_file_path", type=str, help = "Path to data")
    args = parser.parse_args()

    # read features from file
    _, X_train, y_train = read_feature_file(args.train_file_path)

    # create trainer and configure it
    trainer = pycrfsuite.Trainer(verbose=False)
    for xseq, yseq in zip(X_train, y_train):
        trainer.append(xseq, yseq)

    trainer.set_params({
        'c1': 0.2,  # coefficient for L1 penalty
        'c2': 0.001,  # coefficient for L2 penalty
        'max_iterations': 1000,
        'feature.possible_states': True,

        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    #train the model and save it
    trainer.train(args.model_name)
