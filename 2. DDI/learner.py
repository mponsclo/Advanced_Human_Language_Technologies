# Import 
import argparse

from nltk.classify import megam

from utils import read_feature_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file_path", type=str, help = "Path to data")
    args = parser.parse_args()
    

    

    # read features from file
    train_data = read_feature_file(args.train_file_path)

    #config learner
    megam.config_megam('../megam_i686.opt',encoding=)
    megam.write_megam_file(train_data)

