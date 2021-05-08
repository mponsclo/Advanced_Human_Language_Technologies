# Import 
import argparse


from utils import read_feature_file, read_test_feature_file
import nltk.classify

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_file_path", type=str, help = "Path to data")
    parser.add_argument("test_file_path", type=str, help = "Path to test data")
    parser.add_argument("output_file_path", type=str, help = "Path to save classified test data")
    args = parser.parse_args()



    # read features from file
    train_data = read_feature_file(args.train_file_path)
    test_data = read_test_feature_file(args.test_file_path)
    nltk.classify.megam.config_megam('/home/zosia/uni-dev/ahlt/2. DDI/megam_i686.opt')
    mymodel = nltk.classify.MaxentClassifier.train(train_data, 'megam')

    output_f = open(args.output_file_path, 'w')

    for sid, e1, e2, feats in test_data:
        prediction = mymodel.classify(feats)
        if prediction != "null":
            print ( sid , e1 , e2 , prediction , sep ="|")
            output_f.write(f'{sid}|{e1}|{e2}|{prediction}\n')
