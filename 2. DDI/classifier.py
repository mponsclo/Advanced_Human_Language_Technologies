import argparse

def main(model_file_path, feature_path):
    model_file = open(model_file_path)
    feature_file = open(feayure_file_path)
    # read each vector in input file
    for line in model_file:
        # split line into elements
        fields = line.strip ('\n').split("\t")
        # first 4 elements are sid ,e1 ,e2 , and ground truth ( ignored since we are classifying )
        ( sid , e1 , e2 , gt ) = fields [0:4]
        # Rest of elements are features , passed to the classifier of choice to get a prediction
        prediction = mymodel.classify ( fields [4:])
        # if the classifier predicted a DDI , output it   in the right format
        if prediction != "null":
            print ( sid , e1 , e2 , prediction , sep ="|")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help = "Path to model")
    parser.add_argument("feature_path", type=str, help = "Path to features")
    args = parser.parse_args()
    main(args.model_path, args.feature_path)