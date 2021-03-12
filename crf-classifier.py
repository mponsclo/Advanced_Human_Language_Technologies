import argparse
import pycrfsuite

from utils import read_feature_file

def output_entities(sid, tokens, tags):
    '''
    Task:
        Given a list of tokens and the B-I-O tag for each token, produce a list
            of drugs in the format expected by the evaluator.
    Input:
        sid: sentence identifier (required by the evaluator output format)
        tokens: List of tokens in the sentence , i.e. list of tuples (word,
            offsetFrom, offsetTo)
        tags: List of B-I-O tags for each token
    Output:
        Prints to stdout the entities in the right format: one line per entity,
        fields separated by ’|’, field order : id, offset, name, type.
    '''
    for token, tag in zip(tokens, tags):
        word, offset_from, offset_to = token
        entity = sid + "|" + offset_from + '-' + offset_to + "|" + word + "|" + tag
        print(entity)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str, help="Name of the model file")
    parser.add_argument("dataset_path", type=str, help="Path to file containing the dataset")
    args = parser.parse_args()

    # read features from file
    sentence_ids_and_tokens, X_devel, y_devel = read_feature_file(args.dataset_path)

    # load trained model to the tagger
    tagger = pycrfsuite.Tagger()
    tagger.open(args.model_name)
    # classify the dataset
    tags = [tagger.tag(sentence) for sentence in X_devel]

    # print the classified tokens
    for sentence_tags, sentence_data in zip(tags, sentence_ids_and_tokens):
        sid, tokens = sentence_data
        output_entities(sid, tokens, sentence_tags)

            

