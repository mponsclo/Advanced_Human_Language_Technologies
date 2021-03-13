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
    i = 0
    while i < len(tokens):
        word, offset_from, offset_to = tokens[i]
        tag = tags[i]
        if tag =='O':
            i += 1
            continue
        
        if tag[0] == 'B':
            tag_name = tag[2:]
            # in case the entity is longer than one word, concatenate it
            j = i + 1
            while j < len(tokens):
                word_next, offset_from_next, offset_to_next = tokens[j]
                tag_next = tags[j]
                if int(offset_from_next) - int(offset_to) != 2 or tag_next[0] != 'I':
                    break
                if tag_next[2:] == tag_name:
                    word = word + ' ' + word_next
                    offset_to = offset_to_next
                j = j+1
            # print the whole entity
            entity = sid + "|" + offset_from + '-' + offset_to + "|" + word + "|" + tag_name
            print(entity)
        i += 1



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
