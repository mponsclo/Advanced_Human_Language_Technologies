from itertools import groupby

# helper functions for crf-learner and crf-classifier

def parse_sentence_strings(sentence):
    '''
    Task:
        Given a stringified sentence, parse it and return the data in relevant data structures.
    Input:
        sentence: list of strings representing the given sentence, where each piece of information is separated by a tab, e.g.
            "DDI-DrugBank.d695.s0	While	0	4	O	form=While	formlower=while	suf3=ile	suf4=hile"
    Output:
        tokens: list of tuples representing tokens: (word, offset_from, offset_to)
        features: list of features for every token
        tags: list of tags (e.g. B-drug, O, I-brand etc.) for every token
    '''
    tags = []
    features = []
    tokens = []
    for token in sentence:
        split_data = token[:-1].split('\t')
        sentence_id = split_data[0]
        features.append(split_data[5:])
        tags.append(split_data[4])
        tokens.append((split_data[1], split_data[2], split_data[3]))
    return tokens, features, tags

def read_feature_file(filepath):
    '''
    Task:
        Given the path to the file containing tokenized sentences, read it and return the necessary data structures
    Input:
        filepath: Path to the data
    Output:
        metadata: list of tuples: (sentence_id, list_of_tokens). Each token represents a sentence,
            where in the list_of_tokens each token is represented by a tuple (word, offset_from, offset_to)
        features: list of lists of features per sentence
        tags: list of lists of tags (e.g. B-drug, O, I-brand etc.) per sentence
    '''
    features = []
    tags = []
    metadata=[]
    
    f = open(filepath, 'r')
    lines = f.readlines()

    # group the tokens by sentence
    sentences = groupby(lines, lambda l: l.split('\t')[0])

    # process each sentence
    for sid, sentence in sentences:
        s_tokens, s_features, s_tags = parse_sentence_strings(sentence)
        features.append(s_features)
        tags.append(s_tags)
        metadata.append((sid, s_tokens))
           
    return metadata, features, tags