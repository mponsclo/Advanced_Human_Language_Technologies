'''Program that parses all XML files in the folder given as argument and recognizes and classifies drug names. 
Based on simple heuristic rules'''

import string
from evaluator import *
import argparse
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


def tokenize(s):
    '''
    Given a sentence , calls nltk.tokenize to split it in tokens, and adds to each token its start / end offset 
    in the original sentence .
    Input - s: string containing the text for one sentence
    Output - Returns a list of tuples (word , offsetFrom , offsetTo )
    '''

    token_list = []
    tokens = word_tokenize(s)
    stop_words = set(stopwords.words("english"))

    for t in tokens:
        if (t in stop_words) | (not t.isalpha()):  # reduce workload
            continue
        else:
            offsetFrom = s.find(t)
            offsetTo = offsetFrom + len(t) - 1
            token_list.append((t, offsetFrom, offsetTo))

    return token_list

def read_drug_list_file(resource_path):
    '''
    Read the HSDB drug database.
    Input:
    resource_path - path to HSDB.txt file
    Output:
    drugs - set of drugs
    '''
    resource_file = open(resource_path, 'r')
    lines = resource_file.readlines()
    drugs = set([d[:-1].lower() for d in lines])
    return drugs

def token_type_classifier(word, drug_set, should_look_up=False):
    '''
    Classify the tokens.
    Input:
    word - string: token to classify
    drug_set - list of drugs to look up the drugs
    should_look_up - True when words should be looked up in the drug database
    Output:
    tuple (b, s) - b is a boolean value, True when the word is classiied to one of the groups. s is a string - name of the group
    '''
    threes = ["nol", "lol", "hol", "lam", "pam"]
    fours = ["arin", "oxin", "toin", "pine", "tine", "bital", "inol", "pram"]
    fives = ["azole", "idine", "orine", "mycin", "hrine", "exate", "amine", "emide"]

    groups = ["depressants", "steroid", "ceptives", "urates", "amines", "azines", "phenones",
              "inhib", "coagul", "block", "acids", "agent"]

    if word.isupper() & (len(word) >= 4):
        return True, "brand"
    elif should_look_up & (word in drug_set):   
        return True, "drug"
    elif (word[-3:] in threes) | (word[-4:] in fours) | (word[-5:] in fives):
        return True, "drug"
    elif (True in [t in word for t in groups]) | ((word[-1:] == "s") & (word[-2].isupper())) | (
            word.isupper() & (len(word) < 4)):
        return True, "group"

    else:
        return False, ""


def extract_entities(sentence, drug_set, should_look_up):
    '''
    Given a tokenized sentence , identify which tokens (or groups of consecutive tokens ) are drugs
    Input:
    sentence - A tokenized sentence (list of triples (word, offsetFrom, offsetTo))
    drug_set - list of drugs to look up the drugs
    should_look_up - True when words should be looked up in the drug database
    Output:
    output - A list of entities. Each entity is a dictionary with the keys 'name', 'offset', and 'type'
    '''

    output = []
    for t in sentence:
        token_text = t[0]  # get the only the text from (text, offset_from, offset_to)
        (is_brand_drug_group, type_text) = token_type_classifier(token_text, drug_set, should_look_up)

        if is_brand_drug_group:
            offset_from = t[1]
            offset_to = t[2]
            entity = {"name": token_text,
                      "offset": str(offset_from) + "-" + str(offset_to),
                      "type": type_text}
            output.append(entity)

    return (output)

def main(datadir, outfile, drug_path, should_look_up=False):
    '''
    Main function.
    Input:
    datadir - directory with XML files
    outfile - name for the output file
    drug_path - path to the drug database file
    should_look_up - bollean indicating if the words should be looked up in a drug database
    '''

    outf = open(outfile, 'w')
    drugs = read_drug_list_file(drug_path)
    # process each file in directory
    for f in listdir(datadir):
        try:
            # parse XML file, obtaining a DOM tree
            tree = parse(datadir + "/" + f)
            # process each senetence in the file
            sentences = tree.getElementsByTagName("sentence")
            for s in sentences:
                sid = s.attributes["id"].value  # get sentence id
                stext = s.attributes["text"].value  # get sentence text
                # tokenize text
                tokens = tokenize(stext)
                # extract entities from tokenized sentence text
                entities = extract_entities(tokens, drugs, should_look_up)

                # print sentence entities in format requested for evaluation
                for e in entities:
                    entity = sid + "|" + e["offset"] + "|" + e["name"] + "|" + e["type"]
                    outf.write(entity + '\n')

        except:
            pass
    evaluate("NER", datadir, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("train_data_path", type=str, help="Path to train data")
    parser.add_argument("output_file_name", type=str, help="Output file name")
    parser.add_argument("drug_list_path", type=str, help="Path to list of drugs")
    parser.add_argument("-l", nargs='?', const=True, default=False, help="Include looking up in the dictionary")
    args = parser.parse_args()
    main(args.train_data_path, args.output_file_name, args.drug_list_path, args.l)
