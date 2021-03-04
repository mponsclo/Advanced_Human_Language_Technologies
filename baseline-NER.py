'''Program that parses all XML files in the folder given as argument and recognizes and classifies drug names. 
Based on simple heuristic rules'''

import string
from evaluator import *
import argparse
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

SIMPLE_DB_PATH = './resources/HSDB.txt'
DRUG_BANK_PATH = './resources/DrugBank.txt'

SimpleDrugDb = set()
DrugBank = {'drug' : set(), 'brand': set(), 'group': set()}

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

def read_drug_list_files():
    '''
    Read the drug databases.
    '''
    resource_file = open(SIMPLE_DB_PATH, 'r')
    lines = resource_file.readlines()
    global SimpleDrugDb
    SimpleDrugDb = set([d[:-1].lower() for d in lines])

    resource_file = open(DRUG_BANK_PATH, 'r')
    lines = resource_file.readlines()
    global DrugBank
    split_lines = [(line.split('|')[0].lower(), line.split('|')[1][:-1]) for line in lines]

    for name, n_type in split_lines:
        DrugBank[n_type].add(name)



def token_type_classifier(word, should_look_up=False):
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

    drug_n = ["PCP", "18-MC", "methyl", "phenyl", "tokin", "fluo", "ethyl"]

    groups = ["depressants", "steroid", "ceptives", "urates", "amines", "azines", 
              "phenones", "inhib", "coagul", "block", "acids", "agent", "+", "-",
              "NSAID", "TCA", "SSRI", "MAO"]

    if should_look_up:
        if (word.lower() in SimpleDrugDb): # Priority
            return True, "drug"
        if (word.lower() in DrugBank["drug"]):
            return True, "drug"
        if (word.lower() in DrugBank["brand"]):
            return True, "brand"
        if (word.lower() in DrugBank["group"]):
            return True, "group"
    if word.isupper() & (len(word) >= 4):
        return True, "brand"
    elif (word[-3:] in threes) | (word[-4:] in fours) | (word[-5:] in fives):
        return True, "drug"
    elif (True in [t in word for t in groups]) | ((word[-1:] == "s") & len(word) >= 8):
        return True, "group"
    elif (True in [t in word for t in drug_n]) | (word.isupper() & (len(word) < 4 & len(word) >= 2)): 
        return True, "drug_n"
    else:
        return False, ""

def extract_entities(sentence, should_look_up):
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
        (is_brand_drug_group, type_text) = token_type_classifier(token_text, should_look_up)

        if is_brand_drug_group:
            offset_from = t[1]
            offset_to = t[2]
            entity = {"name": token_text,
                      "offset": str(offset_from) + "-" + str(offset_to),
                      "type": type_text}
            output.append(entity)

    return (output)

def main(datadir, outfile, should_look_up=False):
    '''
    Main function.
    Input:
    datadir - directory with XML files
    outfile - name for the output file
    drug_path - path to the drug database file
    should_look_up - bollean indicating if the words should be looked up in a drug database
    '''

    outf = open(outfile, 'w')
    if (should_look_up):
        read_drug_list_files()

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
                # create bigrams from consecutive tokens
                bigrams = filter(lambda t: t[1][1] - t[0][2] == 2, zip(tokens[:-1], tokens[1:]))
                bigrams = [(f'{t0[0]} {t1[0]}', t0[1], t1[2]) for (t0, t1) in bigrams]
                tokens.extend(bigrams)

                # extract entities from tokenized sentence text
                entities = extract_entities(tokens, should_look_up)

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
    parser.add_argument("-l", nargs='?', const=True, default=False, help="Include looking up in the dictionary")
    args = parser.parse_args()
    main(args.train_data_path, args.output_file_name, args.l)