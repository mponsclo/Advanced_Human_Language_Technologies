'''Program that parses all XML files in the folder given as argument and recognizes and classifies drug names. 
Based on simple heuristic rules'''

import sys
import string
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from evaluator import evaluate

## ------------- Tokenize Sentence -------------
def tokenize(s):
    '''
    Given a sentence , calls nltk.tokenize to split it in tokens, and adds to each token its start / end offset 
    in the original sentence .
    Input - s: string containing the text for one sentence
    Output - Returns a list of tuples (word , offsetFrom , offsetTo )'''

    offset = 0
    


## ------------- Classify Token -------------



## ------------- Entity Extractor -------------
def extract_entities(s):
    ''' Given a tokenized sentence , identify which tokens (or groups of consecutive tokens ) are drugs
    Input - s: A tokenized sentence ( list of triples (word , offsetFrom , offsetTo ) )
    Output - A list of entities . Each entity is a dictionary with the keys 'name ', ' offset ', and 'type '''



def main(datadir, outfile):
    '''datadir - directory with XML files
       outfile - name for the outputfile'''

    # process each file in directory
    for f in listdir(datadir):
        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)
        # process each senetence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
                sid = s.attributes["id"].value        # get sentence id
                stext = s.attributes["text"].value    # get sentence text
                # tokenize text
                tokens = tokenize(stext)
                # extract entities from tokenized sentence text
                entities = extract_entities(tokens)

                # print sentence entities in format requested for evaluation
                for e in entities:
                    print(sid + "|" + e["offset"] + "|" + e["text"] + "|" e["type"], file = outf)
        # print performance score
        evaluator.evaluate("NER", datadir, outfile)