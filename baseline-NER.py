'''Program that parses all XML files in the folder given as argument and recognizes and classifies drug names. 
Based on simple heuristic rules'''

import sys
import string
from os import listdir
from xml.dom.minidom import parse
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('punkt')
nltk.download('stopwords')

#from evaluator import evaluate

## ------------- Tokenize Sentence -------------
def tokenize(s):
    '''
    Given a sentence , calls nltk.tokenize to split it in tokens, and adds to each token its start / end offset 
    in the original sentence .
    Input - s: string containing the text for one sentence
    Output - Returns a list of tuples (word , offsetFrom , offsetTo )'''

    token_list = []
    tokens = word_tokenize(s)
    stop_words = set(stopwords.words("english"))
    
    for t in tokens:
        if (t in stop_words) | (not t.isalpha()): # reduce workload
            continue
        else:
            offsetFrom = s.find(t)
            offsetTo = offsetFrom + len(t) - 1
            token_list.append((t, offsetFrom, offsetTo))
            
    return token_list

#sent = tokenize("Phenothiazines and butyrophenones may reduce or reverse the depressor effect of epinephrine.")
#print(sent) 

## ------------- Classify Token -------------
def token_type_classifier(word):
        
    threes = ["nol", "lol", "hol", "lam", "pam"]
    fours = ["arin", "oxin", "toin","pine", "tine", "bital", "inol", "pram"]
    fives = ["azole", "idine", "orine", "mycin", "hrine", "exate", "amine", "emide"]

    groups = ["depressants", "steroid", "ceptives", "urates", "amines", "azines", "phenones", 
              "inhib", "coagul", "block", "acids", "agent"]
    
    if word.isupper() & (len(word) >= 4): 
        return True, "brand"  # add word[0].isupper ?
    elif (word[-3:] in threes) | (word[-4:] in fours) | (word[-5:] in fives):
        return True, "drug"
    elif (True in [t in word for t in groups]) | ((word[-1:] == "s") & (word[-2].isupper())) | (word.isupper() & (len(word) < 4)): 
        return True, "group"
    #elif word in drug_set:        # Drug Database Checking
    #    return True, "drug"
    else: 
        return False, ""

# print(token_type_classifier("NSAIDs"))
# print(sent[0])
#for t in sent:
#    tokenText = t[0]
#    print(token_type_classifier(tokenText))

## ------------- Entity Extractor -------------
def extract_entities(s):
    ''' Given a tokenized sentence , identify which tokens (or groups of consecutive tokens ) are drugs
    Input - s: A tokenized sentence ( list of triples (word , offsetFrom , offsetTo ) )
    Output - A list of entities . Each entity is a dictionary with the keys 'name ', ' offset ', and 'type '''

    output = []
    for t in s:
        tokenText = t[0] # get the only the text from (text, offsetFrom, offsetTo)
        (is_brand_drug_group, type_text) = token_type_classifier(tokenText)
        
        if is_brand_drug_group:
            offsetFrom = t[1]
            offsetTo = t[2]
            entity = {"name" : tokenText,
                     "offset" : str(offsetFrom) + "-" + str(offsetTo), 
                     "type" : type_text}
            output.append(entity)
    
    return(output)

#entity = extract_entities(sent)
#print(entity)

#for f in listdir(datadir):
#    print(f)

def main(datadir): #, outfile):
    '''datadir - directory with XML files
       outfile - name for the outputfile'''

    # process each file in directory
    for f in listdir(datadir):
        try:
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
                        print(sid + "|" + e["offset"] + "|" + e["name"] + "|" + e["type"]) #, file = outf)
                        
        except:
            pass
        # print performance score
        #evaluator.evaluate("NER", datadir, outfile)

path = "/Users/mponsclo/Documents/DataScience/ALHT_Project/train"
result = main(path)
print(result)
