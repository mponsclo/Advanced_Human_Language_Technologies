# Import libraries
import argparse
from os import listdir
from xml.dom.minidom import parse
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag

wordnet_lemmatizer = WordNetLemmatizer()

stopwords = set(stopwords.words("english"))

SIMPLE_DB_PATH = './resources/HSDB.txt'
DRUG_BANK_PATH = './resources/DrugBank.txt'

SimpleDrugDb = set()
DrugBank = {'drug' : set(), 'brand': set(), 'group': set()}

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

def tokenize(s):
    token_list = []
    tokens = word_tokenize(s)
    
    for t in tokens:
            offsetFrom = s.find(t)
            offsetTo = offsetFrom + len(t) - 1
            token_list.append((t, offsetFrom, offsetTo))
            
    return token_list

def get_tag(token, gold):
    '''
    Input:
        token: A token, i.e. one triple (word, offsetFrom, offsetTo)
        gold: A list of ground truth entities, i.e. a list of triples (offsetFrom, offsetTo, type)
        
    Output:
        The B-I-O ground truth tag for the given token ("B-drug", "I-drug", "B-group", "I-group", "O", ...)
    '''
    (form, start, end) = token
    for (offsetFrom, offsetTo, Type) in gold:
        if start == offsetFrom and end<=offsetTo:
            return "B-" + Type # First letter of token equals 0 -> Beginning
        elif start > offsetFrom and end <=offsetTo:
            return "I-" + Type # Word not in the beginning
    return "O"

def has_numbers(word):
    return any(l.isdigit() for l in word)

def num_digits(word):
    return sum(l.isdigit() for l in word)

def token_type_classifier(word, should_look_up=False):
    #fives = ["azole", "idine", "orine", "mycin", "hrine", "exate", "amine", "emide"]

    #drug_n = ["PCP", "18-MC", "methyl", "phenyl", "tokin", "fluo", "ethyl"]

    #groups = ["depressants", "steroid", "ceptives", "urates", "amines", "azines", 
              #"phenones", "inhib", "coagul", "acids", "NSAID", "TCA", "SSRI", "MAO"]

    if should_look_up:
        if (word.lower() in SimpleDrugDb): 
            return True, "drug"
        if (word.lower() in DrugBank["drug"]):
            return True, "drug"
        if (word.lower() in DrugBank["brand"]):
            return True, "brand"
        if (word.lower() in DrugBank["group"]):
            return True, "group"
        else: 
            return False, ""
    #if (word[-5:] in fives):
        return True, "drug"
    #elif (True in [t in word for t in groups]):
        return True, "group"
    #elif (True in [t in word for t in drug_n]): 
        return True, "drug_n"
    #else:
        return False, ""

def extract_features(tokenized_sentence, should_look_up = False):
    '''
    Input:
        s: A tokenized sentence (list of triples (word, offsetFrom, offsetTo) )
        
    Output: 
        A list of feature vectors, one per token.
        Features are binary and vectors are in sparse representeation (i.e. only active features are listed)
    '''
    
    features = []

    for i in range(0, len(tokenized_sentence)):
        t = tokenized_sentence[i][0]
        punct = [".",",",";",":","?","!"]
        
        # length, number of digits, rules 
        
        tokenFeatures = [
            "form=" + t,
            "formlower=" + t.lower(),
            "suf3=" + t[-3:],
            "suf4=" + t[-4:],
            "suf5=" + t[-5:],
            "prfx3=" + t[:3],
            "prfx4=" + t[:4],
            "prfx5=" + t[:5], 
            "capitalized=%s " % t.istitle(),
            "uppercase=%s" % t.isupper(),
            "digit=%s" % t.isdigit(),
            #"hasNumber=%s" % has_numbers(t),
            "stopword=%s" % (t in stopwords),
            "punctuation=%s" % (t in punct),
            "length=%s" % len(t),
            "posTag=%s" % pos_tag(t)[0][1],
            "lemma=%s" % wordnet_lemmatizer.lemmatize(t),
            "numDigits=%s" % num_digits(t), 
            "containsDash=%s" % ('-' in t)
        ]
            
        features.append(tokenFeatures)
        
    if should_look_up:
        read_drug_list_files()
        (is_drug, isType) = token_type_classifier(t, should_look_up)
        if is_drug: 
            tokenFeatures.append("Ruled = %s" %isType) 
        else: 
            tokenFeatures.append("Ruled = O") 
            
    for i, current_token in enumerate(features):
        # add previous token
        if i > 0:
            prev_token = features[i-1][0][5:]
            current_token.append("prev=%s" % prev_token)
            current_token.append("suf3Prev = %s" % prev_token[-3:])
            current_token.append("suf4Prev = %s" % prev_token[-4:])
            current_token.append("prevIsTitle = %s" % prev_token.istitle())
            current_token.append("prevIsUpper = %s" % prev_token.isupper())
            current_token.append("PrevIsDigit = %s" % prev_token.isdigit())

        else:
            current_token.append("prev=_BoS_") #beginning of sentence
            
        # add next token
        if i < len(features)-1:
            next_token = features[i+1][0][5:]
            current_token.append("next=%s" % next_token)
            current_token.append("suf3Next = %s" % next_token[-3:])
            current_token.append("suf4Next = %s" % next_token[-3:])
            current_token.append("NextIsTitle = %s" % next_token.istitle())
            current_token.append("NextIsUpper = %s" % next_token.isupper())
            current_token.append("NextIsDigit = %s" % next_token.isdigit())

        else:
            current_token.append("next=_EoS_") # end of sentence

        # we could also add the suffixes of the previous/next word
            
    return features

def feature_extractor(datadir, resultpath, should_look_up = False):
    result_f = open(resultpath, 'w')
    # process each file in directory
    for f in listdir(datadir):

        # parse XML file, obtaining a DOM tree
        tree = parse(datadir + "/" + f)

        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes["id"].value # get sentence id
            stext = s.attributes["text"].value # get sentence text
            # load ground truth entities
            gold = []
            entities = s.getElementsByTagName("entity")
            for e in entities:
                # for discontinuous entities, we only get the first span
                offset = e.attributes["charOffset"].value      # e.g. 24-44
                try: # too many values to unpack in some iteration
                    (start, end) = offset.split(":")[0].split("-") # e.g. start:24, end:44
                except:
                    pass
                gold.append((int(start), int(end), e.attributes["type"].value)) # e.g. [(24, 44, 'drug')] 

            # tokenize text
            tokens = tokenize(stext)

            # extract features for each word in the sentence
            features = extract_features(tokens, should_look_up)

            # print features in format suitable for the learner/classifier
            for i in range (0, len(tokens)):
                # see if the token is part of an entity, and which part (B/I)
                tag = get_tag(tokens[i], gold)
                joined_features = "\t".join(features[i])
                result_f.write("{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(sid, tokens[i][0], tokens[i][1], tokens[i][2], tag, joined_features))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_to_extract_path", type=str, help = "Path to data")
    parser.add_argument("output_file_name", type=str, help="Output file name")
    parser.add_argument("-l", nargs='?', const=True, default=False, help="Include looking up in the dictionary")
    args = parser.parse_args()
    feature_extractor(args.data_to_extract_path, args.output_file_name, args.l)