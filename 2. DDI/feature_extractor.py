from os import listdir
from xml.dom.minidom import parse
import networkx
import argparse
import string
# import nltk CoreNLP module (just once)
from nltk.parse.corenlp import CoreNLPDependencyParser
# connect to your CoreNLP server (just once)
corenlp_parser = CoreNLPDependencyParser(url="http://localhost:9000")


from evaluator import *

def do_indices_overlap(start1, end1, start2, end2):
    if start1 == start2 and end1==end2:
        return True

def find_entity_in_tree(eid, entities, tree):
    start_e1 = int(entities[eid][0])
    end_e1 = int(entities[eid][1].split(';')[0])

    for n in tree.nodes.items():
        node = n[1]
        if node['word'] and (node['start'] == start_e1 or node['end'] == end_e1):
            return node

def get_offsets(word, s):
    '''
    Task:
        Given a word and sentence, returns its starting end ending index in the sentence.
    
    Input:
        word: word to find offsets for
        s: sentence containing the word
    
    Output:
        Returns a tuple containing the start and end offset.
    '''
    start = s.find(word)
    end = start + len(word) - 1
    return start, end

def preprocess(s):
    '''
    Task:
        Helper function
    '''
    # because otherwise CoreNLP throws 500
    return s.replace("%", "<percentage>")

def analyze(s):
    '''
    Task:
        Given one sentence, sends it to CoreNLP to obtain the tokens, tags,
        and dependency tree. It also adds the start/end offsets to each token.
    
    Input:
        s: string containing the text for one sentence
    
    Output:
        Returns the nltk DependencyGraph object produced by CoreNLP, enriched with token  offsets.

    '''
    s = s.replace("%", "<percentage>")
    tree, = corenlp_parser.raw_parse(s)
    for n in tree.nodes.items():
        node = n[1]
        if node['word']:
            start, end = get_offsets(node['word'], s)
            node['start'] = start
            node['end'] = end
            
    return tree

CLUE_VERBS = ['administer', 'enhance', 'interact', 'coadminister', 'increase', 'decrease'] # add more?
NEGATIVE_WORDS = ['No', 'not', 'neither', 'without', 'lack', 'fail', 'unable', 'abrogate',
                  'absence', 'prevent','unlikely', 'unchanged', 'rarely']

def find_clue_verbs(path, tree):
    path_nodes = [tree.nodes[x]['lemma'] for x in path]
    feats = []
    for pn in path_nodes:
        if pn in CLUE_VERBS:
            feats.append('lemmainbetween=%s' % pn)
            
    return feats

def negative_words_path(path, tree):
    path_nodes = [tree.nodes[x]['word'] for x in path]
    count = 0
    for pn in path_nodes:
        if pn in NEGATIVE_WORDS or pn[-3:] == "n't":
            count += 1
    return count

def negative_words_sentence(tree):
    count = 0
    for n in tree.nodes.items():
        word = n[1]['word']
        if word in NEGATIVE_WORDS:
            count += 1
    return count

def traverse_path(path, tree):
    if len(path) == 0:
        return None, None
    path_nodes = [tree.nodes[x] for x in path]
    str_path = ""
    # traverse from e1 up
    current_node = path_nodes[0]
    while (current_node['head'] in path):
        rel = current_node['rel']
        current_node = tree.nodes[current_node['head']]
        str_path += (rel + '<')

    tag_path = str_path + current_node['tag']
    str_path += current_node['lemma']
    # traverse from e2 up
    current_node = path_nodes[-1]
    while(current_node['head'] in path):
        rel = current_node['rel']
        current_node = tree.nodes[current_node['head']]
        str_path += ('>' + rel)
        tag_path += ('>' + rel)

    return str_path, tag_path

def find_words_outside_path(path, tree):
    words = []
    for node in tree.nodes.items():
        node = node[1]
        if node['address'] not in path and node['lemma'] and node['lemma'] not in string.punctuation and not node['lemma'].isdigit():
            words.append(node['lemma'])
    return words



def find_head(tree, entity):
    for n in tree.nodes.items():
            node = n[1]
            if  node['address'] == entity['head']:
                return node
    
    
def extract_features(tree, entities, e1, e2) :
    '''
    Task:
        Given an analyzed sentence and two target entities , compute a feature
        vector for this classification example .
    Input:
        tree: a DependencyGraph object with all sentence information .
        entities: A list of all entities in the sentence (id and offsets).
        e1, e2: ids of the two entities to be checked for an interaction
    Output:
        A vector of binary features .
        Features are binary and vectors are in sparse representation (i.e. only
        active features are listed)
   '''
    int_verbs = ['interact', 'interaction']
    mech_verbs = ['metabolism', 'concentration', 'clearance', 'level', 'absorption', 'dose',
                 'presence', 'interfere']
    adv_verbs = ['co-administration', 'take', 'coadminister', 'treatment', 'therapy', 'tell']
    eff_verbs = ['effect', 'alcohol', 'action','use', 'combination', 'inhibitor',
                'response', 'effect', 'enhance', 'diminish']
      
    e1_node = find_entity_in_tree(e1, entities, tree)
    e2_node = find_entity_in_tree(e2, entities, tree)
    
    e1_head = find_head(tree, e1_node) if e1_node else None
    e2_head = find_head(tree, e2_node) if e2_node else None
    
    h1_lemma = e1_head['lemma'] if e1_node else None
    h2_lemma = e2_head['lemma'] if e2_node else None
    
    tag_head_e1 = e1_head['tag'] if e1_head else None
    tag_head_e2 = e2_head['tag'] if e2_head else None
    
    nxgraph = tree.nx_graph().to_undirected()
    shortest_path = networkx.shortest_path(nxgraph, e1_node['address'], e2_node['address']) if (e1_node and e2_node) else []
    path_with_word, path_with_tag = traverse_path(shortest_path, tree)
    find_clue_verbs(shortest_path, tree)
    count_neg_p = negative_words_path(shortest_path, tree)
    count_neg_s = negative_words_sentence(tree)

    
    # --- FEATURES ---
    features = ['h1_lemma=%s' %h1_lemma,
                'h2_lemma=%s' %h2_lemma,
                'h1_tag=%s' %tag_head_e1,
                'h2_tag=%s' %tag_head_e2,
                'path=%s' % path_with_word,
                'tagpath=%s' % path_with_tag,
                'neg_words_p=%s' %count_neg_p,  # only 28 with 1, 1 with 2
                'neg_words_s=%s' %count_neg_s   # 3144 with 1, 270 with 2, 4 with 3 
                ] + find_clue_verbs(shortest_path, tree)
    
    
    if (e1_head and e2_head):
        if h1_lemma == h2_lemma:  # should use address? 
            features.append('under_same=True') # 5609 occurrences
            if tag_head_e1[0].lower() == 'v':
                features.append('under_same_verb=True') # 173 occurrences
            # else:
            #     features.append('under_same_verb=False')
        # else:
        #     features.append('under_same=False')
        #     features.append('under_same_verb=False')
            
        if h1_lemma == e2_node['lemma']:
            features.append('1under2=True') # 136 occ
        # else:
        #     features.append('1under2=False')
            
        if h2_lemma == e1_node['lemma']:
            features.append('2under1=True') # 1953 occ
        # else:
        #     features.append('2under1=False')
        
        if h1_lemma in int_verbs or h2_lemma in int_verbs:
            features.append('intVerbs=True') # 458
        # else:
        #     features.append('intVerbs=False')
            
        if h1_lemma in mech_verbs or h2_lemma in mech_verbs:
            features.append('mechVerbs=True') # 1030
        # else:
        #     features.append('mechVerbs=False')

        if h1_lemma in adv_verbs or h2_lemma in adv_verbs:
            features.append('advVerbs=True') # 569
        # else:
        #     features.append('advVerbs=False')

        if h1_lemma in eff_verbs or h2_lemma in eff_verbs:
            features.append('effVerbs=True') # 3480
        # else:
        #     features.append('effVerbs=False')
        
    else:
        None

    words_outside_path = find_words_outside_path(shortest_path, tree)
    for word in words_outside_path:
        features.append(f'lemmaoutside={word}')
        
    # Distance between entities 
    # Type of entities (drug, brand, group) in the pair
        
    return features



def main(datadir):
    for f in listdir(datadir):
        # parse XML file , obtaining a DOM tree
        tree = parse(datadir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:

            sid = s.attributes["id"].value # get sentence id
            stext = s.attributes["text"].value # get sentence text
            
            # CoreNLP throws error for empty sentences
            if len(stext) == 0:
                continue

            # load sentence ground truth entities
            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                eid = e . attributes["id"].value
                entities[eid] = e.attributes["charOffset"].value.split("-")

            # analyze sentence if there is at least a pair of entities
            if len(entities) > 1: analysis = analyze(stext)

            # for each pair in the sentence , decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                # get ground truth
                ddi = p.attributes["ddi"].value
                dditype = p.attributes["type"].value if ddi == "true" else "null"
                
                # target entities
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                
                # feature extraction
                feats = extract_features(analysis, entities, id_e1, id_e2)
                
                # resulting feature vector
                print(sid, id_e1, id_e2, dditype, "\t".join(feats), sep="\t") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_to_extract_path", type=str, help = "Path to data")
    args = parser.parse_args()
    main(args.data_to_extract_path)