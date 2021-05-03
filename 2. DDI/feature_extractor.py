from os import listdir
from xml.dom.minidom import parse
import networkx
import argparse
# import nltk CoreNLP module (just once)
from nltk.parse.corenlp import CoreNLPDependencyParser
# connect to your CoreNLP server (just once)
corenlp_parser = CoreNLPDependencyParser(url="http://localhost:9000")


from evaluator import *

def find_entity_in_tree(eid, entities, tree):
    start_e1 = entities[eid][0]
    end_e1 = entities[eid][1]
    
    for n in tree.nodes.items():
        node = n[1]
        if node['word'] and node['start'] == int(start_e1): # and node['end'] == int(end_e1)):
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

def find_clue_verbs(path, tree):
    path_nodes = [tree.nodes[x]['lemma'] for x in path]
    feats = []
    for pn in path_nodes:
        if pn in CLUE_VERBS:
            feats.append('lemmainbetween=%s' % pn)
            
    return feats

def traverse_path(path, tree):
    if len(path) == 0:
        return None
    path_nodes = [tree.nodes[x] for x in path]
    str_path = ""
    # traverse from e1 up
    current_node = path_nodes[0]
    while (current_node['head'] in path):
        
        rel = current_node['rel']
        current_node = tree.nodes[current_node['head']]
        str_path += (rel + '<')
    
    str_path += current_node['lemma']
    # traverse from e2 up
    current_node = path_nodes[-1]
    while(current_node['head'] in path):
        rel = current_node['rel']
        current_node = tree.nodes[current_node['head']]
        str_path += ('>' + rel)
        
    return str_path

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
    path = traverse_path(shortest_path, tree)
    clue_verbs = find_clue_verbs(shortest_path, tree)

    
    # --- FEATURES ---
    features = ['h1_lemma=%s' %h1_lemma,
                'h2_lemma=%s' %h2_lemma,
                'h1_tag=%s' %tag_head_e1,
                'h2_tag=%s' %tag_head_e2,
                'path=%s' % path
                
                ] + clue_verbs
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
                print(sid, id_e1, id_e2, dditype, "|".join(feats), sep="|") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_to_extract_path", type=str, help = "Path to data")
    args = parser.parse_args()
    main(args.data_to_extract_path)