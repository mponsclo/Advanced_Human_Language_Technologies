import argparse
from os import listdir
from xml.dom.minidom import parse
# import nltk CoreNLP module (just once)
from nltk.parse.corenlp import CoreNLPDependencyParser
# connect to your CoreNLP server (just once)
corenlp_parser = CoreNLPDependencyParser(url="http://localhost:9000")

from evaluator import evaluate

def get_offsets(word, s):
    start = s.find(word)
    end = start + len(word) -1
    return start, end

def extend_node_with_offsets(node, start, end):
    print(node.children)
    for child in node.children:
        extend_node_with_offsets(child, start, end)
    if not node.children:
        node['start'] = start
        mode['end'] = end


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

    tree, = corenlp_parser.raw_parse(s)
    print(tree.tree())
    
    return tree

def check_interaction(analysis, entities, e1, e2):
    '''
    Task:
        Decide whether a sentence is expressing a DDI between two drugs.
    
    Input:
        analysis: a DependencyGraph object with all sentence information
        entities: a list of all entities in the sentence (id and offsets)
        e1, e2: ids of the two entities to be checked
    
    Output:
        Returns the type of interaction ('effect', 'mechanism', 'advice', 'int') between e1 and e2
        expressed by the sentence, or 'None' if no interaction is described.
    '''

    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datadir", type=str, help="Path to data folder")
    parser.add_argument("output_file_name", type=str, help="Output file name")
    args = parser.parse_args()

    outf = open(args.output_file_name, 'w')
    datadir = args.datadir

    # process each file in directory
    for f in listdir (datadir):
        # parse XML file , obtaining a DOM tree
        tree = parse ( datadir + "/" + f)
        # process each sentence in the file
        sentences = tree.getElementsByTagName("sentence")
        for s in sentences:
            sid = s.attributes ["id"].value # get sentence id
            stext = s.attributes ["text"].value # get sentence text

            # load sentence entities into a dictionary
            entities = {}
            ents = s.getElementsByTagName("entity")
            for e in ents:
                eid = e . attributes ["id"]. value
                entities[eid] = e.attributes["charOffset"].value.split("-")

            # Tokenize, tag, and parse sentence
            analysis = analyze(stext)

            # for each pair in the sentence , decide whether it is DDI and its type
            pairs = s.getElementsByTagName("pair")
            for p in pairs:
                id_e1 = p.attributes["e1"].value
                id_e2 = p.attributes["e2"].value
                ddi_type = check_interaction(analysis, entities , id_e1 , id_e2 )
                if ddi_type != None :
                    print (sid +"|"+ id_e1 +"|"+ id_e2 +"|"+ ddi_type, file = outf )
    # get performance score
    # evaluate ("DDI", inputdir, outputfile)



if __name__ == "__main__":
    main()