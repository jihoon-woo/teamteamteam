import jsonlines
import json
import nltk
from nltk import word_tokenize, pos_tag

data = []
queries = [] # List containing all the original queries
with jsonlines.open("highlight_train_release.jsonl") as f:
    for line in f.iter():
        queries.append(line['query'])
        data.append(line)

nouns_verbs = [] # List containing nouns and verbs of the queries
for query in queries:
    tokens = word_tokenize(query)
    words = nltk.pos_tag(tokens)
    noun_verb_list = list(filter(lambda x: x[1] == "NN" or x[1] == "NNS" or x[1] == "NNP" or x[1] == "NNPS"
                                  or x[1] == "VB" or x[1] == "VBD" or x[1] == "VBG" or x[1] == "VBZ", words))
    only_words = [noun for (noun, type) in noun_verb_list] # Only extract word from the nltk tuple
    
    nouns_verbs.append(' '.join(only_words))

id = 0 # To assign qids to newly augmented data
for d in data:
    if d['qid'] > id:
        id = d['qid']
id += 1

for i in range(len(data)): # Append newly created data to original train dataset jsonl file
    new_data = data[i]
    new_data['qid'] = id
    new_data['query'] = nouns_verbs[i]
    with open("highlight_train_release.jsonl", "a") as f:
        f.write("\n")
        json.dump(new_data, f)
    id += 1