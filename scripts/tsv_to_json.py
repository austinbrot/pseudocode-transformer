from csv import reader
import json

with open('data/tok-eval.tsv') as raw_tsv:
    with open('data/tok-eval.json', 'w') as dest:
        tsv = reader(raw_tsv, delimiter='\t')
        for row in tsv:
            dest.write(json.dumps({ 'translation': { 'en': row[0], 'c++': row[1] } }))
            dest.write('\n')