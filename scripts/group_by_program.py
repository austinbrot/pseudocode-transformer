import argparse, json
from csv import reader

parser = argparse.ArgumentParser()
parser.add_argument('--in_file', type=str, default=None, help='Source file path (tsv)')
parser.add_argument('--tokenized_in_file', type=str, default=None, help='Source file tokenized path (tsv)')
parser.add_argument('--out_file', type=str, default=None, help='Output file path (json)')
args = parser.parse_args()

assert args.in_file != None, 'Provide a path to a .tsv with input'
assert args.out_file != None, 'Provide an output file path'

with open(args.in_file) as raw_infile:
    with open(args.out_file, 'w') as outfile:
        with open(args.tokenized_in_file) as tok_infile:
            tok_infile = reader(tok_infile, delimiter='\t')
            tsv = reader(raw_infile, delimiter='\t')
            source_lines, target_lines = [], []
            for i, row in enumerate(tsv):
                if row[6] == '0' and i != 0:
                    outfile.write(json.dumps({ 'translation': { 'en': ' [ENDL] '.join(source_lines), 'c++': ' '.join(target_lines) } }))
                    outfile.write('\n')
                    source_lines.clear()
                    target_lines.clear()

                if row[0] == '':
                    target_lines.append(row[1])
                else:
                    try:
                        next_tok_row = next(tok_infile)
                        source_lines.append(next_tok_row[0])
                        target_lines.append(next_tok_row[1])
                    except:
                        source_lines.append(row[0])
                        target_lines.append(row[1])


