import argparse, json
from csv import reader
from dataset_sanity_check import sanitycheck_dataset

def comp(data_src, data_tgt, ref_src, ref_tgt):
    return data_src.split(' [PRED] ')[-1] == ref_src and data_tgt == ref_tgt

def make_prefix(prefix_lines: list):
    if len(prefix_lines) == 0: return ''
    return ' [ENDL] '.join(prefix_lines) + ' [ENDL] '

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file', type=str, default=None, help='Source file path (tsv)')
    parser.add_argument('--tokenized_in_file', type=str, default=None, help='Source file tokenized path (tsv)')
    parser.add_argument('--out_file', type=str, default=None, help='Output file path (json)')
    args = parser.parse_args()

    assert args.in_file != None, 'Provide a path to a .tsv with input'
    assert args.out_file != None, 'Provide an output file path'

    with open(args.in_file) as raw_infile:
        with open(args.out_file, 'w') as outfile:
            with open(args.tokenized_in_file) as tok_target:
                tok_target = reader(tok_target, delimiter='\t')
                tsv = reader(raw_infile, delimiter='\t')
                next(tsv) # skip header

                code_pfx = []
                for i, row in enumerate(tsv):
                    source = row[0]

                    if i == 0: source_pfx = []
                    
                    if source:
                        try:
                            tok_entry = next(tok_target)
                            source, target = tok_entry[0], tok_entry[1]
                        except StopIteration:
                            target = row[1]

                        prefix = make_prefix(code_pfx)
                        outfile.write(json.dumps({ 'translation': { 'en': '[PRE] ' + prefix +  ' [PRED] ' + source , 'c++': target } }))
                        outfile.write('\n')

                        code_pfx = code_pfx[-9:] + [target]
    sanitycheck_dataset(args.out_file, args.tokenized_in_file, comp)

if __name__ == '__main__':
    main()