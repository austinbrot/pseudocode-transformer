import json
from csv import reader

def sanitycheck_dataset(dataset_fname: str, reference_fname: str, cmp):
    with open(dataset_fname) as dataset:
        with open(reference_fname) as reference:
            reference = reader(reference, delimiter='\t')

            # try:
            perfect = True
            for i, (data_entry, reference_entry) in enumerate(zip(dataset, reference)):
                data_entry = json.loads(data_entry)['translation']
                data_src, data_tgt = data_entry['en'], data_entry['c++']
                reference_src, reference_tgt = reference_entry[0], reference_entry[1]
                if not cmp(data_src, data_tgt, reference_src, reference_tgt):
                    print('Fails at entry ' + str(i))
                    print('Expected src: ' + reference_src)
                    print('Actual src: ' + data_src)
                    print('Expected tgt: ' + reference_tgt)
                    print('Actual tgt: ' + data_tgt)
                    print()
                    perfect = False
            if perfect: print('No errors found!')
            # except Exception:
            #     print(exception)
