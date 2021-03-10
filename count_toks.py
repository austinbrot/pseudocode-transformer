import json, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, help='Path to file of translation entries to count (json)')
    args = parser.parse_args()

    assert args.file != None, 'Please provide a file path using --file'
    assert args.file.split('.')[-1] == 'json', 'File must be in .json format'

    max_src_len, max_tgt_len = 0, 0
    with open(args.file) as f:
        for entry in f:
            translation = json.loads(entry)['translation']
            src, tgt = translation['en'], translation['c++']
            max_src_len = max(max_src_len, len(src.split(' ')))
            max_tgt_len = max(max_tgt_len, len(tgt.split(' ')) + 3)
    print('Max src length: ' + str(max_src_len) + ' tokens')
    print('Max tgt length: ' + str(max_tgt_len) + ' tokens')


if __name__ == '__main__':
    main()