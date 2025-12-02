import random, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default=None, help='Path to file to shuffle')
    args = parser.parse_args()

    assert args.file != None, 'Please provide a file path using --file'

    infile = args.file
    outfile = args.file.split('.')
    outfile = '.'.join(outfile[:-1]) + '-shuf.json'

    with open(infile,'r') as source:
        data = [ (random.random(), line) for line in source ]
    data.sort()
    with open(outfile,'w') as target:
        for _, line in data:
            target.write( line )

if __name__ == '__main__':
    main()