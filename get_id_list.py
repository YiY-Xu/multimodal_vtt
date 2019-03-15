import numpy as np
import sys
from tqdm import tqdm

if __name__ == '__main__':
    path = sys.argv[1] + '/cmvn.scp'
    output = sys.argv[2]

    ids = []

    with open(path, 'r') as f:
        for line in tqdm(f.readlines()):
            ids.append(line.split(' ')[0])

    with open(output, 'w') as f:
        for item in ids:
            f.write("%s\n" % item)