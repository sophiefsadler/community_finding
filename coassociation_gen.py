"""
Calculate the coassociation matrix from the runs of the community finding
algorithm.

The parts_folder is where the generated partitions have been saved. The save
location of the coassociation matrices will be determined automatically
relative to this.

Usage:
  coassociation_gen.py <parts_folder>

Options:
  -h --help            Show this help message

"""

import os
import numpy as np
import itertools

from docopt import docopt
from tqdm import tqdm, trange


def calc_C(C, partitions, n_nodes):
    for i in trange(partitions.shape[0]):
        part = partitions[i, :]
        for node1, node2 in itertools.combinations(np.arange(n_nodes), r=2):
            if part[node1] == part[node2]:
                C[node1,node2] += 1
                C[node2,node1] += 1 
    C /= partitions.shape[0]
    np.fill_diagonal(C,1)
    return C


if __name__ == '__main__':
    args = docopt(__doc__)
    parts_folder = args.get('<parts_folder>')
    parts_files = [x for x in os.listdir(parts_folder) if x.endswith('.npy')]
    for fil in tqdm(parts_files):
        fil_path = os.path.join(parts_folder, fil)
        parts = np.load(fil_path)
        n_nodes = parts.shape[1]
        C = np.zeros((n_nodes, n_nodes))
        C = calc_C(C, parts, n_nodes)
        coassociation_fil = fil.strip('runs.npy') + 'coassociation.npy'
        coassociation_folder = parts_folder.strip('/').strip('Runs') + 'Coassociation'
        coassociation_file = os.path.join(coassociation_folder, coassociation_fil)
        np.save(coassociation_file, C)
