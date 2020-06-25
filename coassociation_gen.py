"""
Calculate the coassociation matrix from the runs of the community finding
algorithm.

Usage:
  coassociation_gen.py (louvain | infomap | lpa) (200 | 1000)

Options:
  -h --help            Show this help message

"""

import os
import numpy as np
import yaml
import itertools

from docopt import docopt
from tqdm import tqdm, trange


def get_file_names(i,j, args, n_nodes):
    if args.get('louvain'):
        folder = 'Louvain'
    elif args.get('infomap'):
        folder = 'Infomap'
    elif args.get('lpa'):
        folder = 'LPA'
    parts_file = os.path.join('LFR_Graph_Data/{0}_Node/'.format(n_nodes), 'Community_Data', folder, 'Runs',
                              'graph_0{0}_mu_0_{1}_runs.npy'.format(j,i))
    coassociation_file = os.path.join('LFR_Graph_Data/{0}_Node/'.format(n_nodes), 'Community_Data', folder, 'Coassociation',
                                      'graph_0{0}_mu_0_{1}_coassociation.npy'.format(j,i))
    return parts_file, coassociation_file


def calc_C(C, partitions):
    print('Applying partitions to C')
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
    if args.get('200'):
        n_nodes = 200
    elif args.get('1000'):
        n_nodes = 1000
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4, 5]:
            parts_file, coassociation_file = get_file_names(i, j, args, n_nodes)
            parts = np.load(parts_file)
            C = np.zeros((n_nodes, n_nodes))
            C = calc_C(C, parts, n_nodes)
            np.save(coassociation_file, C)