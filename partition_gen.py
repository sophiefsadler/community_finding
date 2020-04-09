"""
Run a community finding algorithm for many runs to obtain data from which to
calculate the coassociation matrix and certain node features.

Usage:
  partition_gen.py (louvain | gn | infomap | cfinder | infomod)

Options:
  -h --help            Show this help message

"""

import numpy as np

import community
from infomap import Infomap

from docopt import docopt
from tqdm import tqdm, trange


def calc_louvain(G):
    print('Calculating partitions')
    partitions = []
    for k in trange(1000):
        partition = community.best_partition(G)
        partition_list = []
        for _, comm_index in partition.items():
            partition_list.append(comm_index)
        partitions.append(partition_list)
    partitions = np.array(partitions)
    return partitions


def calc_gn(G):
    return partitions


def calc_infomap(G):
    print('Calculating partitions')
    partitions = []
    for k in trange(1000):
        im = Infomap()
        im.add_edges(range(200))
        im.add_links(list(G.edges()))
        im.run()
        partition = im.get_modules()
        partition_list = []
        for _, comm_index in partition.items():
            partition_list.append(comm_index)
        partitions.append(partition_list)
    partitions = np.array(partitions)
    return partitions


def calc_cfinder(G):
    return partitions


def calc_infomod(G):
    return partitions


def calc_partitions(G, args):
    '''
    In all cases, the partitions returned at the end should be a 1000 x n numpy array,
    where 1000 is the number of runs of the community finding algorithm, and n is the
    number of nodes for the graph. Each entry in the array is the integer community label
    for that node during that run of the algorithm.
    '''
    if args.get('louvain'):
        partitions = calc_louvain(G)
    elif args.get('gn'):
        partitions = calc_gn(G)
    elif args.get('infomap'):
        partitions = calc_infomap(G)
    elif args.get('cfinder'):
        partitions = calc_cfinder(G)
    elif args.get('infomod'):
        partitions = calc_infomod(G)
    return partitions


if __name__ == '__main__':
    args = docopt(__doc__)
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4, 5]:
            with open('lfr_graphs/mu_0_{0}/graph_0{1}/graph_0{1}_mu_0_{0}.yml'.format(i, j)) as f:
                graph_info = yaml.load(f, Loader=yaml.Loader)
            G = graph_info['G']
            for node in G.nodes:
                del G.nodes[node]['community']
            partitions = calc_partitions(G, args)