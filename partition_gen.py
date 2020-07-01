"""
Run a community finding algorithm for many runs to obtain data from which to
calculate the coassociation matrix and certain node features.

Usage:
  partition_gen.py (louvain | infomap | lpa) (200 | 1000)

Options:
  -h --help            Show this help message

"""

import time
import random
import os
import numpy as np
import yaml
import pickle

import community
from networkx.algorithms.community.centrality import girvan_newman
from networkx.algorithms.community import modularity
from networkx.algorithms.community.label_propagation import asyn_lpa_communities
from infomap import Infomap

from docopt import docopt
from tqdm import tqdm, trange


def calc_louvain(G):
    print('Calculating partitions')
    partitions = []
    seeds = []
    for k in trange(1000):
        seed = int(time.time() * 1000)
        random.seed(seed)
        seeds.append(seed)
        partition = community.best_partition(G)
        partition_list = []
        for _, comm_index in partition.items():
            partition_list.append(comm_index+1) # Louvain indexes from 0, so add 1
        partitions.append(partition_list)
    partitions = np.array(partitions)
    return partitions, seeds


def calc_infomap(G, n_nodes):
    print('Calculating partitions')
    partitions = []
    seeds = []
    for k in trange(1000):
        seed = int(time.time() * 1000)
        seeds.append(seed)
        im = Infomap('-s {0}'.format(seed))
        im.add_nodes(range(n_nodes))
        im.add_links(list(G.edges()))
        im.run('--silent')
        partition = im.get_modules()
        partition_list = []
        for _, comm_index in partition.items():
            partition_list.append(comm_index)
        partitions.append(partition_list)
    partitions = np.array(partitions)
    return partitions, seeds


def calc_lpa(G, n_nodes):
    print('Calculating partitions')
    partitions = []
    seeds = []
    for k in trange(1000):
        seed = int(time.time() * 1000)
        random.seed(seed)
        seeds.append(seed)
        partition_list = [0 for _ in range(n_nodes)]
        partition = list(asyn_lpa_communities(G))
        for comm_index, comm in enumerate(partition):
            for node in comm:
                partition_list[node] = comm_index + 1 # Index from 1
        partitions.append(partition_list)
    partitions = np.array(partitions)
    return partitions, seeds


def calc_partitions(G, args, n_nodes):
    '''
    In all cases, the partitions returned at the end should be a 1000 x n numpy array,
    where 1000 is the number of runs of the community finding algorithm, and n is the
    number of nodes for the graph. Each entry in the array is the integer community label
    for that node during that run of the algorithm.

    For consistency, all outputs will index communities from 1.
    '''
    if args.get('louvain'):
        partitions, seeds = calc_louvain(G)
        folder = 'Louvain'
    elif args.get('infomap'):
        partitions, seeds = calc_infomap(G, n_nodes)
        folder = 'Infomap'
    elif args.get('lpa'):
        partitions, seeds = calc_lpa(G, n_nodes)
        folder = 'LPA'
    return partitions, seeds, folder


if __name__ == '__main__':
    args = docopt(__doc__)
    if args.get('200'):
        n_nodes = 200
    elif args.get('1000'):
        n_nodes = 1000
    for i in [1, 2, 3, 4]:
        for j in [1, 2, 3, 4, 5]:
            with open('LFR_Graph_Data/{2}_Node/mu_0_{0}/graph_0{1}/graph_0{1}_mu_0_{0}.yml'.format(i, j, n_nodes)) as f:
                graph_info = yaml.load(f, Loader=yaml.Loader)
            G = graph_info['G']
            for node in G.nodes:
                del G.nodes[node]['community']
            partitions, seeds, folder = calc_partitions(G, args, n_nodes)
            path = os.path.join('LFR_Graph_Data/{0}_Node/'.format(n_nodes), 'Community_Data', folder, 'Runs', 
                                'graph_0{1}_mu_0_{0}_runs.npy'.format(i, j))
            np.save(path, partitions)
            seeds_path = os.path.join('LFR_Graph_Data/{0}_Node/'.format(n_nodes), 'Community_Data', folder, 'Runs', 
                                    'graph_0{1}_mu_0_{0}_seeds'.format(i, j))
            with open(seeds_path, 'wb') as fp:
                pickle.dump(seeds, fp)