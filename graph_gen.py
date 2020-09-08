"""
Generate a specified number of graphs, at a specified mu level and allocate the folder
to save the graphs into. If the folder doesn't exist, it will be created.

Usage:
  graph_gen.py <mu> <folder> [--graphs=<gn>]

Options:
  -h --help            Show this help message
  --graphs=<gn>        Number of graphs to generate [default: 5]

"""

import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from docopt import docopt
import community
import random
import yaml
from tqdm import trange

def graph_info(mu):
    # Parameters used for 200 node graphs
    # n, tau1, tau2 = 200, 3, 1.5
    # av, mx, mn = 7, 35, 20

    # Parameters used for 1000 node graphs
    # n, tau1, tau2 = 1000, 3, 1.5
    # av, mx, mn = 25, 250, 100

    # Parameters more closely matching LFR papers
    n, tau1, tau2 = 1000, 3, 2
    av, mx = 20, 50

    # Parameters exactly matching email-eu-core graph
    # n, tau1, tau2 = 1000, 5.5, 2.6
    # av, mx = 33, 350

    graph_gen_info = {'n': n, 'tau1': tau1, 'tau2': tau2, 'mu': mu, 'av_degree': av, 
                      'max_degree': mx, 'seed': 0, 'G': 0}

    return graph_gen_info

def generate_graph(mu, folder, info):
    graph_gen_success = 0
    seed = random.randint(0, 5000)
    while graph_gen_success == 0:
        try:
            G = nx.generators.community.LFR_benchmark_graph(info['n'], info['tau1'], info['tau2'], mu, 
                                                            average_degree=info['av_degree'], 
                                                            max_degree=info['max_degree'])
            graph_gen_success = 1
        except:
            pass
    info['seed'] = seed
    info['G'] = G
    with open(os.path.join(folder, 'graph_0{0}/graph_0{0}_mu_0_{1}.yml'.format(i+1, int(mu*10))), 'w') as f:
        yaml.dump(info, f)
    pos = nx.spring_layout(G, k=0.5)
    plt.figure(1, figsize=(20,20)) #Figsize (20,20) for 1000 node graph required, but (12,12) fine for 200 node
    communities = []
    ground_truth_communities = []
    for node in G.nodes:
        if G.nodes[node]['community'] not in communities:
            communities.append(G.nodes[node]['community'])
        ground_truth = communities.index(G.nodes[node]['community'])
        G.nodes[node]['community'] = ground_truth
        ground_truth_communities.append(ground_truth)
    nx.draw(G, pos, with_labels=False, node_color=ground_truth_communities, width=0.3, cmap=plt.cm.get_cmap('rainbow'))
    plt.savefig(os.path.join(folder, 'graph_0{0}/graph_0{0}_mu_0_{1}.png'.format(i+1, int(mu*10))))
    plt.close()
    return

if __name__ == '__main__':
    args = docopt(__doc__)
    mu = float(args.get('<mu>'))
    folder = args.get('<folder>')
    gn = int(args.get('--graphs'))
    if not os.path.exists(folder):
        os.mkdir(folder)
    info = graph_info(mu)
    for i in trange(gn):
        if not os.path.exists(os.path.join(folder, 'graph_0{0}'.format(i+1))):
            os.mkdir(os.path.join(folder, 'graph_0{0}'.format(i+1)))
        generate_graph(mu, folder, info)
