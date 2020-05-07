'''
This script was used to obtain the 5 LFR graphs for each of the 4 desired mu values
'''

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import community
import random
import yaml

n, tau1, tau2 = 200, 3, 1.5
av, mx, mn = 7, 35, 20

graph_gen_info = {'n': n, 'tau1': tau1, 'tau2': tau2, 'mu': 0, 'av_degree': av, 
            'max_degree': mx, 'min_c': mn, 'seed': 0, 'G': 0}

mu_vals = [0.1, 0.2, 0.3, 0.4]

scheme = ['#8F1FB4', '#E00393', '#FF486E', '#FF8851', '#FFC34C', '#F9F871', '#A47FAE', '#FFE8FF', 
          '#5EBAB1', '#0093EE', '#C1FCF5']

for mu in mu_vals:
    graph_gen_info['mu'] = mu
    for i in range(5):
        graph_gen_success = 0
        seed = random.randint(0, 5000)
        while graph_gen_success == 0:
            try:
                G = nx.generators.community.LFR_benchmark_graph(n, tau1, tau2, mu, 
                                average_degree=av, max_degree=mx, min_community=mn)
                graph_gen_success = 1
            except:
                pass
        graph_gen_info['seed'] = seed
        graph_gen_info['G'] = G
        with open('LFR_Graph_Data/mu_0_{1}/graph_0{0}/graph_0{0}_mu_0_{1}.yml'.format(i+1, int(mu*10)), 'w') as f:
            yaml.dump(graph_gen_info, f)
        pos = nx.spring_layout(G, k=0.5)
        plt.figure(1, figsize=(12,12))
        communities = []
        ground_truth_communities = []
        for node in G.nodes:
            if G.nodes[node]['community'] not in communities:
                communities.append(G.nodes[node]['community'])
            ground_truth = communities.index(G.nodes[node]['community'])
            G.nodes[node]['community'] = ground_truth
            ground_truth_communities.append(ground_truth)
        cols = [scheme[G.nodes[i]['community']] for i in range(len(G.nodes))]
        nx.draw_networkx(G, with_labels=False, pos=pos, node_color=cols, width=0.3)
        plt.savefig('LFR_Graph_Data/mu_0_{1}/graph_0{0}/graph_0{0}_mu_0_{1}.png'.format(i+1, int(mu*10)))
        plt.close()