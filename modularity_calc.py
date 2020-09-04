import numpy as np
import networkx as nx
import yaml
import community
from tqdm import tqdm
import pickle

def mean_modularity(G, runs):
    modularities = []
    for i in range(1000):
        part = runs[i, :]
        part_dict = {node: comm for node, comm in enumerate(part)}
        modularity = community.modularity(part_dict, G)
        modularities.append(modularity)
    mean_modularity = np.mean(modularities)
    return mean_modularity

modularities_dict = {}

for algo in ['Louvain', 'Infomap', 'LPA']:
    for mu in [1, 2, 3, 4]:
        graph_modularities = []
        for graph in tqdm([1, 2, 3, 4, 5]):
            runs_path = 'LFR_Graph_Data/1000_Node_Params/Community_Data/{2}/Runs/graph_0{0}_mu_0_{1}_runs.npy'.format(graph, mu, algo)
            graph_path = 'LFR_Graph_Data/1000_Node_Params/mu_0_{1}/graph_0{0}/graph_0{0}_mu_0_{1}.yml'.format(graph, mu)
            runs = np.load(runs_path)
            with open(graph_path) as f:
                graph_info = yaml.load(f, Loader=yaml.Loader)
            G = graph_info['G']
            graph_modularity = mean_modularity(G, runs)
            graph_modularities.append(graph_modularity)
        mu_mean_modularity = np.mean(graph_modularities)
        modularities_dict['{1}_mu_0_{0}'.format(mu, algo)] = mu_mean_modularity

    G = nx.read_edgelist('Node_Feature_Experiments/6-email-eu-core/email-eu-core.edges', nodetype=int)
    runs = np.load('Node_Feature_Experiments/6-email-eu-core/{0}/runs.npy'.format(algo))
    email_modularity = mean_modularity(G, runs)
    modularities_dict['{0}_email_eu_core'.format(algo)] = email_modularity

with open('modularities_params', 'wb') as fp:
    pickle.dump(modularities_dict, fp)