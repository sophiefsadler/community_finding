"""
Calculate the node features for every graph, given the coassociation matrix
produced by each of the algorithms.

Usage:
  node_feature_calc.py (louvain | infomap | lpa)

Options:
  -h --help            Show this help message

"""

import os
import yaml
import math
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import entropy
from sklearn.model_selection import train_test_split

from docopt import docopt
from tqdm import tqdm, trange


def calc_node_metrics(G):
    node_degrees = dict(G.degree())
    node_clustering_coefficients = nx.clustering(G)
    node_betweenness = nx.betweenness_centrality(G)
    node_closeness = nx.closeness_centrality(G)
    node_eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
    node_av_shortest_paths = {}
    for i in range(G.number_of_nodes()):
        shortest_paths = nx.algorithms.shortest_paths.generic.shortest_path_length(G, source=i)
        if list(shortest_paths.values())[1:] != []:
            average_shortest_path = np.mean(list(shortest_paths.values())[1:])
        else:
            average_shortest_path = 0
        node_av_shortest_paths[i] = average_shortest_path
    node_metrics = {'Degree': node_degrees, 'Clustering Coefficient': node_clustering_coefficients, 'Betweenness': node_betweenness, 
                    'Closeness': node_closeness, 'Shortest Path': node_av_shortest_paths, 'Eigenvector': node_eigenvector}
    return node_metrics, node_degrees


def convert_parts_format(parts):
    final_parts_list = []
    for i in range(parts.shape[0]):
        current_part = parts[i, :]
        converted_part = [[] for _ in range(max(current_part))]
        for node, comm in enumerate(current_part):
            converted_part[comm - 1].append(node) # Subtract 1 since communities are indexed from 1
        final_parts_list.append(converted_part)
    return final_parts_list


def initialise_new_metrics():
    e_in_list = {i: [] for i in range(200)}
    e_out_list = {i: [] for i in range(200)}

    e_in_over_e_out = {i: [] for i in range(200)}
    odf = {i: [] for i in range(200)}

    expansion = {i: [] for i in range(200)}
    cut_ratio = {i: [] for i in range(200)}
    conductance = {i: [] for i in range(200)}
    normalised_cut = {i: [] for i in range(200)}

    triangle_participation = {i: [] for i in range(200)}
    
    new_metric_dict = {'E In': e_in_list, 'E Out': e_out_list, 'E In Over E Out': e_in_over_e_out,
                       'ODF': odf, 'Expansion': expansion, 'Cut Ratio': cut_ratio,
                       'Conductance': conductance, 'Normalised Cut': normalised_cut, 
                       'Triangle Participation': triangle_participation}
    return new_metric_dict


def calc_new_metrics(new_metrics, G, partitions, node_degrees):
    for part in partitions:
        for comm in part:

            comm_subgraph = G.subgraph(comm)
            comm_degrees = comm_subgraph.degree()
            
            w = len(comm)
            N = G.number_of_nodes()
            m = G.number_of_edges()

            # In order to find the triangle participation for all nodes, find all the triangles in a community
            all_cliques = nx.enumerate_all_cliques(comm_subgraph)
            triangle_cliques = [k for k in all_cliques if len(k) == 3]

            for nod in dict(comm_degrees).keys():
                e_in = comm_degrees[nod]
                e_out = node_degrees[nod] - e_in

                new_metrics['E In'][nod].append(e_in)
                new_metrics['E Out'][nod].append(e_out)

                # For e_in divided by e_out, if e_out is 0, just return the value of e_in
                try:
                    new_metrics['E In Over E Out'][nod].append(e_in/e_out)
                except ZeroDivisionError:
                    new_metrics['E In Over E Out'][nod].append(e_in)

                new_metrics['ODF'][nod].append(e_out/node_degrees[nod])

                new_metrics['Expansion'][nod].append(e_out/w)
                try:
                    new_metrics['Cut Ratio'][nod].append(e_out/(N-w))
                except ZeroDivisionError:
                    new_metrics['Cut Ratio'][nod].append(0)

                ct = e_out/(node_degrees[nod] + e_in)
                new_metrics['Conductance'][nod].append(ct)

                nc = ct + e_out/(2*m - 2*e_in + e_out)
                new_metrics['Normalised Cut'][nod].append(nc)

                # Calculate triangle participation
                nods_in_triangles = []
                for triangle in triangle_cliques:
                    if nod in triangle:
                        nods_in_triangles += triangle
                tp = len(set(nods_in_triangles))/w
                new_metrics['Triangle Participation'][nod].append(tp)
                
    return new_metrics


def average_metrics(new_metrics):
    averaged_metrics = new_metrics.copy()
    for met in averaged_metrics.keys():
        for nod in averaged_metrics[met].keys():
            averaged_metrics[met][nod] = np.mean(new_metrics[met][nod])
    return averaged_metrics


def new_node_metrics(node_metrics, partitions, node_degrees):
    new_metrics = initialise_new_metrics()
    new_metrics = calc_new_metrics(new_metrics, G, partitions, node_degrees)
    new_metrics = average_metrics(new_metrics)
    updated_node_metrics = node_metrics.copy()
    updated_node_metrics.update(new_metrics)
    return updated_node_metrics


def algo_retrieve(args):
    if args.get('louvain'):
        algo = 'Louvain'
    elif args.get('infomap'):
        algo = 'Infomap'
    elif args.get('lpa'):
        algo = 'LPA'
    return algo


def get_file_names(i,j, algo):
    parts_file = os.path.join('LFR_Graph_Data', 'Community_Data', algo, 'Runs',
                              'graph_0{0}_mu_0_{1}_runs.npy'.format(j,i))
    coassociation_file = os.path.join('LFR_Graph_Data', 'Community_Data', algo, 'Coassociation',
                                      'graph_0{0}_mu_0_{1}_coassociation.npy'.format(j,i))
    graph_file = 'LFR_Graph_Data/mu_0_{0}/graph_0{1}/graph_0{1}_mu_0_{0}.yml'.format(i,j)
    return parts_file, coassociation_file, graph_file


def append_to_dataframe(X, node_metrics, i, j):
    df = pd.DataFrame(node_metrics)
    new_indices = ['graph_{0}_{1}_node_{2}'.format(i, j, k) for k in range(200)]
    df.index = new_indices
    X = pd.concat([X, df])
    return X


def save_datasets(X_train, X_test, y_train, y_test, algo):
    final_folder = 'LFR_Graph_Data/' + algo + '_Data/'
    X_train.to_csv(final_folder + 'node_x_train.csv')
    X_test.to_csv(final_folder + 'node_x_test.csv')
    y_train.to_csv(final_folder + 'node_y_train.csv')
    y_test.to_csv(final_folder + 'node_y_test.csv')


def element_entropy(C):
    E = np.empty_like(C)
    rows, cols = E.shape
    for row in range(rows):
        for col in range(cols):
            p = C[row,col]
            if p > 0:
                E[row,col] = -p * math.log(p, 2)
            else:
                E[row,col] = 0
    entrop = np.mean(E, axis=1)
    return entrop


if __name__ == '__main__':
    args = docopt(__doc__)
    algo = algo_retrieve(args)
    graphs = [(i,j) for i in [1,2,3,4] for j in [1,2,3,4,5]]
    X = pd.DataFrame()
    node_entropies = []
    for i, j in tqdm(graphs):
        parts_file, coassociation_file, graph_file = get_file_names(i, j, algo)
        parts = np.load(parts_file)
        C = np.load(coassociation_file)
        with open(graph_file) as f:
            graph_info = yaml.load(f, Loader=yaml.Loader)
        G = graph_info['G']
        parts = convert_parts_format(parts)
        node_metrics, node_degrees = calc_node_metrics(G)
        node_metrics = new_node_metrics(node_metrics, parts, node_degrees)
        X = append_to_dataframe(X, node_metrics, i, j)
        entropies = element_entropy(C)
        node_entropies += list(entropies)
    node_entropies = np.array(node_entropies)
    y = np.where(node_entropies < 0.2, 0, 1)
    y = pd.DataFrame(y, index=X.index, columns=['Stability'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)
    save_datasets(X_train, X_test, y_train, y_test, algo)