import os, os.path
import networkx as nx
import scipy.spatial as sp, scipy.cluster.hierarchy as hc
import seaborn as sns

def read_communities( comm_path ):
	communities = {}
	assigned_nodes = set()
	lines = open(comm_path,"r").readlines()
	for l in lines:
		parts = l.strip().split("\t")
		node_id = parts[0]
		assigned_nodes.add(node_id)
		for x in parts[1].split(" "):
			comm_id = int(x)
			if not comm_id in communities:
				communities[comm_id] = set()
			communities[comm_id].add( node_id )
	return communities

def find_all_networks( dir_root ):
	networks, communities = {}, {}
	rows = []
	for root, dirs, files in os.walk(dir_root):
		for fname in files:
			base, ext = os.path.splitext( fname )
			if ext != ".edges":
				continue
			in_path = os.path.join( root, fname )
			# read the edgelist
			g = nx.read_edgelist( in_path )
			networks[base] = g
			# read the corresponding communities
			comm_path = in_path.replace(".edges", ".comm")
			communities[base] = read_communities( comm_path )
	return networks, communities

def plot_correlation_clustermap( df_corr, figsize=(8,10) ):
	df_dist = 1 - df_corr
	linkage = hc.linkage(sp.distance.squareform(df_dist), method='average')
	cmap = sns.color_palette("RdBu_r", 7);
	cg = sns.clustermap(df_corr, row_linkage=linkage, col_linkage=linkage, vmin = -1, vmax = 1,
	                    cmap=cmap, center=0, square=True, figsize=figsize)
	return cg

def plot_feature_ranking( feature_ranking, figsize=(9,6), color="coral"):
	ax = feature_ranking[::-1].plot.barh( fontsize=13, figsize=(9,6), color="coral", zorder=3 );
	ax.set_xlabel("Mean Feature Importance", fontsize=13);
	ax.xaxis.grid()
	return ax

