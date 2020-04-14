import math
import networkx as nx
import pandas as pd
import numpy as np

# --------------------------------------------------------------

def get_feature_generator( feature_id, g ):
	if feature_id == "relative_density":
		return RelativeDensity( g )
	elif feature_id == "relative_degree":
		return RelativeDegree( g )
	elif feature_id == "relative_between":
		return RelativeBetweenness( g )
	elif feature_id == "relative_closeness":
		return RelativeCloseness( g )
	elif feature_id == "relative_diameter":
		return RelativeDiameter( g )
	elif feature_id == "relative_pathlength":
		return RelativePathLength( g )
	elif feature_id == "cut":
		return EdgeCut( g )
	elif feature_id == "internal_external":
		return InternalExternal( g )
	elif feature_id == "conductance":
		return Conductance( g )
	raise Exception("Unknown feature '%s'" % feature_id )
	
# --------------------------------------------------------------

class RelativeDensity:
	""" Measure the density within a community, relative to density in the entire network """
	def __init__( self, g ):
		self.g = g 
		self.overall_density = nx.density( g )
		
	def evaluate( self, comm ):
		comm_subgraph = nx.subgraph( self.g, comm )
		subgraph_density = nx.density(comm_subgraph)
		return subgraph_density/self.overall_density


class RelativeDiameter:
	""" Measure the diameter within a community, relative to diameter for the entire network """
	def __init__( self, g ):
		self.g = g 
		self.overall_diameter = nx.diameter(g)
	
	def evaluate( self, comm ):
		comm_subgraph = nx.subgraph( self.g, comm )
		if not nx.is_connected(comm_subgraph):
			return 0
		subgraph_diameter = nx.diameter( comm_subgraph )
		return subgraph_diameter/self.overall_diameter


class RelativePathLength:
	""" Measure the average shortest path length within a community, relative to the entire network """
	def __init__( self, g ):
		self.g = g 
		self.average_shortest_path_length = nx.average_shortest_path_length( g )

	def evaluate( self, comm ):
		comm_subgraph = nx.subgraph( self. g, comm )
		if not nx.is_connected(comm_subgraph):
			return 0
		comm_average_shortest_path_length = nx.average_shortest_path_length( comm_subgraph )
		return comm_average_shortest_path_length/self.average_shortest_path_length


class RelativeDegree:
	""" Measure average degree centrality  within the community, relative to the entire network"""
	def __init__( self, g ):
		self.g = g 
		self.overall_cen_degrees = dict( nx.degree_centrality(g) )
		self.overall_mean_deg = np.array( list(self.overall_cen_degrees.values()) ).mean()

	def evaluate( self, comm ):
		comm_subgraph = nx.subgraph( self.g, comm )
		comm_degrees = dict(nx.degree_centrality(comm_subgraph))
		comm_mean_deg = np.array( list(comm_degrees.values()) ).mean()
		return comm_mean_deg/self.overall_mean_deg


class RelativeBetweenness:
	""" Measure average betweenness centrality  within the community, relative to the entire network"""
	def __init__( self, g ):
		self.g = g 
		self.overall_cen_between = dict(nx.betweenness_centrality(g, normalized=True))
		self.overall_mean_between = np.array( list(self.overall_cen_between.values()) ).mean()

	def evaluate( self, comm ):
		comm_subgraph = nx.subgraph( self.g, comm )
		comm_between = dict(nx.betweenness_centrality(comm_subgraph, normalized=True))
		relative_between = np.array( list(comm_between.values()) ).mean()
		return relative_between/self.overall_mean_between


class RelativeCloseness:
	""" Measure average closeness centrality  within the community, relative to the entire network"""
	def __init__( self, g ):
		self.g = g 
		self.overall_cen_closeness = dict(nx.closeness_centrality(g))
		self.overall_mean_closeness = np.array( list(self.overall_cen_closeness.values()) ).mean()

	def evaluate( self, comm ):
		comm_subgraph = nx.subgraph( self.g, comm )
		comm_closeness = dict(nx.closeness_centrality(comm_subgraph))
		comm_mean_deg = np.array( list(comm_closeness.values()) ).mean()
		return comm_mean_deg/self.overall_mean_closeness


class EdgeCut:
	def __init__( self, g ):
		self.g = g 

	def evaluate( self, comm ):
		""" Basic edge cut, where set A is the community and set B is the rest of the network. 
		In this case we normalize with respect to the number of possible pairs."""
		setb = set(self.g.nodes()) - set(comm)
		cut_ab = 0
		for node1 in comm:
			for node2 in setb:
				if self.g.has_edge(node1,node2):
					cut_ab += 1
		return cut_ab/( len(comm) * len(setb) )


class InternalExternal:
	""" Measure based on the community finding fitness function proposed
	by Lancichinetti et al (2009) """
	def __init__( self, g, alpha = 1.0 ):
		self.g = g 
		self.alpha = alpha

	def evaluate( self, comm ):
		kin, kout = 0.0, 0.0
		for node1 in comm:
			for node2 in self.g.neighbors(node1):
				if node1 == node2:
					continue
				if node2 in comm:
					kin += 1
				else:
					kout += 1
		denom = math.pow( kin+kout, self.alpha )
		if denom == 0:
			return 0.0
		return kin/denom


class Conductance:
	""" The conductance of a graph measures how well-knit the graph is. It
	is another time of normalized edge cut."""
	def __init__( self, g ):
		self.g = g 
		self.overall_degree = dict( g.degree() )

	def evaluate( self, comm ):
		# calculate the numerator as the edge cut
		cut_ab = 0
		for node1 in comm:
			for node2 in self.g.neighbors(node1):
				if not node2 in comm:
					cut_ab += 1
		# calculate the denominator as minimum of
		# sum of degrees in either set
		vol_a, vol_b = 0, 0
		for node in self.overall_degree:
			if node in comm:
				vol_a += self.overall_degree[node]
			else:
				vol_b += self.overall_degree[node]
		denom = min(vol_a, vol_b )
		if denom == 0:
			return 0.0
		return cut_ab/denom

