# community_finding

## Random Forest Feature Selection Experiments
You can find all the experiments that have been run so far in jupyter notebooks. There 3 folders - Node_Features, Pair_Features, and Community_Features, splitting the code and data into experiments run on these 3 different levels. Each of these has 2 subfolders - Notebooks, which contains the jupyter notebook, and Data, which contains the CSV files.
### Node Feature Level
- **Initial Experiment**: The node-level features are described and calculated in the node_feature_calculation notebook. The first 5 basic features were compared in the node_feature_ranking notebook.
- **Additional Features**: An experiment was run with a larger number of features in the node_feature_ranking_v2 notebook.
- **Training Methodology**: For the first 2 experiments mentioned above, the random forest was trained using a stratified k-fold cross-validation. In the node_feature_ranking_v2 notebook, a random forest was also trained and validated on pairs of graphs, though this caused problems due to extremely unbalanced classification classes.
- **Classification Classes**: For the node feature level, the nodes were categorised into "stable" and "unstable". This was determined by their entropy. In the notebook entropy_plot you can see histograms of entropy across all nodes. The notebook cutoff_accuracy shows how the mean accuracy scores of the random forest varied depending on the cutoff value of entropy used to split nodes into "stable" and "unstable".
### Pair Feature Level
- **Initial Experiment**: The node-pair-level features are described and calculated in the node_pair_feature_calculation notebook.
- **Classification Classes**: For the node-pair feature level, pairs are categorised into "same community" or "different community". Unlike with the node feature level, this allows us to conduct an experiment using ground truth. As well as the ground truth experiment, we can classify pairs according to their communities in runs of the community finding algorithm. For this purpose, they are classified as "same community" if they are sorted into the same community in more than (or = to) 50% of the runs, and "different community" if they are sorted into the same community in less than 50% of the runs.

## Data
- **Graph Generation**: The python file `LFR_Graphs/graph_gen.py` was used to generate the 20 LFR graphs.
- **Graph Folders**: In each of the mu folders with the LFR_Graphs directory, there are 5 graph subfolders, which each contain files with information on 1 graph. There is a raw COMMS and EDGES file for each graph, containing the edge information and the ground truth communities. There is also a yaml file which contains a NetworkX version of the graph, the communities, the parameters used to create the graph (including random seed) and the coordinates used to plot the graph in the png file, which displays an image of the graph.
- **Node Level Features**: The x files contain a list of nodes with all the calculated node features. These should be read into pandas dataframes using, for example, `x_train = pd.read_csv('node_x_train.csv', index_col=0)` to ensure that the first column is recognised as the node names rather than an additional feature. The y files contain a list of nodes with a single value determining whether they are classified as "stable" or "unstable", and should be read into pandas dataframes in the same way. Both x and y are split into test and train sets - the train sets have been used for training and validation in the experiments described above, and the test sets have not been touched at all. The split is stratified so that there is the same proportion of "stable" and "unstable" nodes in the train and test sets.
- **Node Pair Level Features**: The equivalent csv files for node pairs have not yet been uploaded to GitHub, although node pair features have been calculated as described in the relevant jupyter notebook.

## Examples
The notebook community_finding.ipynb shows an example of running the louvain community finding algorithm and visualizing a graph with or without communities labelled.
