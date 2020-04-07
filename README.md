# community_finding

## Random Forest Feature Selection
You can find all the experiments that have been run so far in the jupyter notebooks folder. These are saved in 3 subfolders - node_features, node_pair_features, and community_features, splitting the code into experiments run on these 3 different levels. Outside the subfolders is a notebook community_finding.ipynb that shows an example of running the louvain community finding algorithm and visualizing a graph with or without communities labelled.
### Node Feature Level
- **Initial Experiment**: The node-level features are described and calculated in the node_feature_calculation notebook. The first 5 basic features were compared in the node_feature_ranking notebook.
- **Additional Features**: An experiment was run with a larger number of features in the node_feature_ranking_v2 notebook.
- **Training Methodology**: For the first 2 experiments mentioned above, the random forest was trained using a stratified k-fold cross-validation. In the node_feature_ranking_v2 notebook, a random forest was also trained and validated on pairs of graphs, though this caused problems due to extremely unbalanced classification classes.
- **Classification Classes**: For the node feature level, the nodes were categorised into "stable" and "unstable". This was determined by their entropy. In the notebook entropy_plot you can see histograms of entropy across all nodes. The notebook cutoff_accuracy shows how the mean accuracy scores of the random forest varied depending on the cutoff value of entropy used to split nodes into "stable" and "unstable".
### Node Pair Feature Level
- **Initial Experiment**: The node-pair-level features are described and calculated in the node_pair_feature_calculation notebook.
- **Classification Classes**: For the node-pair feature level, pairs are categorised into "same community" or "different community". Unlike with the node feature level, this allows us to conduct an experiment using ground truth. As well as the ground truth experiment, we can classify pairs according to their communities in runs of the community finding algorithm. For this purpose, they are classified as "same community" if they are sorted into the same community in more than (or = to) 50% of the runs, and "different community" if they are sorted into the same community in less than 50% of the runs.
