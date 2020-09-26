"""
Run an experiment to compute the permutation importance for each node feature
on a specific graph, using kmeans to split nodes into stable and unstable
based on entropy.

The features_file is the path to a CSV file containing the values of node
features for each node.

The entropies_file is the path to a CSV file containing the entropy of each
node.

The results_loc is the folder in which to save the final results. This folder
should ideally be specific to this experiment and will be created if it doesn't
already exist.

Usage:
  importance_experiment.py <features_file> <entropies_file> <results_loc>

Options:
  -h --help            Show this help message

"""

import os
import pickle
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn import metrics

from docopt import docopt
from tqdm import trange


RUNS = 100


def create_and_save_data(feats_fil, entropies_fil, results_folder):
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    X = pd.read_csv(feats_fil, index_col=0)
    entrops = pd.read_csv(entropies_fil, index_col=0)
    entropy_values = np.array(entrops['Entropy'])
    entropy_values = entropy_values.reshape(-1, 1)
    kmeans_seed = random.randint(0, 10000)
    kmeans = KMeans(n_clusters=2, random_state=kmeans_seed).fit(entropy_values)
    cutoff = np.mean(kmeans.cluster_centers_)
    y = np.where(entropy_values.reshape(-1) < cutoff, 0, 1)
    stab_unstab = np.bincount(y)
    y = pd.DataFrame(y, index=X.index, columns=['Stability'])
    split_seed = random.randint(0, 10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=split_seed)
    X_train.to_csv(os.path.join(results_folder, 'X_train.csv'))
    y_train.to_csv(os.path.join(results_folder, 'y_train.csv'))
    X_test.to_csv(os.path.join(results_folder, 'X_test.csv'))
    y_test.to_csv(os.path.join(results_folder, 'y_test.csv'))
    return X_train, y_train, X_test, y_test, stab_unstab, cutoff


def train(X_train, X_test, y_train, y_test):
    feature_list = list(X_train.columns)
    data = np.array(X_train)
    labels = np.squeeze(np.array(y_train))

    accuracy_scores = []
    feature_importances = {}
    for f in feature_list:
        feature_importances[f] = []
    rows = []

    skf = StratifiedKFold(n_splits=5)
    rf = RandomForestClassifier()

    fold_count = 0

    for run in trange(1, RUNS+1):

        for i in range(5):

            fold_count += 1

            folds = next(skf.split(data, labels), None)

            X_train_fold = data[folds[0], :]
            X_val_fold = data[folds[1], :]
            y_train_fold = labels[folds[0]]
            y_val_fold = labels[folds[1]]

            model = rf.fit(X_train_fold, y_train_fold)
            predictions = rf.predict(X_val_fold)

            accuracy_scores.append(metrics.accuracy_score(y_val_fold, predictions))
            
            result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

            row = {"run": fold_count}
            for j, feature in enumerate(feature_list):
                row[feature] = result.importances_mean[j]
            rows.append(row)

    df_importances = pd.DataFrame(rows).set_index("run")
    perm_importances = df_importances.mean()
    return perm_importances, accuracy_scores


if __name__ == '__main__':
    args = docopt(__doc__)
    feats_fil = args.get('<features_file>')
    entropies_fil = args.get('<entropies_file>')
    results_folder = args.get('<results_loc>')
    X_train, y_train, X_test, y_test, stab_unstab, cutoff = create_and_save_data(feats_fil, 
                                                                                 entropies_fil, 
                                                                                 results_folder)
    perm_importances, accuracy_scores = train(X_train, X_test, y_train, y_test)
    results_dict = {'Stable Nodes': stab_unstab[0], 'Unstable Nodes': stab_unstab[1],
                    'Stability Cutoff': cutoff, 'Accuracy Scores': accuracy_scores,
                    'Feature Importances': dict(perm_importances)}
    results_path = os.path.join(results_folder, 'results')
    with open(results_path, 'wb') as fp:
        pickle.dump(results_dict, fp)
