"""
Run an experiment to compute the permutation importance for each node feature
on a specific graph, using kmeans to split nodes into stable and unstable
based on entropy.

The features_file is the path to a CSV file containing the values of
features for each node or pair of nodes.

The entropies_file is the path to a CSV file containing the entropy of each
node, or the value in the coassociation matrix for the pair of nodes.

The results_loc is the folder in which to save the final results. This folder
should be specific to this experiment and will be created if it doesn't already 
exist.

NB: If mode is set to "pair", i.e. the experiment is for pair-node metrics, the undersample
rate is ignored and the number of each class will automatically match. This is
because there is an excess of data for the pair-node experiments.

Usage:
  importance_experiment.py <features_file> <entropies_file> <results_loc> (node | pair) [--undersample=<us>]

Options:
  -h --help            Show this help message
  --undersample=<us>   Rate at which to undersample majority class [default: None]

"""

import os
import pickle
import random
import numpy as np
import pandas as pd

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.cluster import KMeans
from sklearn import metrics

from docopt import docopt
from tqdm import trange


RUNS = 10


def node_dataset_gen(X, entropy_values, us):
    kmeans_seed = random.randint(0, 10000)
    kmeans = KMeans(n_clusters=2, random_state=kmeans_seed).fit(entropy_values.reshape(-1, 1))
    cutoff = np.mean(kmeans.cluster_centers_)
    y = np.where(entropy_values < cutoff, 0, 1)
    if us == 'strat':
        stab_unstab = np.bincount(y)
        num_unstab = stab_unstab[1]
        num_stab = int(num_unstab / 0.4) ## Strategic undersampling rate should be changed at some point
        lowest_entropy_indices = list(entropy_values.argsort()[:num_stab][::-1])
        highest_entropy_indices = list(entropy_values.argsort()[-num_unstab:][::-1])
        X_lowest = X.iloc[lowest_entropy_indices]
        X_highest = X.iloc[highest_entropy_indices]
        X = pd.concat([X_lowest, X_highest])
        y = [0 for _ in range(num_stab)] + [1 for _ in range(num_unstab)]
        y = pd.DataFrame(y, index=X.index, columns=['Stability'])
    elif us != None:
        y = pd.DataFrame(y, index=X.index, columns=['Stability'])
        under = RandomUnderSampler(sampling_strategy=us)
        undersampler = Pipeline(steps=[('us', under)])
        X, y = undersampler.fit_resample(X, y)
    else:
        y = pd.DataFrame(y, index=X.index, columns=['Stability'])
    split_seed = random.randint(0, 10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=split_seed)
    return X_train, X_test, y_train, y_test, cutoff


def pair_dataset_gen(X, entropy_values):
    lowest_entropy_indices = list(entropy_values.argsort()[:500][::-1])
    highest_entropy_indices = list(entropy_values.argsort()[-500:][::-1])
    X_lowest = X.iloc[lowest_entropy_indices]
    X_highest = X.iloc[highest_entropy_indices]
    X = pd.concat([X_lowest, X_highest])
    y = [0 for _ in range(500)] + [1 for _ in range(500)]
    y = pd.DataFrame(y, index=X.index, columns=['Same Community'])
    split_seed = random.randint(0, 10000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=split_seed)
    return X_train, X_test, y_train, y_test


def save_dataset(X_train, X_test, y_train, y_test):
    X_train.to_csv(os.path.join(results_folder, 'X_train.csv'))
    y_train.to_csv(os.path.join(results_folder, 'y_train.csv'))
    X_test.to_csv(os.path.join(results_folder, 'X_test.csv'))
    y_test.to_csv(os.path.join(results_folder, 'y_test.csv'))
    return


def create_and_save_data(feats_fil, entropies_fil, results_folder, us, mode):
    X = pd.read_csv(feats_fil, index_col=0)
    entrops = pd.read_csv(entropies_fil, index_col=0)
    entropy_values = np.array(entrops['Entropy'])
    entropy_values = entropy_values.reshape(-1)
    if mode == 'node':
        X_train, X_test, y_train, y_test, cutoff = node_dataset_gen(X, entropy_values, us)
    elif mode == 'pair':
        X_train, X_test, y_train, y_test = pair_dataset_gen(X, entropy_values)
    save_dataset(X_train, X_test, y_train, y_test)
    if mode == 'node':
        label_counts = np.bincount(y_train['Stability'])
        results_dict = {'Stable Nodes': label_counts[0], 'Unstable Nodes': label_counts[1],
                        'Stability Cutoff': cutoff, 'Undersampling Level': us}
    elif mode == 'pair':
        label_counts = np.bincount(y_train['Same Community'])
        results_dict = {'Different Communities': label_counts[0], 'Same Communities': label_counts[1],
                        'Undersampling Level': us}
    return X_train, y_train, X_test, y_test, results_dict


def train(X_train, X_test, y_train, y_test):
    feature_list = list(X_train.columns)
    data = np.array(X_train)
    labels = np.squeeze(np.array(y_train))

    accuracy_scores = []
    balanced_accuracy_scores = []
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
            balanced_accuracy_scores.append(metrics.balanced_accuracy_score(y_val_fold, predictions))
            
            result = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2)

            row = {"run": fold_count}
            for j, feature in enumerate(feature_list):
                row[feature] = result.importances_mean[j]
            rows.append(row)

    df_importances = pd.DataFrame(rows).set_index("run")
    perm_importances = df_importances.mean()
    return perm_importances, accuracy_scores, balanced_accuracy_scores


def set_mode_and_us(args):
    us = args.get('--undersample')
    if args.get('node'):
        mode = 'node'
        if us == 'None':
            us = None
        elif us == 'strat':
            us = 'strat'
        else:
            us = float(us)
    elif args.get('pair'):
        mode = 'pair'
        us = 1.0
    return us, mode


if __name__ == '__main__':
    args = docopt(__doc__)
    feats_fil = args.get('<features_file>')
    entropies_fil = args.get('<entropies_file>')
    results_folder = args.get('<results_loc>')
    us, mode = set_mode_and_us(args)
    if not os.path.exists(results_folder):
        os.mkdir(results_folder)
    X_train, y_train, X_test, y_test, results_dict = create_and_save_data(feats_fil, entropies_fil, 
                                                                          results_folder, us, mode)
    perm_importances, accuracy_scores, balanced_accuracy_scores = train(X_train, X_test, y_train, y_test)
    results_dict['Feature Importances'] = dict(perm_importances)
    results_dict['Accuracy Scores'] = accuracy_scores
    results_dict['Balanced Accuracy Scores'] = balanced_accuracy_scores
    results_path = os.path.join(results_folder, 'results')
    with open(results_path, 'wb') as fp:
        pickle.dump(results_dict, fp)
