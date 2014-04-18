__author__ = 'maru'

import numpy as np


def auc_ranked(neutral_target, ranking, pos_label=0):
    class_dist = np.bincount(neutral_target)
    neutral_count = 0.0
    auc_sum = 0.0
    for i in ranking:

        if neutral_target[i] == pos_label:
            neutral_count += 1
        else:
            auc_sum += neutral_count * (1.0 / class_dist[0]) * (1.0 / class_dist[1])
    return auc_sum


def auc_uncertainty(neutral_target, proba_pos, pos_label=0):

    """
    compute AUC values based on uncertainty ranking. This only works for neutral classificaiton of instances
    Note: only works for binary neutral classification
    @param neutral_target: target values of neutral classification(class)
    @param proba_pos: probabilities of positive class
    @param pos_label: index of positive labels
    """
    uncertainty = proba_pos.min(axis=1)
    ranking = np.argsort(uncertainty)[::-1] # descending order of uncertainty

    return auc_ranked(neutral_target, ranking, pos_label=pos_label)
