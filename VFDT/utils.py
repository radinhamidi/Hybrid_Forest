from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
import numpy as np
from scipy import stats
from enum import Enum, auto


def calc_metrics(y_test, y_pred, row_name):
    accuracy = accuracy_score(y_test, y_pred)
    metrics = list(precision_recall_fscore_support(y_test, y_pred, average='weighted',
                                                   labels=np.unique(y_pred)))
    metrics = pd.DataFrame({'accuracy': accuracy, 'precision': metrics[0], 'recall': metrics[1],
                            'f1': metrics[2]}, index=[row_name])
    return metrics


def get_weaklearner_number(n_features, prob = 0.9):
    m = 1
    while True:
        p = (1 - (((n_features - 1) / n_features) ** (n_features * m)))
        if p > prob:
            break
        m += 1
    return m, p


def get_weaklearner_feature_indexes(n_trees, feature_vector):
    wl_trees_features = []
    for i in range(n_trees):
        wl_features = np.random.choice(np.arange(feature_vector.__len__()), int(np.sqrt(feature_vector.__len__())))
        wl_trees_features.append(wl_features)
    return wl_trees_features


def get_predictions(y_pred, y_pred_hybrid):
    all_pred = np.concatenate((y_pred_hybrid.T, y_pred.T), axis=1)
    return [stats.mode(i).mode[0] for i in all_pred]


def get_majority(all_pred):
    return [stats.mode(i).mode[0] for i in all_pred]


class NNmode(Enum):
    MLP = auto()
    LSTM = auto()