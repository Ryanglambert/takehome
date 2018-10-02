from collections import OrderedDict
import itertools
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import twenty_newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, confusion_matrix
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from sklearn.model_selection import StratifiedKFold


def to_categorical(x, num_classes=20):
    return np.eye(num_classes)[x]


def make_cv_results(model, X, y, splits=5):
    splitter = StratifiedKFold(splits)
    results_true = []
    results_pred = []
    num_classes = np.unique(y).shape[0]
    for train_index, val_index in splitter.split(X, y):
        model.fit(X[train_index], y[train_index])
        true = y[val_index]
        pred = model.predict_proba(X[val_index])
        results_pred.append(pred)
        results_true.append(to_categorical(true, num_classes))

    results = {
        'true': results_true,
        'pred': results_pred
    }
    return results



def plot_confusion(y_true_list: list, y_pred_list: list, classes: list, lines_per_plot: int=5, title='', save_name=None, figsize=(10, 5), normalize_confusion_matrix=True, probabilities=True):
    """Plots precision and recall curves for results dictionary
    Parameters
    ----------
    y_true : list
        List of np.arrays that are the truth array for each kfold split
    y_pred : list
        List of np.arrays that are the pred_proba for each kfold split
    title : str
        The title for the plot
    save_name : str
        If you want to save this figure
    figsize : tuple (n, n)
        Size of figure
    Returns
    -------
    None
    """

    fig = plt.figure(1, figsize=figsize)
    # plot confusion matrix

    if probabilities:
        cm = confusion_matrix(np.argmax(np.concatenate(y_true_list), axis=1),
                            np.argmax(np.concatenate(y_pred_list), axis=1))
    else:
        cm = confusion_matrix(np.concatenate(y_true_list),
                              np.concatenate(y_pred_list))

    if normalize_confusion_matrix:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(cm, cmap=plt.cm.Blues)
    ax1.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, )

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    ax1.yaxis.set_label_coords(-0.1, 1.02)
    plt.xlabel('Predicted label')


def plot_prec_rec(y_true_list: list, y_pred_list: list, classes: list, lines_per_plot: int=5, title='', save_name=None, figsize=(10, 5), normalize_confusion_matrix=True, probabilities=True):
    """Plots precision and recall curves for results dictionary
    Parameters
    ----------
    y_true : list
        List of np.arrays that are the truth array for each kfold split
    y_pred : list
        List of np.arrays that are the pred_proba for each kfold split
    title : str
        The title for the plot
    save_name : str
        If you want to save this figure
    figsize : tuple (n, n)
        Size of figure
    Returns
    -------
    None
    """

    # plot curves
    fig = plt.figure(figsize=figsize)
    # plot each split
    ax = fig.add_subplot(1, 1, 1)
    # num_plots = len(classes) / lines_per_plot
    # figure_counter = 0

    # plot each class in each fold
    for i, (y_true, y_proba) in enumerate(zip(y_true_list, y_pred_list)):
        # reset colors so classes share the same color
        ax.set_prop_cycle(None)
        for i, cls in enumerate(classes):
            if not probabilities:
                precision, recall, _ = \
                    precision_recall_curve(y_true[:, i],
                                           y_proba[:, i])
            else:
                precision, recall, _ = \
                    precision_recall_curve(y_true[:, i],
                                           y_proba[:, i])
            ax.plot(recall, precision, label=cls)

    # show legends matching colors to splits
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    # plot mean precision and recalls in black
    for i, cls in enumerate(classes):
        if not probabilities:
            precision, recall, _ = \
                precision_recall_curve(y_true[:, i],
                                       y_proba[:, i])
        else:
            precision, recall, _ = \
                precision_recall_curve(y_true[:, i],
                                       y_proba[:, i])
        ax.plot(recall, precision, label=cls, color='black')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)