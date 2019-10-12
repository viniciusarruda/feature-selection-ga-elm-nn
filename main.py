from genetic_algorithm_bin import genetic_algorithm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

import matplotlib.pyplot as plt

from functools import partial
import numpy as np
import random

SEED = 0


def data_loader(path):

    data = np.load(path)

    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


def data_normalizer(X):
    
    X /= np.amax(X, axis=0) 
    X = (X * 2.0) - 1.0
    return X


###################### Multi-layer Perceptron ######################

@ignore_warnings(category=ConvergenceWarning)
def mlp_fitness(X_train, Y_train):

    clf = MLPClassifier(hidden_layer_sizes=(30,), random_state=SEED, max_iter=10) # in this case, max_iter is max_epochs
    clf.fit(X_train, Y_train) 
    train_score = clf.score(X_train, Y_train) 
    return train_score


@ignore_warnings(category=ConvergenceWarning)
def mlp_train_eval(X_train, Y_train, X_test, Y_test, mask=None):

    if mask is not None:
        X_train = mask_input(X_train, mask)
        X_test = mask_input(X_test, mask)

    # should be concistent with net above
    clf = MLPClassifier(hidden_layer_sizes=(10,), random_state=SEED, max_iter=10) # in this case, max_iter is max_epochs
    clf.fit(X_train, Y_train)
    train_accuracy = clf.score(X_train, Y_train)

    _y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, _y_pred)
    recall = recall_score(Y_test, _y_pred, average='weighted')
    precision = precision_score(Y_test, _y_pred, average='weighted')
    fmeasure = f1_score(Y_test, _y_pred, average='weighted')
    confusion = confusion_matrix(Y_test, _y_pred)
    return train_accuracy, accuracy, recall, precision, fmeasure, confusion   

####################################################################


###################### Extreme Learning Machine ######################
# based on https://github.com/ivallesp/simplestELM/blob/master/ELM.py

def sigmoid(X):
    return 1. / (1. + np.exp(-X))

class ELMRegressor():
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

    def fit(self, X, Y):
        Y = self._format(Y)
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.random_weights = np.random.randn(X.shape[1], self.n_hidden_units)
        G = sigmoid(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(Y)

    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = sigmoid(X.dot(self.random_weights))
        Y_pred = G.dot(self.w_elm)
        Y_pred = np.argmax(Y_pred, axis=1).astype(np.float)
        return Y_pred

    def _format(self, Y):
        n_classes = int(np.amax(Y) + 1) # it will work only for data formated with classes [0, ..., n]
        Y_ = np.zeros((Y.shape[0], n_classes)) - 1.0
        Y_[np.arange(Y.shape[0]), Y.astype(np.uint)] = 1.0
        return Y_


# Keeping the ELM score consistent with the sklearn MLP score
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.score
# https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/base.py#L332
def elm_fitness(X_train, Y_train):

    elm = ELMRegressor(n_hidden_units=100)
    elm.fit(X_train, Y_train)
    train_score = accuracy_score(Y_train, elm.predict(X_train))
    return train_score


def elm_train_eval(X_train, Y_train, X_test, Y_test, mask=None):

    if mask is not None:
        X_train = mask_input(X_train, mask)
        X_test = mask_input(X_test, mask)

    elm = ELMRegressor(n_hidden_units=100)
    elm.fit(X_train, Y_train)
    train_accuracy = accuracy_score(Y_train, elm.predict(X_train))

    _y_pred = elm.predict(X_test)
    accuracy = accuracy_score(Y_test, _y_pred)
    recall = recall_score(Y_test, _y_pred, average='weighted')
    precision = precision_score(Y_test, _y_pred, average='weighted')
    fmeasure = f1_score(Y_test, _y_pred, average='weighted')
    confusion = confusion_matrix(Y_test, _y_pred)
    return train_accuracy, accuracy, recall, precision, fmeasure, confusion


######################################################################

# based ont the scikit learn documentation
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confusion_matrix(cm, classes, title, normalize=False, filepath=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    
    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if filepath == None:
        plt.show()
    else:
        plt.savefig(filepath)

def mask_input(X, mask):

    return X * mask


def fitness_func_wrapper(fitness_func, X_train, Y_train, mask):

    rho = 0.9
    omega = 1.0 - rho
    X_train = mask_input(X_train, mask)
    train_score = fitness_func(X_train, Y_train)
    score = rho * (1.0 - train_score) + omega * mask.sum().astype(np.float) / mask.shape[1]
    return score


def main(net, dataset):

    np.random.seed(SEED)
    random.seed(SEED)

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)

    dim = X_train.shape[1]
    
    if net == 'NN':
        net_fitness, net_train_eval = mlp_fitness, mlp_train_eval
    elif net == 'ELM':
        net_fitness, net_train_eval = elm_fitness, elm_train_eval
    else:
        ValueError('Algorithm {} not implemented.'.format(net))

    fitness_func = partial(fitness_func_wrapper, net_fitness, X_train, Y_train)
    
    best, fbest = genetic_algorithm(fitness_func=fitness_func, dim=dim, n_individuals=50)

    # I have added some extra and optinal outputs in the net_train_eval function
    # It's not elegant but...
    train_accuracy, accuracy, recall, precision, fmeasure, confusion = net_train_eval(X_train, Y_train, X_test, Y_test, best)

    # This result may match the 0.9 * (1.0 - train_score) + 0.1 * (n/nclasses) = fbest
    # If it does not match, it is because of the retraining of the net (net_train_eval) that used a different set of random weights
    # So, it is not a bug !
    print("##############################################################################")
    print("Let's see the training accuracy:", train_accuracy)
    print("Let's see the best feature size: {} of {}".format(best.sum(), best.shape[1]))
    print("Let's see the fitness found by the best feature:", fbest)
    print("Let's see the accuracy:", accuracy)
    print("Let's see the recall score:", recall)
    print("Let's see the precision score:", precision)
    print("Let's see the f1 score:", fmeasure)
    # I think if the normalized plot is better
    plot_confusion_matrix(confusion, unique_labels(Y_test), 'Confusão', filepath='output/{}/confusion_matrix/{}.jpg'.format(dataset, net))
    print("##############################################################################")

if __name__ == '__main__':

    for dataset in ['breastEW', 'hepatitis', 'multiple_features']:
        for net in ['NN', 'ELM']:
            print('### Running experiment with net {} and dataset {} ###'.format(net, dataset))
            main(net, dataset)