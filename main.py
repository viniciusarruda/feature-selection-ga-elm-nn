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
import time

import matplotlib.pyplot as plt

from functools import partial
import numpy as np
import random

def data_loader(path):

    data = np.load(path)

    X = data[:, :-1]
    Y = data[:, -1]

    return X, Y


def data_normalizer(X):
    
    X /= np.amax(X, axis=0) 
    X = (X * 2.0) - 1.0
    return X


def compute_number_neurons(n_features, n_classes):
    # Masters (1993)
    return int(np.ceil(np.sqrt(n_features * n_classes)))


###################### Multi-layer Perceptron ######################

@ignore_warnings(category=ConvergenceWarning)
def mlp_fitness(n_features, n_classes, X_train, Y_train):

    n = compute_number_neurons(n_features, n_classes)
    clf = MLPClassifier(hidden_layer_sizes=(n,), random_state=SEED, max_iter=10) # in this case, max_iter is max_epochs
    clf.fit(X_train, Y_train) 
    train_score = clf.score(X_train, Y_train) 
    return train_score


@ignore_warnings(category=ConvergenceWarning)
def mlp_train_eval(n_features, n_classes, X_train, Y_train, X_test, Y_test, mask=None):

    if mask is not None:
        X_train = mask_input(X_train, mask)
        X_test = mask_input(X_test, mask)

    # should be concistent with net above
    n = compute_number_neurons(n_features, n_classes)
    clf = MLPClassifier(hidden_layer_sizes=(n,), random_state=SEED, max_iter=10) # in this case, max_iter is max_epochs
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

class ELMClassifier():
    def __init__(self, n_features, n_classes):

        self.n_features = n_features
        self.n_classes = n_classes
        self.n_hidden_units = compute_number_neurons(n_features, n_classes)

    def fit(self, X, Y):
        Y = self._format(Y)
        assert X.shape[1] == self.n_features and Y.shape[1] == self.n_classes
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.random_weights = np.random.randn(self.n_features + 1, self.n_hidden_units)
        G = sigmoid(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(Y)

    def predict(self, X):
        assert X.shape[1] == self.n_features
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = sigmoid(X.dot(self.random_weights))
        Y_pred = G.dot(self.w_elm)
        Y_pred = np.argmax(Y_pred, axis=1).astype(np.float)
        return Y_pred

    def _format(self, Y):
        Y_ = np.zeros((Y.shape[0], self.n_classes)) - 1.0
        Y_[np.arange(Y.shape[0]), Y.astype(np.uint)] = 1.0
        return Y_


# Keeping the ELM score consistent with the sklearn MLP score
# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.score
# https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/base.py#L332
def elm_fitness(n_features, n_classes, X_train, Y_train):

    elm = ELMClassifier(n_features, n_classes)
    elm.fit(X_train, Y_train)
    train_score = accuracy_score(Y_train, elm.predict(X_train))
    return train_score


def elm_train_eval(n_features, n_classes, X_train, Y_train, X_test, Y_test, mask=None):

    if mask is not None:
        X_train = mask_input(X_train, mask)
        X_test = mask_input(X_test, mask)

    elm = ELMClassifier(n_features, n_classes)
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
def plot_confusion_matrix(cm, classes, title, normalize=True, filepath=None):

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

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
    plt.close()


def save_metrics(filename, metrics):

    with open(filename, 'a') as f:
        f.write(','.join(['{}'.format(m) for m in metrics]) + '\n')


def mask_input(X, mask):

    return X * mask


def fitness_func_wrapper(fitness_func, n_features, n_classes, X_train, Y_train, mask):

    rho = 0.9
    omega = 1.0 - rho
    X_train = mask_input(X_train, mask)
    train_score = fitness_func(n_features, n_classes, X_train, Y_train)
    score = rho * (1.0 - train_score) + omega * mask.sum().astype(np.float) / mask.shape[1]
    return score


def baseline(net, dataset):

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    n_features, n_classes = {'breastEW':(30, 2), 'hepatitis':(19, 2), 'multiple_features':(649, 10)}[dataset]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)
    
    if net == 'NN':
        net_train_eval = mlp_train_eval
    elif net == 'ELM':
        net_train_eval = elm_train_eval
    else:
        ValueError('Algorithm {} not implemented.'.format(net))
        
    start_time = time.clock()
    train_accuracy, accuracy, recall, precision, fmeasure, confusion = net_train_eval(n_features, n_classes, X_train, Y_train, X_test, Y_test, mask=None)
    final_time = time.clock() - start_time
    
    # This result may match the 0.9 * (1.0 - train_score) + 0.1 * (n/nclasses) = fbest
    # If it does not match, it is because of the retraining of the net (net_train_eval) that used a different set of random weights
    # So, it is not a bug !
    print("##############################################################################")
    print("Let's see the training accuracy:", train_accuracy)
    print("Let's see the accuracy:", accuracy)
    print("Let's see the recall score:", recall)
    print("Let's see the precision score:", precision)
    print("Let's see the f1 score:", fmeasure)
    print("Let's see the time spent:", final_time)
    save_metrics('output/{}/baseline_{}.log'.format(dataset, net), [SEED, final_time, accuracy, recall, precision, fmeasure])
    plot_confusion_matrix(confusion, unique_labels(Y_test), 'Confusão', filepath='output/{}/confusion_matrix/baseline_{}_{}.jpg'.format(dataset, SEED, net))
    print("##############################################################################")


def main(net, dataset):

    X, Y = data_loader('datasets/{}.npy'.format(dataset))
    n_features, n_classes = {'breastEW':(30, 2), 'hepatitis':(19, 2), 'multiple_features':(649, 10)}[dataset]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train) # Yes, should be normalized after the train test split
    X_test = data_normalizer(X_test)
    
    if net == 'NN':
        net_fitness, net_train_eval = mlp_fitness, mlp_train_eval
    elif net == 'ELM':
        net_fitness, net_train_eval = elm_fitness, elm_train_eval
    else:
        ValueError('Algorithm {} not implemented.'.format(net))

    fitness_func = partial(fitness_func_wrapper, net_fitness, n_features, n_classes, X_train, Y_train)
    
    start_time = time.clock()
    best, fbest = genetic_algorithm(fitness_func=fitness_func, dim=n_features)
    train_accuracy, accuracy, recall, precision, fmeasure, confusion = net_train_eval(n_features, n_classes, X_train, Y_train, X_test, Y_test, best)
    final_time = time.clock() - start_time

    # This result may match the 0.9 * (1.0 - train_score) + 0.1 * (n/nclasses) = fbest
    # If it does not match, it is because of the retraining of the net (net_train_eval) that used a different set of random weights
    # So, it is not a bug !
    print("##############################################################################")
    print("Let's see the training accuracy:", train_accuracy)
    print("Let's see the best feature size: {} of {}".format(best.sum(), n_features))
    print("Let's see the fitness found by the best feature:", fbest)
    print("Let's see the accuracy:", accuracy)
    print("Let's see the recall score:", recall)
    print("Let's see the precision score:", precision)
    print("Let's see the f1 score:", fmeasure)
    print("Let's see the time spent:", final_time)
    save_metrics('output/{}/{}.log'.format(dataset, net), [SEED, final_time, best.sum(), fbest, accuracy, recall, precision, fmeasure])
    plot_confusion_matrix(confusion, unique_labels(Y_test), 'Confusão', filepath='output/{}/confusion_matrix/{}_{}.jpg'.format(dataset, SEED, net))
    print("##############################################################################")

if __name__ == '__main__':

    for SEED in range(20):
        for dataset in ['breastEW', 'hepatitis', 'multiple_features']:
            for net in ['NN', 'ELM']:

                np.random.seed(SEED)
                random.seed(SEED)
                
                print('### Running experiment with net {}, dataset {} and SEED {} ###'.format(net, dataset, SEED))
                main(net, dataset)

                print('### Running baseline experiment with net {}, dataset {} and SEED {} ###'.format(net, dataset, SEED))
                baseline(net, dataset)