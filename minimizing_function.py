from genetic_algorithm_bin import genetic_algorithm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.testing import ignore_warnings
from sklearn.metrics import accuracy_score
from functools import partial
import numpy as np

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
    clf = MLPClassifier(hidden_layer_sizes=(30,), random_state=SEED, max_iter=10) # in this case, max_iter is max_epochs
    clf.fit(X_train, Y_train) 
    train_score = clf.score(X_train, Y_train) 
    test_score = clf.score(X_test, Y_test) 
    return train_score, test_score

####################################################################


###################### Extreme Learning Machine ######################
# based on https://github.com/ivallesp/simplestELM/blob/master/ELM.py

# TODO, change tanh to sigmoid.. just substitute
def sigmoid(X):
    return 1. / (1. + np.exp(-X))

class ELMRegressor():
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

    def fit(self, X, Y):
        Y = self._format(Y)
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.random_weights = np.random.randn(X.shape[1], self.n_hidden_units)
        G = np.tanh(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(Y)

    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = np.tanh(X.dot(self.random_weights))
        return G.dot(self.w_elm)

    def _format(self, Y):
        n_classes = int(np.amax(Y) + 1)
        Y_ = np.zeros((Y.shape[0], n_classes)) - 1.0
        Y_[np.arange(Y.shape[0]), Y.astype(np.uint)] = 1.0
        return Y_

    def score(self, X, Y):

        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier.score
        # https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/base.py#L332
        Y_pred = np.argmax(self.predict(X), axis=1).astype(np.float)
        score = np.count_nonzero(Y - Y_pred == 0.0) / float(Y.shape[0])
        return score


def elm_fitness(X_train, Y_train):

    elm = ELMRegressor(n_hidden_units=100)
    elm.fit(X_train, Y_train)
    train_score = elm.score(X_train, Y_train)
    return train_score


def elm_train_eval(X_train, Y_train, X_test, Y_test, mask=None):

    if mask is not None:
        X_train = mask_input(X_train, mask)
        X_test = mask_input(X_test, mask)

    elm = ELMRegressor(n_hidden_units=100)
    elm.fit(X_train, Y_train)
    train_score = elm.score(X_train, Y_train)
    test_score = elm.score(X_test, Y_test) 
    return train_score, test_score


######################################################################



def mask_input(X, mask):

    return X * mask


def fitness_func_wrapper(fitness_func, X_train, Y_train, mask):

    rho = 0.9
    omega = 1.0 - rho
    X_train = mask_input(X_train, mask)
    train_score = fitness_func(X_train, Y_train)
    score = rho * (1.0 - train_score) + omega * mask.sum().astype(np.float) / mask.shape[1]
    return score


def main():
    
    np.random.seed(SEED)

    X, Y = data_loader('datasets/multiple_features.npy')
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=SEED, stratify=Y)
    X_train = data_normalizer(X_train)
    X_test = data_normalizer(X_test)

    dim = X_train.shape[1]
    # fitness_func = partial(fitness_func_wrapper, mlp_fitness, X_train, Y_train)
    fitness_func = partial(fitness_func_wrapper, elm_fitness, X_train, Y_train)

    best, fbest = genetic_algorithm(fitness_func=fitness_func, dim=dim)

    # train_score, test_score = mlp_train_eval(X_train, Y_train, X_test, Y_test, best)
    train_score, test_score = elm_train_eval(X_train, Y_train, X_test, Y_test, best)

    print(train_score, test_score, best.sum(), fbest)


if __name__ == '__main__':
    main()