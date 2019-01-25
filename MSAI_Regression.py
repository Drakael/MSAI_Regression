# -*- coding: utf-8 -*-
"""
=================================================
MSAI Solver Class with linear and logistic regression
=================================================
On 2018 march
Author: Drakael Aradan
"""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
#from sklearn.datasets import make_classification
from functools import reduce


#fonction utile pour le débugging
def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)


#classe abstraite pour les modèles ( = modèles prédictifs)
class MSAI_Model(ABC):
    """Base class for Models"""
    def __init__(self, learning_rate, max_iterations,
                 starting_thetas=None, starting_bias=None,
                 range_x=10, n_samples=1000, regularization='l1'):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.starting_bias = starting_bias
        self.predicted_thetas = None
        self.predicted_bias = None
        self.range_x = range_x
        self.n_samples = n_samples
        self.regularization = regularization
        self.regul_ratio = 1e-3
        self.progression = None
        self.minima = None
        self.maxima = None
        self.mean = None
        self.median = None
        self.std = None
        self.ptp = None
        self.scale_mode = 'ptp'
        self.cout_moyen = 1

    @abstractmethod
    def fit(self, X, Y, max_time=0, tic_time=0):
        """Fit model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Predict using model
        """
        pass

    @abstractmethod
    def rescale(self):
        """Rescaling using model
        """
        pass

    @abstractmethod
    def randomize_model(self, theta, bias, X, range_x, random_ratio=0.0, offsets=None):
        """Calculate target for initial training set and weights
        """
        pass

    @abstractmethod
    def plot_1_dimension(self, X, Y, target_thetas=None):
        """Plot visual for 1 dimentionnal problem
        """
        pass

    def init_attribs_from_X(self, X):
        """Stores means, stds and ranges for X
        """
        self.n_samples = X.shape[0]
        self.range_x = np.max(np.abs(X))
        self.mean = X.mean(axis=0)
        self.median = np.median(X, axis=0)
        self.std = X.std(axis=0)
        self.ptp = X.ptp(axis=0)
        if 0 in self.ptp:
            print('zero ranged value!!!!')
            idx = np.where(self.ptp == 0)
            print('colonnes:', idx)
        self.minima = X.min(axis=0)
        self.maxima = X.max(axis=0)
        return self

    def get_mean(self):
        return self.mean

    def get_median(self):
        return self.median

    def get_std(self):
        return self.std

    def get_ptp(self):
        return self.ptp

    #fonction de normalisation par minimum et maximum
    def scale_(self, X):
        """Scale X values with min and max
        """
        self.init_attribs_from_X(X)
        for i in range(X.shape[1]):
            min_ = np.min(X[:, i])
            max_ = np.max(X[:, i])
            X[:, i] -= min_
            X[:, i] /= max_-min_
        return X

    #fonction de normalisation par moyenne (=mean) et plage de valeurs (=range)
    def scale(self, X, on='ptp'):
        """Scale X values with means and ranges
        """
        self.init_attribs_from_X(X)
        self.scale_mode = on
        if self.scale_mode == 'ptp':
            for i, ptp in enumerate(self.ptp):
                X[:, i] = 2*(X[:, i] - self.mean[i]) / ptp if ptp != 0 else 0
        else:
            for i, std in enumerate(self.std):
                X[:, i] = 2*(X[:, i] - self.mean[i]) / std if std != 0 else 0
        return X

    def linear_regression(self, theta, x):
        """linear regression method
        """
        if isinstance(x, int):
            if theta.shape[0] == len(x)+1:
                x = np.concatenate([[1, ], x])
        elif type(x).__module__ == np.__name__:
            if len(x.shape) == 1:
                x = x.reshape(1, x.shape[0])
            elif len(x.shape) == 0:
                x = np.array(x).reshape(1, 1)
            if theta.shape[0] == x.shape[1]+1:
                x = np.column_stack((np.ones(len(x)), x))
        else:
            print('different type!!!!', type(x))
        return np.matmul(x, theta)

    def get_cost_derivative_(self, model, theta, X, Y):
        """cost derivative calculation from Achille
        """
        if theta.shape[0] == X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples), X))
        diff_transpose = (model(theta, X)-Y).T
        return np.array([np.sum(np.matmul(diff_transpose, X[:, i]))/(X.shape[0])
                         for i, t in enumerate(theta)]).reshape(len(theta), 1)

    def plot_progression(self):
        """plot learning progression
        """
        if self.progression is not None:
            plt.figure()
            plt.plot(self.progression, label='progression')
            plt.legend()
            plt.show()


#Classe de régression linéaire
class LinearRegression(MSAI_Model):
    """Linear Regression Class
    """
    def __init__(self, learning_rate=3*10**-1, max_iterations=4000,
                 starting_thetas=None, range_x=1, n_samples=0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.range_x = range_x
        self.n_samples = n_samples
        MSAI_Model.__init__(self, learning_rate=self.learning_rate,
                            max_iterations=self.max_iterations,
                            starting_thetas=self.starting_thetas,
                            range_x=self.range_x,
                            n_samples=self.n_samples)

    def fit(self, X, Y, max_time=0, tic_time=0):
        """Linear Regression Fit
        """
        X = self.scale(X, 'ptp')
        self.predicted_thetas = self.gradient_descent(self.linear_regression,
                                                      X, Y, self.max_iterations,
                                                      self.learning_rate,
                                                      starting_thetas=self.starting_thetas,
                                                      max_time=max_time, tic_time=tic_time)
        self.rescale()
        return self

    def predict(self, X):
        """Linear Regression Prediction
        """
        return self.linear_regression(self.predicted_thetas, X)

    #fonction de remise à l'échelle des poids prédis selon la normalisation initiale
    def rescale(self):
        """Rescale weights to original scale
        """
        array = []
        theta_zero = self.predicted_thetas[0].copy()
        for col, mean, std, ptp in zip(self.predicted_thetas[1:], self.mean,
                                       self.std, self.ptp):
            if self.scale_mode == 'ptp':
                theta_i = float((2 * (col)) / ptp)
            else:
                theta_i = float((2 * (col)) / std)
            array.append(theta_i)
            theta_zero -= theta_i * mean
        self.predicted_thetas = np.array([float(theta_zero), ]+array).reshape(len(self.predicted_thetas), 1)
        return self

    def regression_cost(self, model, theta, X, Y):
        """Linear Regression cost calculation
        """
        diff = (model(theta, X)-Y)
        return float(1/(2 * len(X)) * np.matmul(diff.T, diff))
        #return float(np.sum(np.abs(model(theta,X)-Y))/ (self.n_samples * X.shape[1] * self.range_x))
        #return float(np.sum(np.abs(model(theta,X)-Y))/ (self.n_samples * X.shape[1])) 
    
    def get_cost_derivative(self, model, theta, X, Y):
        """Linear/Logistic cost derivative calculation
        """
        if theta.shape[0] == X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples), X))
        diff = model(theta, X)-Y
        if self.regularization == 'l1':
            diff += self.regul_ratio * np.sum(np.matmul(theta, theta.T))
        elif self.regularization == 'l2':
            diff += self.regul_ratio * np.sum(np.abs(theta))
        print('theta', theta.shape, theta)
        return np.matmul(X.T, diff) / self.n_samples

    def gradient_descent(self, initial_model, X, Y, max_iterations, alpha,
                         starting_thetas=None, max_time=0, tic_time=None):
        """performs gradient descent
        """
        self.n_samples = len(X)
        if starting_thetas is None:
            self.starting_thetas = np.random.random((X.shape[1]+1, 1))
        self.predicted_thetas = self.starting_thetas
        self.progression = []
        cnt = max_iterations
        cout = 1
        if tic_time is None:
            tic_time = datetime.now()
        while cnt > 0 and ((max_time == 0) or (datetime.now()-tic_time).seconds < max_time):  # and (len(self.progression)<=100 or (len(self.progression)>100 and np.abs(self.cout_moyen)>0.00000001)):#np.abs(cout) > 0.00000001 and
            thetas_cost = self.get_cost_derivative(initial_model,
                                                   self.predicted_thetas, X, Y)
            thetas_cost *= alpha
            self.predicted_thetas = self.predicted_thetas - thetas_cost
            cout = self.regression_cost(initial_model, self.predicted_thetas, X, Y)
            self.progression.append(cout)
            self.cout_moyen = np.mean(self.progression[-100:])
            cnt -= 1
        self.plot_progression()
        #self.plot_1_dimension(X, Y, target_thetas=starting_thetas)
        return self.predicted_thetas

    def randomize_model(self, theta, bias, X, range_x, random_ratio=0.0,
                        offsets=None):
        """Linear Regression randomize function
        """
        self.n_samples = X.shape[0]
        if theta.shape[0] == X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples), X))
        produit = np.matmul(X, theta)
        #option de random pour ajouter du bruit statistique
        if random_ratio != 0.0:
            produit += (np.random.random(produit.shape)-0.5)*range_x*random_ratio
        return produit

    def plot_1_dimension(self, X, Y, target_thetas=None, target_bias=None):
        """Linear Regression 1 dimensionnal plot
        """
        if(len(self.predicted_thetas) == 2):
            plt.figure()
            plt.plot(X, Y  , 'o', label='original data')
            plt.plot(X, self.predicted_thetas[0] + self.predicted_thetas[1]*X,
                     'r', label='fitted line')
            plt.legend()
            plt.show()


class LogisticRegression(MSAI_Model):
    """Logistic Regression Class
    """
    def __init__(self, learning_rate=0.5, max_iterations=4000,
                 starting_thetas=None, range_x=1, n_samples=0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.range_x = range_x
        self.n_samples = n_samples
        MSAI_Model.__init__(self, self.learning_rate, self.max_iterations,
                            self.starting_thetas, self.range_x,
                            self.n_samples)

    def fit(self, X, Y, max_time=0, tic_time=0):
        """Logistic Regression Fit
        """
        X = self.scale(X, 'std')
        self.predicted_thetas = self.gradient_descent(self.sigmoid, X, Y,
                                                      self.max_iterations,
                                                      self.learning_rate,
                                                      starting_thetas=self.starting_thetas,
                                                      max_time=max_time,
                                                      tic_time=tic_time)
        #self.predicted_thetas/= np.absolute(self.predicted_thetas[:,0]).max()
        self.rescale()
        return self

    def predict(self, X):
        """Logistic Regression Prediction
        """
        array = self.sigmoid(self.predicted_thetas, X)
        array = list(map(lambda x: 1 if x >= 0.5 else 0, array[:, 0]))
        return array

    #fonction de remise à l'échelle des poids prédis selon la normalisation initiale
    def rescale(self):
        """Rescale weights to original scale
        """
        array = []
        p('self.predicted_thetas before',self.predicted_thetas)
        theta_zero = self.predicted_thetas[0].copy()
        for col, mean, std, ptp in zip(self.predicted_thetas[1:], self.mean,
                                       self.std, self.ptp):
            if self.scale_mode == 'ptp':
                theta_i = float((2 * (col)) / ptp)
            else:
                theta_i = float((2 * (col)) / std)
            array.append(theta_i)
            theta_zero -= theta_i * mean
#         for col, mean, median, std, ptp in zip(self.predicted_thetas[1:], self.mean, self.median, self.std, self.ptp):
#             if self.scale_mode == 'ptp':
#                 sign = -1 if col < 0.0 else 1
#                 degre = np.floor(np.log10(ptp))
#                 p('degre',degre)
#                 theta_i = col
#                 theta_i = sign*10**(degre-1)
#                 theta_zero -= (theta_i * mean)
#             else:
#                 sign = -1 if col < 0.0 else 1
#                 degre = 1+np.floor(np.log10(ptp))
#                 #degre = -sign*np.floor(np.log10(ptp))
#                 degre2 =  10**(degre)
#                 degre3 = sign*10**np.ceil(np.log10(np.abs(col)))
#                 p('col',col)
#                 p('ptp',ptp)
#                 p('sign',sign)
#                 p('mean',mean)
#                 p('degre',degre)
#                 p('degre2',degre2)
#                 p('degre3',degre3)
# #                if degre < 1:
# #                    degre = 1
#                 theta_i = degre2#((2 * mean)/(ptp))#float((2 * (col)) / std)
#                 theta_zero -= (mean * theta_i) #+ (1.0 * ptp) #* theta_i#theta_i * mean
#             array.append(theta_i)
        self.predicted_thetas = np.array([float(theta_zero), ] + array).reshape(len(self.predicted_thetas), 1)
        
        p('self.predicted_thetas after', self.predicted_thetas)
        return self

    def sigmoid(self, theta, x):
        """Logistic Regression Sigmoid function
        """
        sigmoid = 1/(1+np.exp(self.linear_regression(theta, x)*-1))
        if sigmoid.shape == (1, 1):
            sigmoid = sigmoid[0][0]
        return np.round(sigmoid, 2)

    def regression_cost_(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid(theta, x), 0.00000001, 0.99999999)
            somme += (y * np.log(sig)) + ((1-y) * np.log(1 - sig))
        somme /= -self.n_samples
        return float(somme)

    def regression_cost__(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid(theta, x), 0.00000001, 0.99999999)
            somme += (y * ((1/sig)-1)) + ((1-y) * ((1/(1-sig))-1))
        somme /= self.n_samples
        return float(somme)

    def regression_cost(self, model, theta, X, Y):
        """Logistic Regression Cost calculation (regular)
        """
        #cout = float(np.sum(np.absolute(model(theta,X)-Y))/ self.n_samples) 
        cout = float(np.absolute((model(theta, X)-Y)**2).mean())
        #cout = float(np.mean((model(theta,X)-Y)**2))
        return cout

    def gradient_descent(self, initial_model, X, Y, max_iterations, alpha,
                         starting_thetas=None, max_time=0, tic_time=None):
        """performs gradient descent
        """
        self.n_samples = len(X)
        if starting_thetas is None:
            self.starting_thetas = np.random.random((X.shape[1]+1,1))
        self.predicted_thetas = self.starting_thetas
        self.progression = []
        cnt = max_iterations
        cout = 1
        if tic_time is None:
            tic_time = datetime.now()
        while cnt > 0 and ((max_time == 0) or (datetime.now()-tic_time).seconds < max_time):  # and (len(self.progression)<=100 or (len(self.progression)>100 and np.abs(self.cout_moyen)>0.00000001)):#np.abs(cout) > 0.00000001 and
            thetas_cost = self.get_cost_derivative(initial_model,
                                                   self.predicted_thetas, X, Y)
            thetas_cost *= alpha
            self.predicted_thetas = self.predicted_thetas - thetas_cost
            cout = self.regression_cost(initial_model, self.predicted_thetas, X, Y)
            self.progression.append(cout)
            self.cout_moyen = np.mean(self.progression[-100:])
            cnt -= 1
        self.plot_progression()
        #self.plot_1_dimension(X, Y, target_thetas=starting_thetas)
        return self.predicted_thetas

    def get_cost_derivative(self, model, theta, X, Y):
        """Linear/Logistic cost derivative calculation
        """
        if theta.shape[0] == X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples), X))
        diff = model(theta, X)-Y
        if self.regularization == 'l1':
            diff += self.regul_ratio * np.sum(np.matmul(theta, theta.T))
        elif self.regularization == 'l2':
            diff += self.regul_ratio * np.sum(np.abs(theta))
        return np.matmul(X.T, diff) / self.n_samples

    def randomize_model(self, theta, bias, X, range_x, random_ratio=0.0,
                        offsets=None):
        """Logistic Regression Randomize function
            TODO: test sur random_ratio qui doit être entre 0.0 et 1.0
        """
        self.n_samples = X.shape[0]
        if theta.shape[0] == X.shape[1]+1:
            X = np.column_stack((np.ones(self.n_samples), X))
        produit = []
        for x in X:
            sig = self.sigmoid(theta, x.reshape(1, len(x)))
            val = 1 if sig > 0.5 else 0
            #option de random pour ajouter du bruit statistique
            if random_ratio != 0.0:
                val = val if np.random.random() < random_ratio else 1 - val
            produit.append(val)
        return np.array(produit, order='F').reshape(self.n_samples, 1)

    def plot_1_dimension(self, X, Y, target_thetas=None, target_bias=None):
        """Logistic Regression 1 dimensionnal plot
        """
        if(len(self.predicted_thetas) == 2):
            plt.figure()
            plt.plot(X, Y, 'o', label='original data')
            x = np.linspace(np.min(X), np.max(X), 100)
            y = []
            for var in x:
                sig = self.sigmoid(self.predicted_thetas, var)
                y.append(sig)
            plt.plot(x, y,'r')
            plt.legend()
            plt.show()


class SelectiveRegression(MSAI_Model):
    """Selective Regression Class
    """

    def __init__(self, learning_rate=0.5, max_iterations=4000,
                 starting_thetas=None, starting_bias=None, range_x=1,
                 n_samples=0):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.starting_thetas = starting_thetas
        self.starting_bias = starting_bias
        self.range_x = range_x
        self.n_samples = n_samples
        MSAI_Model.__init__(self, self.learning_rate, self.max_iterations,
                            self.starting_thetas, self.starting_bias,
                            self.range_x, self.n_samples)

    def fit(self, X, Y, max_time=0, tic_time=0):
        """Selective Regression Fit
        """
        #X = self.scale(X, 'ptp')
        self.gradient_descent(self.sigmoid_prime, X, Y, self.max_iterations,
                              self.learning_rate,
                              starting_thetas=self.starting_thetas,
                              starting_bias=self.starting_bias,
                              max_time=max_time, tic_time=tic_time)
        # self.predicted_thetas/= np.absolute(self.predicted_thetas[:,0]).max()
        #self.rescale()
        return self

    def predict(self, X):
        """Selective Regression Prediction
        """
        array = self.selective_regression(self.predicted_thetas,
                                          self.predicted_bias, X)
        return array

    # fonction de remise à l'échelle des poids prédis selon la normalisation initiale
    def rescale(self):
        """Rescale weights to original scale
        """
        array = []
        p('self.predicted_thetas before', self.predicted_thetas)
        theta_zero = self.predicted_thetas[0].copy()
        for col, mean, std, ptp in zip(self.predicted_thetas[1:], self.mean,
                                       self.std, self.ptp):
            if self.scale_mode == 'ptp':
                theta_i = float((2 * (col)) / ptp)
            else:
                theta_i = float((2 * (col)) / std)
            array.append(theta_i)
            theta_zero -= theta_i * mean
        self.predicted_thetas = np.array([float(theta_zero), ] + array).reshape(len(self.predicted_thetas), 1)
        p('self.predicted_thetas after', self.predicted_thetas)
        return self

    def sigmoid(self, theta, x):
        """Selective Regression Sigmoid function
        """
        sigmoid = 1 / (1 + np.exp(self.linear_regression(theta, x) * -1))
        if sigmoid.shape == (1, 1):
            sigmoid = sigmoid[0][0]
        return np.round(sigmoid, 2)

    def sigmoid_prime(self, theta, bias, x):
        """Selective Regression Sigmoid Prime function
        """
        #sigmoid = 1 / (1 + np.exp(self.linear_regression(theta, x) * -1))
        #sigmoid_prime = ( ( T **2 * np.exp( (T * self.linear_regression(theta, x)) + (2 * a * T) ) ) - ( T ** 2 * np.exp( (2 * T * self.linear_regression(theta, x) ) + (a * T)) ) ) / ( ( np.exp( T * self.linear_regression(theta, x)) + exp( T * a) ) ** 3 )
        sigmoid_prime = theta * np.exp(x - bias) / (1 + np.exp(x - bias))**2
        if sigmoid_prime.shape == (1, 1):
            sigmoid_prime = sigmoid_prime[0][0]
        return sigmoid_prime

    def sigmoid_prime_prime(self, theta, bias, x):
        """Selective Regression Sigmoid Prime 2nd function
        """
        #sigmoid = 1 / (1 + np.exp(self.linear_regression(theta, x) * -1))
        #sigmoid_prime = ( ( T **2 * np.exp( (T * self.linear_regression(theta, x)) + (2 * a * T) ) ) - ( T ** 2 * np.exp( (2 * T * self.linear_regression(theta, x) ) + (a * T)) ) ) / ( ( np.exp( T * self.linear_regression(theta, x)) + exp( T * a) ) ** 3 )
        sigmoid_prime_prime = theta * np.exp(x - bias) * (np.exp(x - bias) - 1) / (1 + np.exp(x - bias))**3
        if sigmoid_prime_prime.shape == (1, 1):
            sigmoid_prime_prime = sigmoid_prime_prime[0][0]
        return sigmoid_prime_prime

    def selective_regression(self, model, theta, bias, x):
        selective_regression = model(theta, bias, x)
        #p(model.__func__.__name__,selective_regression)
        if len(selective_regression) > 1:
            reduce_ = reduce((lambda a, b: a*b), selective_regression)
            #print('reduce',reduce_)
            return reduce_
        return selective_regression[0][0]

    def sigmoid_prime_cut(self, theta, bias, x):
        sig = self.sigmoid_prime(theta, bias, x)
        clip = np.clip(sig, 0, 1)
        return clip
        #return np.clip(self.sigmoid_prime(theta, bias, x), 0, 1)

    def sigmoid_prime_max(self, theta, bias):
        """Selective Regression Sigmoid function
        """
        #sigmoid = 1 / (1 + np.exp(self.linear_regression(theta, x) * -1))
        #sigmoid_prime_max = ( T ** 2 * (np.exp(x) - 1) * np.exp(x) ) / (1 + np.exp(x))**3
        #if sigmoid_prime_max.shape == (1, 1):
        #    sigmoid_prime_max = sigmoid_prime_max[0][0]
        #return np.round(sigmoid_prime_max, 2)
        return self.sigmoid_prime(theta, bias, bias)

    def sigmoid_prime_adjust(self, y, T, f):
        if(f == 3):
            y /= -T/4
            y *= 2
            y += 1
        else:
            y /= -T/4
        return y

    def regression_cost_(self, model, theta, X, Y):
        """Selective Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid_prime_cut(theta, x), 0.00000001, 0.99999999)
            somme += (y * np.log(sig)) + ((1 - y) * np.log(1 - sig))
        somme /= -self.n_samples
        return float(somme)

    def regression_cost__(self, model, theta, X, Y):
        """Selective Regression Cost calculation (heavy)
        """
        somme = 0
        for x, y in zip(X, Y):
            sig = np.clip(self.sigmoid_prime_cut(theta, x), 0.00000001, 0.99999999)
            somme += (y * ((1 / sig) - 1)) + ((1 - y) * ((1 / (1 - sig)) - 1))
        somme /= self.n_samples
        return float(somme)

    def regression_cost(self, model, theta, bias, X, Y):
        """Selective Regression Cost calculation (regular)
        """
        # cout = float(np.sum(np.absolute(model(theta,X)-Y))/ self.n_samples)
        #cout = float(np.absolute((model(theta, bias, X) - Y) ** 2).mean())
        sig = model(theta, bias, X)
        diff = (Y*(X)) - (sig*(X))
        diff_mean = np.sum(diff)/self.n_samples
        cout = diff_mean
        # cout = float(np.mean((model(theta,X)-Y)**2))
        return cout

    def get_cost_derivative(self, model, theta, bias, X, Y):
        """SelectiveRegression cost derivative calculation
        """
        sig = model(theta, bias, X)
        diff = (Y*(X)) - (sig*(X))
        diff_sum = np.sum(diff)/self.ptp*self.n_samples
        diff_abs_sum = np.abs(diff_sum)
        #diff_reshaped = diff.reshape(1,X.shape[0])
        theta_new = []
        bias_new = []
        for i, t in enumerate(theta):
            #deriv = (1/self.n_samples) * np.matmul((model(theta, X)-Y).reshape(1,X.shape[0]),X[:,i])
            #theta[i] = np.matmul(diff_reshaped,X[:,i]) / self.n_samples
            theta_inc = theta.copy()
            theta_inc[i] += 1#np.clip(diff_abs_sum/np.exp(1),-1,1)
            sig_inc = model(theta_inc, bias, X)
            diff_inc = Y - sig_inc
            diff_inc_sum = np.sum(diff_inc)/(self.ptp*self.n_samples)
            diff_inc_abs_sum = np.abs(diff_inc_sum)
            p('theta', theta)
            p('bias', bias)
            #p('diff_inc',diff_inc)
            #p('diff_inc-diff',diff_inc-diff)
            print('diff_abs_sum', diff_abs_sum, ' / ', diff_abs_sum/np.exp(1))
            #p('diff_inc_sum',diff_inc_sum)
            #p('self.sigmoid_prime_max',self.sigmoid_prime_max(theta, bias))
            #p('np.max(Y)',np.max(Y))
            if diff_inc_abs_sum < diff_abs_sum or \
                    self.sigmoid_prime_max(theta, bias) < np.max(Y):
                theta_new.append(theta[i] + 1)  # * np.clip(diff_inc_sum,-1,1))
                theta_state = 1
                print('--higher thetas--')
            else:
                theta_inc = theta.copy()
                theta_inc[i] -= 1
                sig_inc = model(theta_inc, bias, X)
                diff_inc = Y - sig_inc
                diff_inc_sum = np.sum(diff_inc)/self.ptp*self.n_samples
                diff_inc_abs_sum = np.abs(diff_inc_sum)
                #p('diff_inc 2',diff_inc)
                #p('diff_inc-diff 2',diff_inc-diff)
                #p('diff_abs_sum',diff_abs_sum)
                #p('diff_inc_sum 2',diff_inc_sum)
                if diff_inc_abs_sum < diff_abs_sum and theta[i] > 1:
                    theta_new.append(theta[i] - 1)  # * np.clip(diff_inc_sum,-1,1))
                    theta_state = -1
                    print('--reduce thetas--')
                else:
                    theta_new.append(theta[i])
                    theta_state = 0
                    print('--same thetas--')

            if bias[i] == 0:
                bias_new.append(np.median(X[:, i]))              
            else:  #if theta_state < 1:
                bias_inc = bias.copy()
                bias_inc[i] += diff_abs_sum * self.learning_rate
                sig_p_p = self.sigmoid_prime_prime(theta_inc, bias, Y*X)
                mean_sig_p_p = np.mean(sig_p_p)
                p('sig_p_p', sig_p_p)
                print('sig_p_p mean', mean_sig_p_p)
                selective_regression = self.selective_regression(self.sigmoid_prime, theta_inc, bias_inc, X)
                p('selective_regression sigmoid_prime', selective_regression)
                selective_regression_2 = self.selective_regression(self.sigmoid_prime_prime, theta_inc, bias_inc, X)
                p('selective_regression_2 sigmoid_prime_prime', selective_regression_2)
                #p('bias_inc[i]',bias_inc[i])
                sig_inc = model(theta_inc, bias_inc, X)
                diff_inc = Y - sig_inc
                diff_inc_sum = np.sum(diff_inc)/(self.ptp*self.n_samples)
                diff_inc_abs_sum = np.abs(diff_inc_sum)
                #p('diff bias_inc_sum',diff_inc_sum)
                #else:
                bias_inc = bias.copy()
                #p('bias_inc[i] copy',bias_inc[i])
                bias_inc[i] = bias_inc[i] - 2 * diff_abs_sum * self.learning_rate * self.learning_rate
                sig_inc = model(theta_inc, bias_inc, X)
                diff_inc = Y - sig_inc
                diff_inc_sum = np.sum(diff_inc)/(self.ptp*self.n_samples)
                diff_inc_abs_sum_2 = np.abs(diff_inc_sum)
                #p('bias_inc[i] 2',bias_inc[i])
                #p('diff_inc-diff 2',diff_inc-diff)
                #p('diff_abs_sum',diff_abs_sum)
                #p('diff bias_inc_sum 2',diff_inc_abs_sum_2)
                if mean_sig_p_p > 0:
                    bias_add = bias + (0.1 * self.range_x)
                elif mean_sig_p_p < 0:
                    bias_add = bias - (0.1 * self.range_x)
                else:
                    bias_add = bias
                bias_new.append(bias_add)
#                if diff_inc_abs_sum < diff_abs_sum and diff_inc_abs_sum < diff_inc_abs_sum_2:
#                    #bias+= diff_abs_sum
#                    bias_new.append(bias[i]+(diff_abs_sum * self.learning_rate))
#                    print('--higher bias--')
#                elif diff_inc_abs_sum_2 < diff_abs_sum and diff_inc_abs_sum_2 < diff_inc_abs_sum and theta[i] > 1:
#                    #bias-= diff_abs_sum
#                    bias_new.append(bias[i]-(diff_abs_sum * self.learning_rate))
#                    print('--lower bias--')
#                else:
#                    bias_new.append(bias[i]+(selective_regression_2*0.5*10**np.log(selective_regression_2)))
#                    print('--same bias--')
#            elif bias[i]==0:
#                bias_new.append(np.median(X[:,i]))
#            else:
#                bias_new.append(bias[i])
            p('theta_state', theta_state)
            p('median', np.median(X[:, i]))
            #bias[i] = np.median(X[:,i])
        theta_new = np.array(theta_new).reshape((len(theta_new), 1))
        bias_new = np.array(bias_new).reshape((len(bias_new), 1))
        return theta_new, bias_new

    def gradient_descent(self, initial_model, X, Y, max_iterations, alpha,
                         starting_thetas=None, starting_bias=None, max_time=0,
                         tic_time=None):
        """performs gradient descent
        """
        self.n_samples = len(X)
        if starting_thetas is None:
            self.starting_thetas = np.ones((X.shape[1], 1))
        if starting_bias is None:
            self.starting_bias = np.zeros((X.shape[1], 1))
        self.predicted_thetas = self.starting_thetas
        self.predicted_bias = self.starting_bias
        self.progression = []
        cnt = max_iterations
        cout = 1
        if tic_time is None:
            tic_time = datetime.now()
        while cnt > 0 and ((max_time == 0) or (datetime.now()-tic_time).seconds < max_time):  # and (len(self.progression)<=100 or (len(self.progression)>100 and np.abs(self.cout_moyen)>0.00000001)):#np.abs(cout) > 0.00000001 and
            theta_new, bias = self.get_cost_derivative(initial_model,
                                                       self.predicted_thetas,
                                                       self.predicted_bias,
                                                       X, Y)
            p('theta_new', theta_new)
            p('bias_new', bias)
            self.predicted_thetas = theta_new
            self.predicted_bias = bias
            cout = self.regression_cost(initial_model, self.predicted_thetas,
                                        self.predicted_bias, X, Y)
            self.progression.append(cout)
            self.cout_moyen = np.mean(self.progression[-100:])
            cnt -= 1
        self.plot_progression()
        #self.plot_1_dimension(X, Y, target_thetas=starting_thetas)
        return self

    def randomize_model(self, theta, bias, X, range_x, random_ratio=0.0,
                        offsets=None):
        """Selective Regression Randomize function
            TODO: test sur random_ratio qui doit être entre 0.0 et 1.0
        """
        self.n_samples = X.shape[0]
        produit = []
        for x in X:
            sig = self.sigmoid_prime(theta, bias, x.reshape(1, len(x)))
            val = sig
            #val = 1 if sig > 0.5*self.sigmoid_prime_max(theta, bias) else 0
            # option de random pour ajouter du bruit statistique
            if random_ratio != 0.0:
                val = val if np.random.random() < random_ratio else 1 - val
            produit.append(val)
        return np.array(produit, order='F').reshape(self.n_samples, 1)

    def plot_1_dimension(self, X, Y, target_thetas=None, target_bias=None):
        """Selective Regression 1 dimensionnal plot
        """
        if (len(self.predicted_thetas) == 1):
            plt.figure()
            plt.plot(X, Y, 'o', label='original data')
            x = np.linspace(np.min(X), np.max(X), 100)
            y = []
            for var in x:
                sig = self.sigmoid_prime(self.predicted_thetas,
                                         self.predicted_bias, var)
                #p('sig',sig)
                y.append(sig)
            plt.plot(x, y, 'r')
            max_ = self.predicted_thetas[0][0].astype('int')
            if max_ > 1:
                for i in range(1, max_):
                    y = []
                    for var in x:
                        sig = self.sigmoid_prime(np.full(self.predicted_thetas.shape, i), self.predicted_bias, var)
                        #p('sig',sig)
                        y.append(sig)
                    plt.plot(x, y, c=(1.0, 1.0-(i/max_)**2, 1.0-(i/max_)**2))

            if target_thetas is not None:
                y = []
                for var in x:
                    sig = self.sigmoid_prime(target_thetas, target_bias, var)
                    y.append(sig)
                plt.plot(x, y, 'g')
            plt.legend()
            plt.show()


class MSAI_Regression():
    """Regression class
    """
    def __init__(self, max_iterations=500, learning_rate=0.3, randomize=0.0,
                 max_time=50, tic_time=None):
        self.__max_iterations = max_iterations
        self.__learning_rate = learning_rate
        self.__randomize = randomize
        self.__max_time = max_time
        self.__tic_time = tic_time
        self.__true_weights = None
        self.__true_biases = None
        self.__model = None
        self.__X = None
        self.__Y = None
        self.__n_dimensions = None
        self.__b_samples = None
        self.__use_classifier = None
        self.__range_x = None
       
    def __format_array(self, array):
        if type(array).__module__ == np.__name__:
            if len(array.shape) == 1:
                array = array.reshape(array.shape[0], 1)
            elif len(array.shape) == 0:
                array = np.array(array).reshape(1, 1)
        elif type(array) == 'pandas.core.series.Series':
            print('array.data', type(array.data), "\n", array.data)
        else:
            print('different type!!!!', type(array), array.shape, "\n", array)
            array = np.array(array).reshape(len(array), 1)
        return array

    def fit(self, X, Y, max_time=0, target_thetas=None, target_bias=None):
        """Solver Fit
        """
        X = self.__format_array(X)
        Y = self.__format_array(Y)
        self.__X = X.copy()
        n_samples, n_dimensions = X.shape
        self.set_n_samples(n_samples)
        self.set_n_dimensions(n_dimensions)
        self.__range_x = np.max(np.abs(X))
        self.__Y = Y.copy()
        #todo: tests sur les données
        self.__choose_classifier()
        self.__model.init_attribs_from_X(self.__X)
        if (max_time == 0) and (self.__max_time != 0):
            max_time = self.__max_time
        self.__model.fit(self.__X, self.__Y, max_time, self.__tic_time)
        self.__model.plot_1_dimension(X, Y, target_thetas, target_bias)
        return self

    def predict(self, X):
        """Solver Prediction
        """
        return self.__model.predict(X)

    def __choose_classifier(self):
        """Solver: automatic classifier choice
        """
        if self.__model is None:
            if(self.__Y.shape[1] == 1):
                self.__use_classifier = 'LinearRegression'
                min_ = self.__Y.min(axis=0)
                if (self.__Y.dtype == 'int32' or self.__Y.dtype == 'int64' \
                        or self.__Y.dtype == 'bool') and min_ >= 0:
                    unique = np.unique(self.__Y.astype(float))
                    if len(unique) < 10:
                        test = True
                        for item in unique:
                            if item.is_integer() == False:
                                test = False
                        if test == True:
                            self.__use_classifier = 'LogisticRegression'
            if self.__use_classifier == 'LogisticRegression':
                #print('---use of Logistic Regression---')
                self.__model = LogisticRegression(learning_rate=self.__learning_rate, max_iterations=self.__max_iterations)
            else:
                #print('---use of Linear Regression---')
                self.__model = LinearRegression(learning_rate=self.__learning_rate, max_iterations=self.__max_iterations)

    def set_linear_regression(self):
        self.__model = LinearRegression(learning_rate=self.__learning_rate, max_iterations=self.__max_iterations)

    def set_logistic_regression(self):
        self.__model = LogisticRegression(learning_rate=self.__learning_rate, max_iterations=self.__max_iterations)

    def set_selective_regression(self):
        self.__model = SelectiveRegression(learning_rate=self.__learning_rate, max_iterations=self.__max_iterations)

    def set_learning_rate(self, learning_rate):
        if self.__model is not None:
            self.__model.learning_rate = learning_rate

    def set_max_iterations(self, max_iterations):
        if self.__model is not None:
            self.__model.max_iterations = max_iterations

    def set_predicted_thetas(self, predicted_thetas):
        if self.__model is not None:
            self.__model.predicted_thetas = predicted_thetas

    def set_range_x(self, range_x):
        if self.__model is not None:
            self.__model.range_x = range_x

    def set_n_samples(self, n_samples):
        self.__n_samples = n_samples
        if self.__model is not None:
            self.__model.n_samples = n_samples

    def set_n_dimensions(self, n_dimensions):
        self.__n_dimensions = n_dimensions
        if self.__model is not None:
            self.__model.n_dimensions = n_dimensions

    def get_starting_thetas(self):
        if self.__model is not None:
            return self.__model.starting_thetas
        return None

    def get_predicted_thetas(self):
        if self.__model is not None:
            return self.__model.predicted_thetas
        return None

    def get_predicted_bias(self):
        if self.__model is not None:
            return self.__model.predicted_bias
        return None

    def get_mean(self):
        return self.__model.get_mean()

    def get_median(self):
        return self.__model.get_median()

    def get_std(self):
        return self.__model.get_std()

    def get_ptp(self):
        return self.__model.get_ptp()

    def get_range_x(self):
        return self.__range_x

    def get_classifier(self):
        return self.__use_classifier

    def severe_randomizer(self, class_type='LinearRegression', n_samples=50,
                          n_dimensions=10, range_x=10000):
        self.__range_x = range_x
        X = []
        self.n_samples = n_samples
        rand_offsets = []
        if class_type == 'LogisticRegression':
            self.__use_classifier = 'LogisticRegression'
            self.__true_weights = np.zeros((n_dimensions+1, 1))  #(np.random.random((n_dimensions+1,1))*range_x)-(range_x/2)
            self.__model = LogisticRegression(learning_rate=self.__learning_rate,
                                            max_iterations=self.__max_iterations,
                                            range_x=range_x,
                                            n_samples=n_samples)
        elif class_type == 'SelectiveRegression':
            self.__true_weights = np.zeros((n_dimensions, 1))  #(np.random.random((n_dimensions+1,1))*range_x)-(range_x/2)
            self.__true_biases = np.zeros((n_dimensions, 1))  #(np.random.random((n_dimensions+1,1))*range_x)-(range_x/2)
            self.__use_classifier = 'SelectiveRegression'
            self.__model = SelectiveRegression(learning_rate=self.__learning_rate,
                                             max_iterations=self.__max_iterations)
        else:
            self.__use_classifier = 'LinearRegression'
            self.__true_weights = np.zeros((n_dimensions+1, 1))  #(np.random.random((n_dimensions+1,1))*range_x)-(range_x/2)
            self.__model = LinearRegression(learning_rate=self.__learning_rate,
                                          max_iterations=self.__max_iterations,
                                          range_x=range_x,
                                          n_samples=n_samples)
        degre = np.floor(np.log10(range_x))
        if degre < 1:
            degre = 1
        rand_categories = []
        rand_signs = []
        for i in range(n_dimensions):
            rand_category = np.random.randint(1, degre+1)
            rand_categories.append(rand_category)
            rand_offset = (np.random.random()-0.5)*10**(rand_category+1)
            rand_offsets.append(rand_offset)
            sign = -1 if np.random.randint(0, 2) == 0 else 1
            rand_signs.append(sign)
            if class_type == 'LogisticRegression':
                sign = -1 if np.random.randint(0, 2) == 0 else 1
                self.__true_weights[i+1] = sign*10**(rand_categories[i])
                self.__true_weights[0] += rand_offsets[i]*self.__true_weights[i + 1]
            if class_type == 'SelectiveRegression':
                self.__true_weights[i] = 10**(rand_category)
                self.__true_biases[i] = -rand_offset+(np.random.random()-0.5) * self.__true_weights[i]
            
        for i in range(n_samples):
            row = []
            for j in range(n_dimensions):
                value = (np.random.random()-0.5)*range_x
                value *= 10**(rand_categories[j])
                value -= rand_offsets[j]
#                if class_type == 'LogisticRegression':
#                    self.__true_weights[j+1] = 0.5 * 10**(rand_categories[j])
#                    self.__true_weights[0] += 0.5 * rand_offsets[j]#*10**(rand_categories[j]) 
                row.append(value)
            X.append(row)

        X = np.array(X)
        if class_type == 'LinearRegression':
            self.__true_weights = (np.random.random((n_dimensions+1, 1)) * range_x)-(range_x/2)
        Y = self.__model.randomize_model(self.__true_weights, self.__true_biases,
                                       X, range_x, self.__randomize, rand_offsets) 
        #if self.__use_classifier == 'LogisticRegression':
        #    self.__true_weights/= np.absolute(self.__true_weights[:,0]).max()
        return X, Y, self.__true_weights, self.__true_biases
