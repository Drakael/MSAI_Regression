# -*- coding: utf-8 -*-
"""
=================================================
MSAI Solver Class with linear and logistic regression
=================================================
On 2018 march
Author: Drakael Aradan
"""
from MSAI_Regression import MSAI_Regression
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np


def p(mess, obj):
    """Useful function for tracing"""
    if hasattr(obj, 'shape'):
        print(mess, type(obj), obj.shape, "\n", obj)
    else:
        print(mess, type(obj), "\n", obj)

#on ferme toutes les éventuelles fenêtres
plt.close('all')
#on enregistre le temps courant
# = on démarre le chronomètre
tic_time = datetime.now()

#variables de base
n_dimensions = 1
n_samples = 38
range_x = 1
max_iterations = 10
learning_rate = 0.1
randomize = 0.0
max_execution_time = 111
true_weights = None

#déclaration du solveur
solver = MSAI_Regression(max_iterations, learning_rate, randomize,
                         max_execution_time, tic_time)

#solver.set_selective_regression()

#initialisation aléatoire du set d'entrainement
X, Y, true_weights, true_biases = solver.severe_randomizer('SelectiveRegression', n_samples, n_dimensions, range_x)

#X = np.random.normal(0, 2, 100)
#Y = np.random.normal(0, 2, 100)

#from sklearn.datasets import fetch_california_housing
#dataset = fetch_california_housing()
#X, Y = dataset.data, dataset.target
#n_samples, n_dimensions = X.shape
#Y = Y.reshape(len(Y),1)

#degre = np.floor(np.log10(range_x))
#if degre < 1:
#    degre = 1
#rand_categories = []
#rand_offsets = []
#for i in range(n_dimensions):
#    rand_category = np.random.randint(0,degre)
#    rand_categories.append(rand_category)
#    #rand_offset = np.random.randint(0,degre)-((degre-1)/2)
#    rand_offset = (np.random.random()-0.5)#*10**rand_category
#    rand_offsets.append(rand_offset)
#X, Y = make_classification(n_samples=n_samples,
#                           n_features=n_dimensions,
#                           n_informative=n_dimensions,
#                           #scale=range_x,
#                           shift=rand_offsets,
#                           n_redundant=0,
#                           n_repeated=0,
#                           n_classes=2,
#                           random_state=np.random.randint(100),
#                           shuffle=False)
#Y = Y.reshape(len(Y),1)

#affichages préliminaires
p('X', X)
p('true_weights', true_weights)
p('true_biases', true_biases)
#p('theta_initial',theta_initial)
p('Y', Y)

#entrainement du modèle sur les données
solver.fit(X, Y, target_thetas=true_weights, target_bias=true_biases)

#affichage finaux
predicted_thetas = solver.get_predicted_thetas()
predicted_bias = solver.get_predicted_bias()
print('Thetas start', "\n", solver.get_starting_thetas())   
if true_weights is not None:
    print("Thetas target\n", true_weights) 
if true_biases is not None:
    print("Biases target\n", true_biases) 
print('Theta end : ', "\n", predicted_thetas)
print('Bias end : ', "\n", predicted_bias)
print('Means : ', "\n", solver.get_mean())
print('Medians : ', "\n", solver.get_median())
print('StDs : ', "\n", solver.get_std())
print('Ranges : ', "\n", solver.get_ptp()) 
if true_weights is not None:
    print('Erreurs : ', "\n", true_weights-predicted_thetas)
    global_error = np.sum(true_weights-predicted_thetas)+np.sum(true_biases-predicted_bias)
    print('Erreur globale : ', "\n", global_error)
    print('Erreur moyenne : ', "\n", global_error/(len(X)))
    print('Erreur relative : ', "\n", global_error/(len(X)*(range_x**2)*(n_dimensions**2)))
range_x = solver.get_range_x()
print('Range of values :', range_x)
print('Model :', solver.get_classifier())
#arrêt du chronomètre
delta_time = (datetime.now()) - tic_time
#affichage du temps de calcul global
print('Script executed in', delta_time.days, 'd', delta_time.seconds, 's',
      delta_time.microseconds, 'µs')
