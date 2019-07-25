#!/usr/bin/env python
# coding: utf-8

# # Réseau de neurones à propagation avant (Feed-Forward)

# In[1]:


from keras.models import model_from_json
import numpy as np
import pandas as pd

class Reseau():
    
    # Initialisation du réseau de neurones
    def __init__(self):
        self.accuracy = 0 # sum of the two accuracies
        self.weights = {} # dictionnaire des poids du réseau de neurones
        self.biases = {} # dictionnaire des bias du réseau de neurones
        
    """
        @model_file: le fichier json où est enregistré le modèle entrainé
        @weights_file: le fichier hdf5 où est enregistré les poids du modèle entrainé
        Initialisation du réseau de neurones avec le modèle entrainé    
    """
    def __init__(self, model_file, weights_file):
        # Charger le fichier json et créer le modèle
        json_file = open(model_file, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # Charger les poids du modèle
        loaded_model.load_weights(weights_file)
        print("Le modèle et les poids sont chargés")
        print("Nombre de couches: {}".format(len(loaded_model.layers)))
        # Initialiser les attributs
        self.accuracy = 0 # sum of the two accuracies
        self.weights = {} # dictionnaire des poids du réseau de neurones
        self.biases = {} # dictionnaire des bias du réseau de neurones
        for indice, layer in enumerate(loaded_model.layers):
            if len(layer.get_weights())!=0:
                weights = layer.get_weights()[0]
                biases = layer.get_weights()[1]
                self.weights[indice] = weights 
                self.biases[indice] = biases 
    
    """
        @x: tableau numpy à 1d, vecteur colonne représentant un exemple d'apprentissage
        @retourne la valeure de sortie du réseau de neurone correspendante à l'exemple d'apprentissage
        Calcule le Feedforward wT*X+b
    """
    def feedforward(self, x):
        for indice, (w, b) in enumerate(zip(self.weights, self.biases)):
            if indice != len(self.weights)-1:
                x = self.relu(np.dot(np.transpose(self.weights[w]),x)+self.biases[b].reshape(-1,1))
            else:
                x = self.sigmoid(np.dot(np.transpose(self.weights[w]),x)+self.biases[b].reshape(-1,1))
        return x
    
    """
        @x: tableau numpy ou valeur
        @retourne le résultat de la fonction Sigmoid
    """
    def sigmoid(self, x):
        x = x.astype(float)
        return 1.0/(1.0+np.exp(-x))
    
    """
        @x: tableau numpy ou valeur
        @retourne le résultat de la fonction ReLu
    """
    def relu(self, x):
        return np.maximum(x, 0)
    
    """
        @yHat: valeur prédite pour un exemple d'apprentissage
        @y: valeur réelle (label) d'un exemple d'apprentissage
        @retourne le résultat de la fonction d'erreur Binary cross entropy
    """
    def BinaryCrossEntropy(self, yHat, y):
        if y == 1:
          return -np.log(yHat.astype(float))
        else:
          return -np.log(1 - yHat.astype(float))
    
    """
        @X: data du test contenant les features 
        @y: labels de la data X
        @retourne l'erreur globale du réseau de neurones
    """
    def calcul_erreur(self, X, y):
        erreur=0
        # Pour chaque exemple d'apprentissage
        for i in range(X.shape[0]): 
            predicted = self.feedforward(X[i].reshape(-1,1)) # La valeur prédite
            actual = y[i].reshape(-1,1) # La valeur réelle 
            erreur += self.BinaryCrossEntropy(predicted, actual)  # La fonction d'erreur binary cross entropy
        return erreur
    
    """
        @X1: data-test de la première dataset (films)
        @y1: labels de la première dataset
        @X2: data-test de la deuxième dataset (titanic)
        @y2: labels de la deuxième dataset
        @retourne la justesse (accuracy) du réseau de neurones sur les deux dataset
    """
    def justesse(self, X1, y1, X2, y2):
        self.accuracy = 0
        # On boucle sur la première dataset test
        for i in range(X1.shape[0]):
            output = self.feedforward(X1[i].reshape(-1,1))
            if output >= 0.5:
                output = 1
            else:
                output = 0
            self.accuracy += int(output == y1[i])
        # On boucle sur la deuxième dataset test
        for i in range(X2.shape[0]):
            output = self.feedforward(X2[i].reshape(-1,1))
            if output >= 0.5:
                output = 1
            else:
                output = 0
            self.accuracy += int(output == y2[i])
        self.accuracy = self.accuracy / (X1.shape[0]+X2.shape[0]) * 100
        return self.accuracy

