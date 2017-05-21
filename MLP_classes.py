#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 17:27:55 2017

@author: bettmensch
"""
import numpy as np
import math
import dill
import os
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
    
def L_2_norm(matrix):
    """Takes a matrix and calculates its L^2 norm.
    Returns the matrix's L^2 norm."""
    return np.linalg.norm(matrix)
    
def sigmoid_mat(matrix):
    """Takes a matrix and applies the sigmoid function element_wise.
    Returns transformed matrix."""
    return np.divide(1, 1 + np.exp(-matrix))
    
def D_sigmoid_mat(matrix):
    """Takes a matrix and applies the sigmoid function's derivative element_wise.
    Returns transformed matrix."""
    return np.multiply(sigmoid_mat(matrix), 1 - sigmoid_mat(matrix))
    
def sigmoid_entropy(t,y):
    """Takes a target vector and a prediction vector y and calculates the
    sigmoidal cross entropy.
    Returns the sigmoidal cross entropy (scalar)."""
    return np.sum(-np.multiply(t, np.log(y)) - np.multiply(1 - t, np.log(1 - y)))

def sigmoid_entropy_mat(T,Y):
    """Takes a matrix T of target row vectors and a matrix Y of prediction row
    vectors and calculates the sigmoidal cross entropy for each row vector pair.
    Returns the total sigmoidal cross entropy (scalar)."""
    m = T.shape[0]
    return 1 / m * sum([sigmoid_entropy(t,y) for t,y in zip(T,Y)])

def tanh_mat(matrix):
    """Takes a matrix and applies the tanh function element_wise.
    Returns transformed matrix."""
    return np.tanh(matrix)
    
def D_tanh_mat(matrix):
    """Takes a matrix and applies the tanh function's derivative element_wise.
    Returns transformed matrix."""
    return 1 - np.multiply(tanh_mat(matrix), tanh_mat(matrix))
    
def softmax(vector):
    """Takes a vector and applies the softmax.
    Returns transformed vector."""
    return np.exp(vector) / np.sum(np.exp(vector))

def softmax_mat(matrix):
    """Takes a matrix and applies the softmax to each row.
    Returns transformed matrix."""
    return np.matrix(np.vstack([softmax(row) for row in matrix]))

def D_softmax_mat(matrix):
    """"""
    pass
    
def softmax_entropy(t,y):
    """Takes a target vector t and a prediction vector y and calculates the 
    cross entropy.
    Returns the cross entropy (scalar)."""
    return np.sum(-np.multiply(t,np.log(y)))

def softmax_entropy_mat(T,Y):
    """Takes a matrix T where "one row = one target label".
    Takes a matrix Y where "one row = one predicted label".
    Returns the total cross entropy for a softmax output layer."""
    m = T.shape[0]
    return 1 / m * sum([softmax_entropy(t,y) for t,y in zip(T,Y)])

def relu_mat(matrix):
    """"""
    pass
    
def D_relu_mat(matrix):
    """"""
    pass

class Layer(object):
    """Multifunctional layer class.
    Will be attached to the model during model initialization."""
    def __init__(self, size, activation):
        
        self.size = size
        self.type = activation
        self.A = None
        self.B = None
        # by default, layer is a hidden layer
        self.first = False
        self.last = False
        
        if activation == 'sigmoid':
            self.activate = sigmoid_mat
            self.D_activate = D_sigmoid_mat
        elif activation == 'tanh':
            self.activate = tanh_mat
            self.D_activate = D_tanh_mat
        elif activation == 'softmax':
            self.activate = softmax_mat
            self.D_activate = D_softmax_mat
        #elif activation == 'relu':
        #    self.activate = relu_mat
        #    self.D_activate = D_relu_mat
        elif activation == 'softmax':
            self.activate = softmax_mat
            self.D_activate = D_softmax_mat
            
    def __str__(self):
        """The object as it presents itself to the console."""
        bio = "Size: " + str(self.size) + "\n" + "Type: " + str(self.type) + "\n"
        
        return bio
            
    def fix(self, position):
        """Takes a layer position within model (type string).
        Updates models settings accordingly."""
        
        if position == 'first':
            self.activate = None
            self.D_activate = None
            self.first = True
        elif position == 'last':
            self.D_activate = None
            self.last = True
                    
class MLP(object):
    """MultiLayerPerceptron class"""
    def __init__(self):
        """Takes an iterable layers containing the layer_sizes. The length of
        the iterable determines the depth of the MLP."""
        
        self.layers = []
        self.lb = LabelBinarizer()
        self.loss_function = None
        self.Ws = None
        self.fs = None
    
    def add_layer(self, size, activation = 'tanh'):
        """Takes a layer size (type int) and an activation (type string).
        Adds a layer object to the blue_print attribute."""
        self.layers.append(Layer(size, activation))
        
    def fix(self):
        """Marks the end of the model design process.
        Sets the input and output layers and the loss function. Model training
        can begin after this method is called."""
        first_layer = self.layers[0]
        first_layer.fix('first')
        last_layer = self.layers[-1]
        
        if last_layer.type == 'softmax':
            last_layer.fix('last')
            self.loss_function = softmax_entropy_mat
        elif last_layer.type== 'sigmoid':
            last_layer.fix('last')
            self.loss_function = sigmoid_entropy_mat
        else:
            print("Last layer must be either sigmoid or softmax!")
        
    def fit(self, X_train, y_train,
            batch_size = 100, max_iter = 700,
            learning_rate = 0.3, reg_param = 0):
        """Takes training data X_train (type array or matrix).
        Takes training labels (type list, array or matrix) in categorical format.
        Takes maximum number of iterations max_iter (type int).
        Takes regularization parameter reg_param (type float)."""
        
        m = X_train.shape[0]
        Ws, fs = self.init_params(batch_size)                                                 
        Y_train = self.lb.fit_transform(y_train)
        
        for i in range(max_iter):
            # create batch for current forward-backward-loss calculation
            idx = np.random.randint(m, size = batch_size)
            X_batch = X_train[idx,:]
            Y_batch = Y_train[idx,:]
            
            L_i = self.forward_prop(X = X_batch, Y = Y_batch, Ws = Ws, fs = fs, mode = 'fit') \
                                   + self.reg_term(m, Ws, reg_param)
            
            print("Iteration ", i)
            print("Loss: ", L_i)
            
            DWs, Dfs = self.backward_prop(Y_batch, Ws, fs)
            
            Ws = [(1 - (reg_param * learning_rate) / m) * W - learning_rate * DW for W, DW in zip(Ws, DWs)]
            fs = [f - learning_rate * Df for f, Df in zip(fs, Dfs)]
            
        self.Ws = Ws
        self.fs = [f[0,:] for f in fs]
        
    def reg_term(self, m, Ws, reg_param = 0):
        """Takes a batch size m (type int).
        Takes weights Ws (type iterable, contains weight matrices).
        Takes regularization parameter reg_param (type float).
        Returns regularization penalty term (type float)."""
        
        if reg_param:
            reg_term = reg_param / (2*m) * sum([np.linalg.norm(W) for W in Ws])
            
            return reg_term
        else:
            return 0
        
    def transform(self, X):
        """Takes data (type array or matrix).
        Returns class predictions (type array or list)."""
        
        m = X.shape[0]
                                  
        if self.Ws:
            Ws = self.Ws
            fs = [np.matrix(np.vstack([f for i in range(m)])) for f in self.fs]
            
            y_pred = self.forward_prop(X = X, Ws = Ws, fs = fs, mode = 'transform')
        
            return y_pred
        
        else:
            print("Model needs to be fitted first!")
    
    def fit_transform(self, X_train, y_train,
                      batch_size = 100, max_iter = 700,
                      learning_rate = 0.3, reg_param = 0):
        """Takes training data X_train (type array or matrix).
        Takes training labels (type list, array or matrix) in categorical format.
        Takes maximum number of iterations max_iter (type int).
        Takes regularization parameter reg_param (type float)."""
        
        self.fit(X_train, y_train, batch_size, max_iter, learning_rate, reg_param)
        
        y_train_pred = self.transform(X_train)
        
        return y_train_pred
    
    def evaluate(self, X_test, y_test):
        """Takes a matrix of test samples where "one row = one sample".
        Takes a matrix of test labels where "one row = one label".
        Calculates predictions and evaluates accuracy based on predictions
        and test labels.
        Returns an accuracy score."""
        
        # calculate model predictions
        y_pred = self.transform(X = X_test)
        
        print(confusion_matrix(y_test, y_pred, labels = self.lb.classes_))
        
        print(classification_report(y_test, y_pred, labels = self.lb.classes_))
            
    def init_params(self, m):
        """Randomly initializes weights Ws (type list, contains matrices) and 
        biases fs (type list, contains matrices)."""
        
        layer_sizes = [layer.size for layer in self.layers]
        
        Ws, fs = [], []
        
        for i in range(len(layer_sizes)-1):
            rows, columns = layer_sizes[i], layer_sizes[i+1]
            
            epsilon = math.sqrt(6) / math.sqrt(rows + columns)
            high = epsilon * np.matrix(np.ones((rows, columns)))
            low = - high
    
            Ws.append(np.matrix(np.random.uniform(low, high)))
            fs.append(np.matrix(np.ones((m, columns))))
        
        return Ws, fs
    
    def forward_prop(self, X, Ws, fs, mode, Y = None):
        """Takes samples X (type matrix) and labels Y (type matrix).
        Takes weights Ws (type list, containing matrices).
        Takes biases fs (type list, contains matrices).
        Takes propagation mode (type string).
        Returns loss function value L_i (type float) if in fit mode.
        Returns model prediction P (type matrix) if in transform mode."""
        
        first_layer = self.layers[0]
        first_layer.B = X
        
        for i in range(len(Ws)):
            current_layer = self.layers[i+1]
            previous_layer = self.layers[i]
            
            current_layer.A = np.dot(previous_layer.B, Ws[i]) + fs[i]
            current_layer.B = current_layer.activate(current_layer.A)
            
        last_layer = self.layers[-1]
        P = last_layer.B
        
        if mode == 'transform':
            y_pred = self.lb.inverse_transform(P)
            
            return y_pred
        
        elif mode == 'fit':
            L_i = self.loss_function(Y, P)
            
            return L_i
    
    def backward_prop(self, Y, Ws, learning_rate):
        """Takes lables Y (type matrix).
        Takes weights Ws (type list, containing matrices).
        Takes biases fs (type list, contains matrices).
        Applies backpropagation.
        Returns updated weights Ws (type list, contains matrices).
        Returns updated biases fs (type list, contains matrices)."""
        
        m = Y.shape[0]
        DWs, Dfs = [], []
        
        last_layer = self.layers[-1]
        P = last_layer.B
        Delta = (P - Y) / m
                                  
        for i in range(len(Ws),0,-1):
            previous_layer = self.layers[i-1]
            
            DW = np.dot(previous_layer.B.T, Delta)
            Df = np.matrix(np.vstack([np.sum(Delta,0) for i in range(m)]))
            
            DWs.append(DW)
            Dfs.append(Df)
            
            if i != 1:
                Delta = np.multiply(np.dot(Delta, Ws[i-1].T),
                                    previous_layer.D_activate(previous_layer.A))
            
        DWs.reverse()
        Dfs.reverse()
        
        return DWs, Dfs
    
    def __str__(self):
        """The way the model presents itself to the console."""
        
        bio = ""
        
        for layer in self.layers:
            if layer.first:
                bio += "Input Layer \n"
            elif layer.last:
                bio += "Output layer \n"
            else:
                bio += "Hidden Layer \n"
                
            bio += layer.__str__() + "------------\n"
            
        return bio
    
    def save(self, file_name, directory):
        """Uses dill to save current model to disk at the specified location."""
        work_dir = os.getcwd()
        
        os.chdir(directory)
                            
        with open(file_name, 'wb') as dump_file:
            dill.dump(self, dump_file)
        
        os.chdir(work_dir)
        
    def clone(self):
        """Returns a copy of current model."""
        
        clone = MLP()
        clone.layers = self.layers
        clone.lb = self.lb
        clone.loss_function = self.loss_function
        clone.Ws = self.Ws
        clone.fs = self.fs
        
        return clone