#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:41:45 2017
Loads MNIST data.
Creates MLP object instance.
Trains MLP model on MNIST training set.
@author: bettmensch
"""

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from mnist_data import get_mnist_data
from MLP_classes import MLP
import argparse
import os

def main():
    """Takes the MLP class for a test drive"""
    
    parser = argparse.ArgumentParser(description = 'Train MLP on MNIST dataset')
    
    parser.add_argument('-mi', '--max_iter',
                        required = False,
                        default = 50,
                        type = int,
                        help = 'Number of iterations for stochastic gradient descent')
    parser.add_argument('-HL_0', '--HL_0',
                        type = int,
                        required = False,
                        default = 784,
                        help = 'Set the size of the input layer.')
    parser.add_argument('-HL_1', '--HL_1',
                        type = int,
                        required = False,
                        default = 500,
                        help = 'Set the size of the second layer.')
    parser.add_argument('-HL_2', '--HL_2',
                        type = int,
                        required = False,
                        default = 500,
                        help = 'Set the size of the third layer.')
    parser.add_argument('-HL_3', '--HL_3',
                        type = int,
                        required = False,
                        default = 500,
                        help = 'Set the size of the fourth layer.')
    parser.add_argument('-HL_4', '--HL_4',
                        type = int,
                        required = False,
                        default = 10,
                        help = 'Set the size of the output layer.')
    parser.add_argument('-bs', '--batch_size',
                        required = False,
                        type = int,
                        default = 100,
                        help = 'Set the size of the random samples chosen in each stochastic gradient computation.')
    parser.add_argument('-lr', '--learning_rate',
                        required = False,
                        type = float,
                        help = 'Set the learning rate for the stochastic gradient descent.',
                        default = 0.03)
    parser.add_argument('-rp', '--reg_param',
                        required = False,
                        type = float,
                        default = 0,
                        help = 'Set weight parameter for regularization penalty term.')
    parser.add_argument('-ot','--output_type',
                        required = False,
                        default = 'softmax',
                        help = 'Set the type of the output layer activation function.',
                        choices = ['sigmoid','softmax'])
    
    
    opts = vars(parser.parse_args())
    
    max_iter = opts['max_iter']
    HL_0 = opts['HL_0']
    HL_1 = opts['HL_1']
    HL_2 = opts['HL_2']
    HL_3 = opts['HL_3']
    HL_4 = opts['HL_4']
    reg_param = opts['reg_param']
    output_type = opts['output_type']
    batch_size = opts['batch_size']
    learning_rate = opts['learning_rate']
    
    print("Getting data...")
    X_train, y_train, X_test, y_test = get_mnist_data()
    print("Got data. Creating...")

    model = MLP()
    
    model.add_layer(HL_0)
    model.add_layer(HL_1)
    model.add_layer(HL_2)
    model.add_layer(HL_3)
    model.add_layer(HL_4, output_type)
    
    model.fix()
    
    input("Created model. Press enter to view blue print.")
    
    print(model)
    
    input("Press enter to fit model on training data.")
    # testing fit method
    model.fit(X_train = X_train, y_train = y_train,
              reg_param = reg_param, batch_size = batch_size,
              max_iter = max_iter, learning_rate = learning_rate)
    
    input("Press enter to transform test data.")
    print("Predicting...")
    
    # testing transform method
    y_pred_test = model.transform(X_test)
    
    input("Test data transformed. Press enter to view evaluation summary.")
    #testing evaluate method
    model.evaluate(X_train, y_train)
    
    print(classification_report(y_test, y_pred_test))
    
    input("Press enter to quit.")
    
    
if __name__ == "__main__":
    main()