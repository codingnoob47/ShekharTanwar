#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 16:34:47 2019

@author: shekhartanwar
"""

import numpy as np
import operator
import math
from sys import argv
import time
import os


def read_input(input_file):
    store_dict = {}
    
    with open(input_file,'r') as infile:
        store_dict =  infile.readlines()
    return store_dict

def initialize_paramters(read_dict):
    length_vector =  len(read_dict)
    weights = np.zeros((length_vector,1))
    bias = 0
    return weights,bias


def process_features(temp_dict, features):
    for each_data_point in features:
            results = each_data_point.split(":")
            key = results[0]
            value = int(results[1])
            temp_dict[key] = value
    return temp_dict
       
def parse_input(input_file):
    
    store_dict = read_input(input_file)
    labels = []
    word_index_dict = []
    for each_line in store_dict:
        temp_dict = {}
        content = each_line.rstrip().split('\t')
        labels.append(int(content[0]))
        features = content[1:]
        word_index_dict.append(process_features(temp_dict,features))
    
    return labels, word_index_dict

def f_grad(weights_matrix, data_point, delta_update):
    for key,value in data_point.items():
        weights_matrix[int(key)] = value * delta_update
    return weights_matrix
    

def find_dot_product(input_data, weights, bias):
    result = 0.0    
    for key,value in input_data.items():
         index = weights[int(key)]
         product = value * index
         result +=  (product)

    result = result + bias
    return result
    
def find_gradient(data_point, delta_update, length):
    weights_matrix = np.zeros((length,1))
    return f_grad(weights_matrix, data_point, delta_update)


def compute_weight_bias(theta_T_X, train_labels, each_example,train_data, learning_rate, weights, bias):
   
     factor = 1 + np.exp(-theta_T_X)
     
     sigmoid_value = 1/factor
     delta_update = -1 * (train_labels[each_example] - sigmoid_value)[0]
     weight_gradient = find_gradient(train_data[each_example], delta_update, len(weights))
     weights = np.subtract(weights, learning_rate * weight_gradient)
     bias =  bias  - learning_rate * delta_update
     return weights, bias
    

def stochastic_GD(train_data, train_labels, weights, bias, learning_rate, num_epoch):
    
    length = len(train_data)
    for i in range(0, num_epoch):
        for each_example in range(0, length):            
#            theta_T_X = find_dot_product(train_data[each_example], weights, bias)            
            weights, bias = compute_weight_bias(find_dot_product(train_data[each_example], weights, bias), train_labels, 
                                                each_example,train_data,learning_rate, weights, bias)
    return weights, bias
            

def label_error_computation(data,actual_labels,weights,bias):
    
    labels = []
    for each_data_point in data:
        result = find_dot_product(each_data_point,weights, bias)
        labels.append(int(get_labels(result)))
    
    compute_performance(actual_labels,labels)
    return labels
        
    
error = []
def compute_performance(actual_labels, predicted_labels):
     global error
     mismatch =  [y for x, y in zip(actual_labels, predicted_labels) if y != x]
     error_rate = float(len(mismatch)/ len(actual_labels))
     error.append(error_rate)

def get_labels_error(train_dictionary,train_labels,test_dictionary,test_labels,adjusted_weights,adjusted_bias):
    train_labels = label_error_computation(train_dictionary,train_labels,  adjusted_weights, adjusted_bias)
    test_labels  = label_error_computation(test_dictionary,test_labels,  adjusted_weights, adjusted_bias)
    
    return train_labels, test_labels


def transform_data(formatted_train_input,formatted_validation_input,formatted_test_input, weights, bias, train_out,test_out,metrics_out,learning_rate,num_epoch):

    train_labels, train_dictionary =  parse_input(formatted_train_input)
    test_labels, test_dictionary   =  parse_input(formatted_test_input)
    
    computeSGD_Error(train_labels,train_dictionary,test_labels,test_dictionary,weights, bias, train_out,test_out,metrics_out,learning_rate,num_epoch)
    
    
def computeSGD_Error(train_labels,train_dictionary,test_labels,test_dictionary,weights, bias, train_out,test_out,metrics_out,learning_rate,num_epoch):
    adjusted_weights, adjusted_bias =  stochastic_GD(train_dictionary,train_labels, weights, bias, learning_rate, num_epoch)
    train_labels, test_labels = get_labels_error(train_dictionary,train_labels,test_dictionary,test_labels,adjusted_weights,adjusted_bias)
    write_file(train_out,test_out,metrics_out,train_labels,test_labels)
        
    
def get_labels(result):
    
    sigmoid_computation = 1/ (1 + np.exp(-result))
    label_generated  =  np.round(sigmoid_computation)[0]
    return label_generated
    
def write_file(train_out,test_out,metrics_out,train_labels,test_labels):
    global error
     
    train_error = error[0]
    test_error = error[1]

    with open(metrics_out, 'w') as outfile:
        outfile.write('error(train): ' + str(train_error) + '\n')
        outfile.write('error(test): ' + str(test_error))

    file = open(train_out,"w")
    for i in range(len(train_labels)):
        file.write(str(train_labels[i]) + "\n")
    
    file = open(test_out,"w")
        
    for i in range(len(test_labels)):
        file.write(str(test_labels[i]) + "\n")
    
    file.close()  
    
    
def processData(formatted_train_input,formatted_validation_input,formatted_test_input,dict_input,train_out,test_out,metrics_out,num_epoch):

    read_dict = {}
    learning_rate = 0.1
    
    with open(str(dict_input),'r') as input_dict:
        read_dict = input_dict.readlines()

    weights,bias = initialize_paramters(read_dict)
    transform_data(formatted_train_input,formatted_validation_input,formatted_test_input, weights, bias, train_out,test_out,metrics_out,learning_rate,num_epoch)




formatted_train_input = "formatted_train_out_mine.tsv"
formatted_validation_input = "formatted_valid_out_mine.tsv"
formatted_test_input = "formatted_test_out_mine.tsv"
dict_input = "dict.txt"
train_out = "train_out.labels"
test_out = "test_out.labels"
metrics_out = "metrics_out.txt"
num_epoch  =  30
##    
processData(formatted_train_input,formatted_validation_input,formatted_test_input,dict_input,train_out,test_out,metrics_out,num_epoch)

#
#if len(argv) == 9:  
#    
#    formatted_train_input = str(argv[1])
#    formatted_validation_input = str(argv[2])
#    formatted_test_input = str(argv[3])
#    dict_input = str(argv[4])
#    train_out = str(argv[5])
#    test_out = str(argv[6])
#    metrics_out = str(argv[7])
#    num_epoch  =  int(argv[8])
#    
#    processData(formatted_train_input,formatted_validation_input,formatted_test_input,dict_input,train_out,test_out,metrics_out,num_epoch)
#    
#else:
#    print("Incorrect number of arguments passed")
#
#   
