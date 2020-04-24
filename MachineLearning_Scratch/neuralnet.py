#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 21:34:13 2019

@author: shekhartanwar
"""

import numpy as np
import math
import csv
from sys import argv

y_size = 10
feature_size =129
def initialize_labels_features(length):
    global y_size
    global feature_size
    
    labels = np.zeros((length,y_size))
    features = np.ones((length,feature_size))
    return labels,features

def update_labels_features(labels,features,input_file):
    with open(input_file) as infile:
        csv_reader = csv.reader(infile, delimiter = ',', quotechar='|')
    
        row_index = 0
        for row in csv_reader:
            labels[row_index, int(row[0])] = 1
            features[row_index,1:] = list(map(float, row[1:]))
            row_index += 1
    return labels, features
    
def process_Data(input_file):
    input_list = []
    
    with open(input_file,"r") as infile:
        input_list = infile.readlines()
        
    labels,features =  initialize_labels_features(len(input_list))
    updated_labels,updated_features = update_labels_features(labels,features,input_file)    
    return updated_labels,updated_features

def random_initalizer(hidden_units, X_train):
    global y_size
    alpha = np.zeros((hidden_units,X_train.shape[1]))
    alpha[:,1:] = np.random.uniform(-0.1,0.1, (hidden_units, X_train.shape[1]-1))
    beta = np.zeros((y_size, hidden_units+1))
    beta[:,1:] = np.random.uniform(-0.1,0.1, (y_size, hidden_units))
    
    return alpha,beta
       
def zero_initializer(hidden_units, X_train):
    global y_size
    alpha = np.zeros((hidden_units, X_train.shape[1]))
    beta = np.zeros((y_size,hidden_units+1))
    return alpha,beta


def initialize_alpha_beta(initialization_flag, hidden_units, X_train):

    if initialization_flag == 1:
        alpha,beta = random_initalizer(hidden_units, X_train)
    if initialization_flag == 2:
        alpha,beta = zero_initializer(hidden_units,X_train)
    
    return alpha,beta
    
def sigmoid(matrix1, matrix2,mode):
    if mode == "forward":
        result =  1/(1+np.exp(-matrix1))
        result = np.append(1,result)
    if mode == "backward":
        result = np.multiply(matrix2.ravel(), np.multiply(matrix1,1-matrix1))
        result = result[1:]
        result = np.reshape(result, (1,result.shape[0]))
    return result

def linear_operation(matrix1,matrix2,matrix3, mode):
    if mode == "forward":
        result =  np.matmul(matrix2,matrix1)
    if mode ==  "backward":
        matrix1 = np.reshape(matrix1, (matrix1.shape[0],1))
        result = np.matmul(matrix3.T, matrix1.T), np.matmul(matrix2.T, matrix3.T)
    return result
        

def softmax(input_vector1,input_vector2, mode):
    if mode == "forward":
        result = np.divide(np.exp(input_vector1),np.sum(np.exp(input_vector1)))
    if mode == "backward":
        vector1 = (input_vector1.shape[0],1)
        vector2 = (input_vector2.shape[0],1)
        grad_b = np.reshape(input_vector2, vector2)
        b_mat = np.reshape(input_vector1, vector1)
        vec1 = np.diag(input_vector1)
        vec2 = np.matmul(b_mat,b_mat.T)
        result = np.matmul(grad_b.T, np.subtract(vec1, vec2))
    return result
        
def predict_labels(X_input,y_input,alpha, beta):
        labels_list = np.zeros((X_input.shape[0],1))
        counter = X_input.shape[0]
        for i in range(0,counter):
            X,y =  X_input[i,:], y_input[i,:]
            X,a,z,b,y_hat,J = forward_propagation(X,y,alpha,beta) 
            labels_list[i] = np.argmax(y_hat)

        return labels_list
    
def cross_entropy(y,y_pred, mode):
    if mode == "forward":
        result =  -np.matmul(y.T,np.log(y_pred))
    if mode == "backward":
        result = -np.divide(y,y_pred)
    return result


def forward_propagation(X,y,alpha,beta):    
# alpha here are the weights    
    mode = "forward"
    a = linear_operation(X,alpha,y, mode)
    z = sigmoid(a,y, mode)
    b = linear_operation(z,beta,y,mode)
    y_pred = softmax(b,z,mode)
    J = cross_entropy(y,y_pred,mode)
    return X,a,z,b,y_pred,J
    
def backward_propagation(X,y, a,z,b,y_pred,J, alpha, beta):
    mode  = "backward"
    gradient_y_hat = cross_entropy(y,y_pred, mode)
    grad_b = softmax(y_pred,gradient_y_hat, mode)
    gradient_beta, gradient_z = linear_operation(z,beta,grad_b,mode)
    grad_a = sigmoid(z,gradient_z ,mode)
    grad_alpha, grad_x = linear_operation(X,alpha,grad_a,mode)
    return grad_alpha, gradient_beta
       
def update_paramters(alpha,beta,learning_rate,gradient_alpha,gradient_beta):
    alpha =  alpha - learning_rate * gradient_alpha
    beta =  beta - learning_rate * gradient_beta
    return alpha,beta

def get_enotrpy_per_epoch(entropy_list,X_input, Y_input, alpha, beta):
    length =  X_input.shape[0]
    for example in range(0,length):
        X,y = X_input[example,:], Y_input[example,:]
        X,a,z,b,y_hat,J = forward_propagation(X,y,alpha,beta) 
        entropy_list.append(J)
    return entropy_list
        
def train_model(alpha,beta, num_epochs, X_train, Y_train,X_test,Y_test,learning_rate, train_cross_entropy, test_cross_entropy):

    len_X_Train =  X_train.shape[0]    
    for epoch_number in range(0,num_epochs):
        for each_row in range(0,len_X_Train):
            X,y = X_train[each_row,:], Y_train[each_row,:]
            X,a,z,b,y_hat,J = forward_propagation(X,y,alpha,beta) 
            gradient_alpha, gradient_beta = backward_propagation(X,y, a,z,b,y_hat,J, alpha,beta)
            alpha,beta = update_paramters(alpha,beta,learning_rate,gradient_alpha,gradient_beta)
        
        train_entropy = []
        train_entropy = get_enotrpy_per_epoch(train_entropy,X_train, Y_train, alpha, beta)   
        
        
        test_entropy = []
        test_entropy = get_enotrpy_per_epoch(test_entropy,X_test, Y_test, alpha, beta)   
        
        
        train_cross_entropy[epoch_number] = sum(train_entropy)/len(train_entropy)
        test_cross_entropy[epoch_number] = sum(test_entropy)/len(test_entropy)

    return train_cross_entropy,test_cross_entropy,alpha,beta

def generate_report(train_cross_entropy,test_cross_entropy,train_labels,test_labels,train_error,test_error,training_output,testing_output, metrics_out, num_epochs):  
    
    output = ""
    for i in range(0, num_epochs):
            output += "epoch=" + str(i+1) + " crossentropy(train): " + str(train_cross_entropy[i][0]) + "\n"
            output += "epoch=" + str(i+1) + " crossentropy(test): " + str(test_cross_entropy[i][0]) + "\n"

    output += "error(train): " + str(train_error) + "\n"
    output += "error(test): " + str(test_error)

    with open(metrics_out, 'w') as outfile:
            outfile.writelines(output)
    
    with open(training_output,"w") as output:
        for item in train_labels:
            output.write(str(int(item[0])) + "\n")
            
    with open(testing_output,"w") as output:
        for item in test_labels:
            output.write(str(int(item[0])) + "\n")
             

def network_architecture(Y_train,X_train,Y_test,X_test ,metrics_out, num_epochs, hidden_units, initialization_flag, learning_rate, train_cross_entropy, test_cross_entropy,training_output,testing_output):
    
    alpha,beta = initialize_alpha_beta(initialization_flag, hidden_units, X_train)
    train_cross_entropy,test_cross_entropy,updated_alpha,updated_beta = train_model(alpha,beta, num_epochs, X_train, Y_train,X_test,Y_test, learning_rate, train_cross_entropy, test_cross_entropy)


    train_labels  = predict_labels(X_train,Y_train,updated_alpha, updated_beta)    
    test_labels   = predict_labels(X_test,Y_test,updated_alpha, updated_beta)

    #np.savetxt("test_labels.txt",test_labels)
    
    
    train_error = get_error(Y_train,train_labels)
    test_error = get_error(Y_test,test_labels)
    generate_report(train_cross_entropy,test_cross_entropy,train_labels,test_labels,train_error,test_error,training_output,testing_output, metrics_out, num_epochs)
            
def compare_labels(actual_values,predicted_values):
   
    length = len(actual_values)
    count = 0
    for i in range(length):
        if actual_values[i] != predicted_values[i]:
            count+=1
            
    error = count/len(predicted_values)
    return error
                        
def get_error(actual_labels,predicted_labels):
    
    predicted_values = []
    label_index_match = []
    for actual,predicted in zip(actual_labels,predicted_labels):
        actual = (np.argwhere(actual == 1)[0])
        label_index_match.append(actual)
        predicted_values.append((predicted[0]))
    
    return compare_labels(label_index_match,predicted_values)
            



if len(argv) == 10:  
    
    training_data = str(argv[1])
    testing_data = str(argv[2])
    training_output = str(argv[3])
    testing_output = str(argv[4])
    metrics_out = str(argv[5])
    num_epochs = int(argv[6])
    hidden_units = int(argv[7])
    initialization_flag = int(argv[8])
    learning_rate = float(argv[9])
    
    train_cross_entropy = np.zeros((num_epochs,1))
    test_cross_entropy = np.zeros((num_epochs,1))
    
    Train_labels, Train_features = process_Data(training_data)
    Test_labels , Test_features = process_Data(testing_data)
    network_architecture(Train_labels,Train_features,Test_labels,Test_features ,metrics_out, num_epochs, hidden_units, initialization_flag, learning_rate, train_cross_entropy, test_cross_entropy,training_output,testing_output) 
    
else:
    print("Incorrect number of arguments passed")



    
