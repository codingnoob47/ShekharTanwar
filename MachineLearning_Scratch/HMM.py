#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 19:22:28 2019

@author: shekhartanwar
"""

import numpy as np
import sys

def process_lists(lists):
    lists = [word.strip() for word in lists]   
    list_length = len(lists)
    return lists,list_length

def process_each_element(inner_list,word_list, word_index,index_tag):
    for word in word_list:
        inner_list.append((word_index.index(word.split('_')[0]), index_tag.index(word.split('_')[1])))
    return inner_list
        
def get_index_tag_word(word_lines,word_index,tag_index):
    
    word_lines, word_length = process_lists(word_lines)
    word_index, index_word_length =  process_lists(word_index)
    tag_index, index_tag_length = process_lists(tag_index)
    
    outer_list = []
    
    for index in range(0,word_length):
        inner_list = []
        inner_list = process_each_element(inner_list,word_lines[index].split(' '), word_index,tag_index)
        outer_list.append(inner_list)
    
    data = []
    data.append(outer_list)
    data.append(index_word_length)
    data.append(index_tag_length)
    return data
    

def read_file(file_name, doc_list):

    with open(file_name,"r") as infile:
        doc_list = infile.readlines()
    return doc_list
        

def file_opeartions(train_input,index_to_word,index_to_tag):
    
    word_lines = []
    word_lines = read_file(train_input, word_lines)
    
    index_word = []
    index_word = read_file(index_to_word, index_word)
                    
    index_tag = []
    index_tag = read_file(index_to_tag, index_tag)
        
 
    return get_index_tag_word(word_lines,index_word,index_tag)    
  
def initialize_paramters(word_length,tag_length):
    
    rows = tag_length
    columns = word_length
    initialization_prob =  np.zeros(rows) #pi
    transition_prob = np.zeros((rows,rows))  #a
    emission_probability = np.zeros((rows,columns)) #b
    
    return initialization_prob,transition_prob,emission_probability

def pseudocount_update(lists,initialization_prob,transition_prob,emission_probability):
    # update the intialization prob matrix
    
    zero_pos = 0
    one_pos = 1
    for each_list in lists:
        index_init = each_list[zero_pos][one_pos]
        initialization_prob[index_init] = initialization_prob[index_init] + 1
    
       # since the counts needed to be updated by +1 and transition_prob is in sync with initialization_prob
        tags_list = [element[1] for element in each_list]
        for a,b in zip(tags_list,tags_list[1:]):
            transition_prob[a][b] = transition_prob[a][b] + 1
        
       # each element in the  emission_probability list is updated
        for each_element in each_list:
            emission_probability[each_element[one_pos]][each_element[zero_pos]] =emission_probability[each_element[one_pos]][each_element[zero_pos]] +  1
             
    return initialization_prob,transition_prob,emission_probability
    

def process_array(array):
    
    array = array/np.sum(array, axis = 1)[:,None]
    return array

def process_matrices(initialization_prob,transition_prob,emission_probability):
    
    initialization_prob = initialization_prob+1
    transition_prob = transition_prob+1
    emission_probability = emission_probability+1
    
    
    
    initialization_prob = initialization_prob/np.sum(initialization_prob)
    transition_prob = process_array(transition_prob)
    emission_probability = process_array(emission_probability)
    
    return initialization_prob,transition_prob,emission_probability
    

def learn_parameters(word_tag_index, word_length, tag_length,hmmprior,hmmemit,hmmtrans):
    

    initialization_prob,transition_prob,emission_probability = initialize_paramters(word_length,tag_length)     
    initialization_prob,transition_prob,emission_probability =  pseudocount_update(word_tag_index,initialization_prob,transition_prob,emission_probability)    
    initialization_prob,transition_prob,emission_probability =  process_matrices(initialization_prob,transition_prob,emission_probability)
    
    generate_report(hmmprior,hmmemit,hmmtrans,initialization_prob,transition_prob,emission_probability)
    
    
def generate_report(hmmprior,hmmemit,hmmtrans,initialization_prob,transition_prob,emission_probability):
        np.savetxt(hmmprior,initialization_prob)
        np.savetxt(hmmemit,emission_probability)
        np.savetxt(hmmtrans,transition_prob)
    
train_input = "trainwords.txt"
index_to_word = "index_to_word.txt"
index_to_tag  = "index_to_tag.txt"
hmmprior = "hmmprior_mine.txt"
hmmemit = "hmmemit_mine.txt"
hmmtrans = "hmmtrans_mine.txt"

data_processed = file_opeartions(train_input,index_to_word,index_to_tag)
word_tag_index, word_length, tag_length =  data_processed[0],data_processed[1],data_processed[2]
learn_parameters(word_tag_index, word_length, tag_length,hmmprior,hmmemit,hmmtrans)