#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 20:58:59 2019

@author: shekhartanwar
"""

import numpy as np
import operator
import math
from sys import argv
import time
import os




def process_lkv(lkv,store_dict,dictionary ,Threshold):
    
    for line in store_dict:
        labels, word_string =  line.strip().split('\t')
        lkv += labels        
        lkv = process_newdict(lkv,word_string.split(' '),dictionary, Threshold)
        lkv += "\n"
    return lkv

def read_write_Data(input_file, output_file, mode, lkv):
    if mode == "r":
        with open(input_file, "r") as infile:
            store_dict = infile.readlines()
        return store_dict
    if mode  == "w":
        with open(str(output_file), "w") as outfile:
            outfile.write(lkv)

def read_file(input_infile, outfile, dictionary, Threshold):
    
    store_dict = {}    
    store_dict = read_write_Data(input_infile, outfile, "r", "")
    
    lkv = ""
    
    lkv = process_lkv(lkv,store_dict,dictionary ,Threshold)    
    read_write_Data(input_infile, outfile, "w", lkv)
       
def process_newdict(lkv, word_list,dictionary, Threshold):
    new_dict = {}
    for word in word_list:
        
            try:
                if word in dictionary.keys():
                    word_index = dictionary[word]
                else:
                    continue
            except KeyError:
                continue
            if word_index in new_dict:
                new_dict[word_index] += 1
            else:
                new_dict[word_index] = 1
          
    return process_BAG(lkv, new_dict, Threshold)


def process_BAG(lkv, new_dict, Threshold):
    if Threshold != 0:
        new_dict = {k: 1 for k,v in new_dict.items() if v<Threshold}
    else:
        new_dict = {k: 1 for k,v in new_dict.items() }
         
    for key,value in new_dict.items():
        lkv += "\t" + str(key) + ":" + str(value)       
    
    return lkv


def handle_flag1(train_infile,valid_infile,test_infile,train_outfile,valid_outfile,test_outfile,processed_dict,Threshold):
        read_file(train_infile, train_outfile, processed_dict, Threshold)
        read_file(valid_infile, valid_outfile, processed_dict, Threshold)
        read_file(test_infile, test_outfile, processed_dict, Threshold)

def handle_flag2(train_infile,valid_infile,test_infile,train_outfile,valid_outfile,test_outfile,processed_dict,Threshold):        
        read_file(train_infile, train_outfile, processed_dict, Threshold)
        read_file(valid_infile, valid_outfile, processed_dict, Threshold)
        read_file(test_infile, test_outfile, processed_dict, Threshold)
    
    
def dict_file_processing(featureFlag,train_infile, train_outfile,valid_infile,valid_outfile,test_infile,test_outfile,processed_dict):
    if featureFlag == 1:
        handle_flag1(train_infile,valid_infile,test_infile,train_outfile,valid_outfile,test_outfile,processed_dict,0)
        

    if featureFlag == 2:
        handle_flag2(train_infile,valid_infile,test_infile,train_outfile,valid_outfile,test_outfile,processed_dict,4)


def process_features(train_infile,valid_infile,test_infile,dictionary,train_outfile,valid_outfile,test_outfile,featureFlag):
    processed_dict = {}

    with open(dictionary, "r") as infile:
        processed_dict = infile.readlines()
              
    processed_dict = dict(element.strip().split(' ') for element in processed_dict)
    
    dict_file_processing(featureFlag,train_infile, train_outfile,valid_infile,valid_outfile,test_infile,test_outfile,processed_dict)
    

train_infile = "smalltrain_data.tsv"
valid_infile = "smallvalid_data.tsv"
test_infile = "smalltest_data.tsv"
dictionary = "dict.txt"
train_outfile = "formatted_train_out_mine.tsv"
valid_outfile = "formatted_valid_out_mine.tsv"
test_outfile = "formatted_test_out_mine.tsv"
featureFlag  =  1
#    
process_features(train_infile,valid_infile,test_infile,dictionary,train_outfile,valid_outfile,test_outfile,featureFlag)      
##    


#
#
#if len(argv) == 9:  
#    
#    train_infile = str(argv[1])
#    valid_infile = str(argv[2])
#    test_infile = str(argv[3])
#    dictionary = str(argv[4])
#    train_outfile = str(argv[5])
#    valid_outfile = str(argv[6])
#    test_outfile = str(argv[7])
#    featureFlag  =  int(argv[8])
#    
#    process_features(train_infile,valid_infile,test_infile,dictionary,train_outfile,valid_outfile,test_outfile,featureFlag)      
#    
#else:
#    print("Incorrect number of arguments passed")
#
##   
