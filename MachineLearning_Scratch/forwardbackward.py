#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed April 06 09:47:25 2019

@author: shekhartanwar
"""

import numpy as np
import sys
import learnhmm as dhmm
from sys import argv

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


def get_pred(index,alpha,beta):
    alpha_indexed= alpha[:,index]
    beta_indexed = beta[:,index]
    result = np.multiply(alpha_indexed,beta_indexed)
    result = np.argmax(result)  
    return result

matches = 0
def compare(predicted,actual):
    global matches
    if predicted == actual:
        matches += 1
    return matches

def get_accuracy_ll(matched_entries,total_predictions,log_likelihood):
    accuracy = float(matched_entries)/total_predictions
    avg_log_likelihood = sum(log_likelihood)/len(log_likelihood)
    
    return accuracy,avg_log_likelihood
        
def process_results(predictions,accuracy,log_likelihood):
    
    total_predictions = 0
    matched_entries = 0
    for x,y in enumerate(accuracy):
        total_predictions = total_predictions + y[1]
        matched_entries = matched_entries + y[0]
       
    accuracy,avg_ll = get_accuracy_ll(matched_entries,total_predictions,log_likelihood)
    
    return accuracy,avg_ll

def process_ab_t(index1,index2,index3,result,hmmemit,hmmtrans,tag_length,mode):
    result = 0.0
    if mode == "alpha":
        result = hmmemit[index1,index2] * sum([result[j,index3-1]*hmmtrans[j,index1] for j in range(0, tag_length)])
    if mode == "beta":
        result = sum([hmmtrans[index1,j]*hmmemit[j,index2]*result[j, index3+1] for j in range(0, tag_length)])        
    return result

# index1 k
# index2 x_0
    
def process_alpha_zero(index1,index2,hmmprior,hmmemit):
    result = hmmprior[index1] * hmmemit[index1,index2]
    return result

def get_initial_beta():
    return 1.0


def get_length_result(each_list,l):
    length = len(each_list)
    result = np.zeros((l, length))
    start_index = length-2
    end_index = -1
    jump_criterion = -1
    
    return (length,result,start_index,end_index,jump_criterion)
    

def get_hmm_result(hmm_prior_prob,q,hmm_emission,x_initial):
    z = hmm_prior_prob[q] * hmm_emission[q,x_initial]
    return z


def get_hmm_result_end1(hmm_emission,r,X_dash,result,z,hmm_transform,l):    
    result = hmm_emission[r,X_dash] * sum([result[p,z]*hmm_transform[p,r] for p in range(0, l)])
    return result


def get_hmm_result_end2(hmm_transform,q,hmm_emission,X_dash,result,z,l):
    result = sum([hmm_transform[q,p]*hmm_emission[p,X_dash]*result[p, z] for p in range(0, l)])
    return result
    
    
def calculate_alpha_beta(each_list,word_length,l,hmm_prior_prob,hmm_emission,hmm_transform,mode):
    length,result,start_index,end_index,jump_criterion = get_length_result(each_list,l)

    if mode == "alpha":        
        x_initial = each_list[0][0]
        for q in range(0, l):
            
            result[q,0] = get_hmm_result(hmm_prior_prob,q,hmm_emission,x_initial)

        for M in range(1, length):
            X_dash = each_list[M][0]
            z = M-1
            for r in range(0, l):
                result[r,M] = get_hmm_result_end1(hmm_emission,r,X_dash,result,z,hmm_transform,l)
                    
    if mode == "beta":
        for r in range(0, l):
            result[r, length-1] = get_initial_beta()

        for M in range(start_index, end_index, jump_criterion):
            z = M+1
            X_dash = each_list[z][0]
            for q in range(0, l):
                result[q,M] = get_hmm_result_end2(hmm_transform,q,hmm_emission,X_dash,result,z,l)
    return result

def get_acc_ll(matched,count,alpha):
    accuracy = (matched,count)
    ll = np.log(np.sum(alpha[:,count-1]))   
    return accuracy,ll

def getPAL(vector):
    
    each_list,alpha,beta,tag_length = vector[0],vector[1],vector[2],vector[3]
    predictions = []
    count = len(each_list)
    matched = 0

    for i in range(0,count):
        y_pred = get_pred(i,alpha,beta)
        if y_pred == each_list[i][1]:
            matched+=1
        predictions.append((each_list[i][0], y_pred))
    accuracy,ll = get_acc_ll(matched,count,alpha)
    
    vector_list = [predictions,accuracy,ll]
    return vector_list

  
def get_output(output_string,predictions,word_index,tag_index):
     for each_prediction in predictions:
         temporary_result = ""
         for each_element in each_prediction:
             temporary_result += word_index[each_element[0]].strip() + "_" + tag_index[each_element[1]].strip() + " "
         temporary_result = temporary_result.rstrip()
         output_string += temporary_result + "\n"
         
     return output_string
             
     
def get_word_lines(wl,index_to_word):
     with open(index_to_word) as infile:
        wl = infile.readlines()
     return wl

def get_index_tag(tl,index_to_tag):
    with open(index_to_tag) as infile:
        tl = infile.readlines()
    return tl
     
def generate_report(data_processed,processed_acc, processed_ll,metric_file,index_to_word,index_to_tag,predictions,predicted):
    metric_string = "Average Log-LikeLikhood: " + str(processed_ll) + "\n" + "Accuracy: " + str(processed_acc)
    with open(metric_file, 'w') as outfile:
        outfile.writelines(metric_string)
        
    wl = []
    wl = get_word_lines(wl,index_to_word)

    tl = []
    tl = get_index_tag(tl,index_to_tag)
    
    output_string = ""
    output_string = get_output(output_string,predictions,wl,tl)
    output_string = output_string.rstrip()
    
    with open(predicted, 'w') as outfile:
        outfile.writelines(output_string)
    
def compute_PAL(data_processed,predictions,accuracy,log_likelihood,hmmprior_m,hmmemit_m,hmmtrans_m):
    list_combinations, length_word, length_tag =  data_processed[0],data_processed[1],data_processed[2]
    

    for each_list in list_combinations:
        alpha = calculate_alpha_beta(each_list,length_word, length_tag,hmmprior_m,hmmemit_m,hmmtrans_m,"alpha")
        beta = calculate_alpha_beta(each_list,length_word, length_tag,hmmprior_m,hmmemit_m,hmmtrans_m,"beta")
        vector = [each_list,alpha,beta,length_tag]
        r = getPAL(vector)

        predictions.append(r[0])
        accuracy.append(r[1])
        log_likelihood.append(r[2])
    
    return predictions,accuracy,log_likelihood

def get_data(hmp, hmet,hmt):
    hmp_m = np.loadtxt(hmp)
    hmet_m = np.loadtxt(hmet)
    hmts_m =  np.loadtxt(hmt)
    return hmp_m,hmet_m,hmts_m
   

def process_forward_backward(data_processed,hmp, hmet,hmt,predicted,metric_file,index_to_word,index_to_tag):
    
    hmmprior_m,hmmemit_m,hmmtrans_m = get_data(hmp, hmet,hmt)
    
    predictions = []
    accuracy = []
    log_likelihood = []
    predictions,accuracy,log_likelihood = compute_PAL(data_processed,predictions,accuracy,log_likelihood,hmmprior_m,hmmemit_m,hmmtrans_m)

    processed_acc, processed_ll = process_results(predictions,accuracy,log_likelihood)
    generate_report(data_processed,processed_acc, processed_ll,metric_file,index_to_word,index_to_tag,predictions,predicted)

if len(argv) == 9:  
    
    t_input = sys.argv[1]
    i_to_w = sys.argv[2]
    i_to_t = sys.argv[3]
    h_p = sys.argv[4]
    h_e = sys.argv[5]
    h_t = sys.argv[6]
    pr = sys.argv[7]
    m_f = sys.argv[8]

    word_tag_index, word_length, tag_length = dhmm.file_opeartions(t_input,i_to_w,i_to_t)
    data_processed = [word_tag_index, word_length, tag_length]
    process_forward_backward(data_processed,h_p, h_e,h_t,pr,m_f,i_to_w,i_to_t)
else:
    print("Incorrect number of arguments passed")


