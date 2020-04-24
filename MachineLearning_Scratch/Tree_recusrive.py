
import numpy as np
import operator
import math
from sys import argv
import time
max_IG = 0
class DecisionTreeClassifier:
    def __init__(self, attribute = None, split =  "Possible", vote = "NoVoteYet", value = "NoValueYet", left_branch = None, right_branch= None, depth = None):
        self.attribute = attribute
        self.split = split
        self.vote = vote
        self.value = value
        self.left_branch = left_branch
        self.right_branch = right_branch
        self.depth = depth
        
        
one_labels = ["democrat", "y", "A", "after1950","yes","morethan3min","fast","expensive","Two","large","high"]

def generateLeftSplit(data,column_number, value):
    left_data = []    
    for each_row in data:
        if each_row[column_number] == value:
            left_data.append(each_row)        
    return left_data

def generateRightSplit(data,column_number, value):

    right_data = []    
    for each_row in data:
        if each_row[column_number] != value:
            right_data.append(each_row)
            
    return right_data

def get_labels(data):
    labels = []
    for row in data:
        labels.append(row[-1])
    return labels


def get_entropy(labels):
    try:
        data_dict = {}
     
        for key in labels:
            data_dict[key] = labels.count(key)
    
        entropy = 0
        for key in data_dict:
            if data_dict[key] != 0:
                prob = float(data_dict[key]/ sum(data_dict.values()))
                entropy += - prob * math.log(prob,2)        
        return entropy
    except:
        print("Error in Entropy Calculation")
        

        
def getMajorityVote(labels):
    
    try:

        major_vote = {}

        for each_key in labels:
            major_vote[each_key] = labels.count(each_key)
    
        majority_vote = max(major_vote.items(), key=operator.itemgetter(1))[0]    
        return majority_vote
    except:
        print("Error in getting Majority Vote")
        
def  majorityVoteClassifier(data_values,data_labels):
    labels = get_labels(data_values) 
    maximum_vote,missclassification_error =  calculateErrorMajorityVote(labels)
    return labels,maximum_vote,missclassification_error
       
    
def calculateErrorMajorityVote(labels):
    error_dict = {}
    for value in labels:
        error_dict[value] = labels.count(value)     
    maximum_vote = max(error_dict.items(), key=operator.itemgetter(1))[0]
    maximum_values = max(error_dict.values())
    total_values = sum(error_dict.values())
    missclassification_error = float((total_values - maximum_values)/total_values)
    return maximum_vote,missclassification_error
  
    

def make_split(column_index, data ,value):
    left_split = []
    right_split = []
    attribute_values = []
    
    for row in data:
        if row[column_index] == value:
            left_split.append(row)
        else:
            right_split.append(row)
    
    
    for row in data:
        attribute_values.append(row[column_index])
       
    col_dict = {}
    for item in attribute_values: 
        col_dict[item] = attribute_values.count(item)
        
    return (col_dict, left_split, right_split)
    
    
    
def get_unique_values_list(each_column,data_values):
    
   
    return list(set([row[each_column] for row in data_values]))
        
    
    
    
def ID3(data_values,Attributes,depth,max_depth):
    global max_IG
    if len(data_values) == 0:
        return DecisionTreeClassifier()
    else:
        IG = 0
        max_IG = 0
        best_attribute = None
        best_attribute_value = None
        total_columns = len(Attributes) - 1
          
        for each_column in range(total_columns):           
                        
            for value in get_unique_values_list(each_column,data_values):
                col_dict, left_data,right_data = make_split(each_column, data_values,value)

                IG = getIG(col_dict,data_values, left_data, right_data)

                if (len(left_data) > 0) and (len(right_data) > 0):
                    if  IG > max_IG:
                        max_IG = IG
                        best_attribute = each_column
                        best_attribute_value = value
                        left_split = left_data
                        right_split = right_data
                       
        
        if depth < max_depth and max_IG> 0:
                left_tree = ID3(left_split,Attributes,depth+1,max_depth)
                right_tree = ID3(right_split,Attributes,depth+1,max_depth)
                return DecisionTreeClassifier(attribute = best_attribute,  value = best_attribute_value, left_branch = left_tree, right_branch= right_tree, depth = depth)
        else:
            y_labels = get_labels(data_values)
            vote_generated = getMajorityVote(y_labels)
            return DecisionTreeClassifier(split = y_labels, vote = vote_generated, depth = depth)

def getIG(col_dict,data, left_data, right_data):
    try:
        IG = 0
        y_labels = get_labels(data)
        root_entropy = get_entropy(y_labels)
        
        left_y_labels = get_labels(left_data)
        
        left_entropy = get_entropy(left_y_labels)
        
        right_y_labels = get_labels(right_data)
        
        right_entropy = get_entropy(right_y_labels)
          
        zero_count = col_dict.get(0)
        one_count = col_dict.get(1)
     
        if zero_count == None:
            zero_count = 0
                      

        if one_count == None:
             one_count = 0
         
        if (zero_count + one_count) == 0:
                IG = root_entropy
        else:
         
            prob_zero = float(zero_count/(zero_count + one_count))
            prob_one = 1- prob_zero
         
            IG = root_entropy - (prob_zero * left_entropy + prob_one * right_entropy)
        return IG
    except:
        print("Error in IG Calculation")

def get_data(filename):
    global one_labels
    try:
        with open(filename) as file_obj:
                data = file_obj.readlines()
            
        data = [x.strip("\r\n").strip("\n").replace(" ","") for x in data]
    
        attr_name = data[0].split(",")
        attr_val = [[ a for a in inst.split(",")] for inst in data[1:]]
        array = np.array([np.array(val) for val in attr_val])
    
        
    
        attr_val = [[ 1 if each_value in one_labels else 0 for each_value in values.split(",")] for values in data[1:]]
    
    # converted array
        
        return (attr_name,attr_val,array)
    except:
        print("Error Reading File")

               
def getLabelsError(decisionTree, data_value,data_labels):
    actual_labels = get_labels(data_value)
    attribute_data = []
    predicted_Labels = []
    for row in data_value:
        attribute_data = row[:-1]
        label_generated = PredictLabel(attribute_data,decisionTree)
        predicted_Labels.append(label_generated)
    
    error = calculateError(actual_labels,predicted_Labels)
    return (predicted_Labels, error)

def PredictLabel(row,decisionTree):
    if decisionTree.split != "Possible":
        return decisionTree.vote
    else:
        best_feature  = row[decisionTree.attribute]
        split_value  = decisionTree.value
        if best_feature!=split_value:
            return PredictLabel(row,decisionTree.right_branch)
        else:
            return PredictLabel(row,decisionTree.left_branch)
                
def calculateError(actual_labels,predicted_Labels):
    try:
        error = 0.0
        for i in range(len(actual_labels)):
            if actual_labels[i] != predicted_Labels[i]:
                error+=1
    
        error = float(error/len(actual_labels))
        return error
    except:
        print("Error in calculateError")
            
        
def get_unique(array):
    p = [value for value in array[:, array.shape[1]-1]]
    unique_list = [] 

    for x in p: 
        # check if exists in unique_list or not 
        if x not in unique_list: 
            unique_list.append(x) 
    return unique_list




def Master(train_data,test_data,train_labels, test_labels,max_depth,metrics):
    start = time.time()
    depth = 0
    if max_depth <0:
        print("Cannot build tree with negative depth")
    else:
        try:
            train_data_attributes, train_data_value,train_array = get_data(train_data)
            test_data_attributes, test_data_value,test_array = get_data(test_data)
            
            
           # print(train_data_attributes)

            unique_list = get_unique(train_array)
            test_unique = get_unique(test_array)
            
            train_value1 = unique_list[0]
            train_value2 = unique_list[1]
                
            test_value1 = test_unique[0]
            test_value2 = test_unique[1]
            
            
            
            for item in unique_list:
                if item in one_labels:
                    index_value_match = unique_list.index(item)
               
            

            
            print("Train Value1", train_value1, sep = " : ")
            print("Test Value1", test_value1, sep = " : ")
            
        except:
            print("Error in Data Generation")
        
        if max_depth == 0:
            train_major_labels,majority_vote,train_error = majorityVoteClassifier(train_data_value,train_labels)
            test_major_labels,majority_vote,test_error = majorityVoteClassifier(test_data_value,test_labels)
            
            

            
            
            train_labels_ = [unique_list[index_value_match] if x == 1 else unique_list[1 - index_value_match] for x in train_major_labels]

            
      
        
            test_labels_ =  [unique_list[index_value_match] if x == 1 else unique_list[1 - index_value_match] for x in test_major_labels]
            
            

                
        else:
            
            decisionTree = ID3(train_data_value,train_data_attributes,depth,max_depth)
            train_labels_generated, train_error = getLabelsError(decisionTree,train_data_value,train_labels)
            test_labels_generated, test_error = getLabelsError(decisionTree,test_data_value,test_labels)
                
            train_labels_ = [unique_list[index_value_match] if x == 1 else unique_list[1 - index_value_match] for x in train_labels_generated]
           
            test_labels_ =  [unique_list[index_value_match] if x == 1 else unique_list[1 - index_value_match] for x in test_labels_generated]  
       
        
  
                
        error_list = []
        error_list.append("error(train): " + str(train_error))
        error_list.append("error(test): " + str(test_error))
                
        file = open(train_labels,"w")
        
        for i in range(len(train_labels_)):
            file.write(str(train_labels_[i]) +  "\n")
    
        file.close()    
        
        
        file = open(test_labels,"w")
        
        for i in range(len(test_labels_)):
            file.write(str(test_labels_[i]) + "\n")
    
        file.close()    
        
        file = open(metrics,"w")
        
        for i in range(len(error_list)):
            file.write(error_list[i] + "\n")
    
        file.close()
        
        print(train_error,test_error, max_depth,sep = ":" )
        end = time.time()
        print("time" , end  -  start, sep = ": ")
        
        
if len(argv) == 7:  
    
    specified_depth = 1

    train_data = str(argv[1])
    test_data = str(argv[2])
    max_depth =  int(argv[3])
    train_labels = str(argv[4])
    test_labels  =  str(argv[5])
    metrics = str(argv[6])
    Master(train_data,test_data,train_labels,test_labels,max_depth,metrics)        
    
else:
    print("Incorrect number of arguments passed")
        
#
#filename = "education_train.csv"
#filename2 = "education_test.csv"
#specified_depth = 3
#
#train_data = filename
#test_data = filename2
#max_depth =  specified_depth
#train_labels = "e_e_train.labels"  
#test_labels  =  "e_test.labels"     
#metrics = "e_e_metrics.txt"
#Master(train_data,test_data,train_labels,test_labels,max_depth,metrics)        
