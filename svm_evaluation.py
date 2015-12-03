# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 14:42:46 2015

@author: carsten
"""
import re

def svm_evaluation(filename):
    f = open(filename, 'r')
    right_labels = 0
    found_labels = 0
    right_found_labels = 0
    line = f.readline()
    while line:
        split = re.sub("\n","",line).split(' ')
        if int(split[0]) == 1:
            right_labels = right_labels+1
        if int(split[1]) == 1:
            found_labels = found_labels+1
            if int(split[0]) == 1:
                right_found_labels = right_found_labels+1
                
        line = f.readline()
    precision = float(right_found_labels)/float(found_labels)
    recall = float(right_found_labels)/float(right_labels)
    print recall
    print precision
        
svm_evaluation("result.txt")