# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 15:29:24 2015

@author: carsten
"""
import re

def get_label_count(filename):
    f = open(filename, 'r')
    labels = {}
    line = f.readline()
    while line:
        line =  line[line.find("\t")+1:]
        for s in re.sub("\n","",line).split(','):
            val = 1
            if s in labels:
                val = val + labels[s]
            labels[s] = val
        line = f.readline()
    f.close()
    return labels          
    

    

def create_data_sets(labelfile, featurefile):
    labels = get_label_count(labelfile)
    for label, amount in labels.items():
        current_label_file = open("training_label_"+label+".txt", 'w')
        current_feature_file = open("training_feature_"+label+".txt", 'w')
        origin_label = open(labelfile, 'r')
        origin_feature = open(featurefile, 'r')
        number_training_image = int(0.9*amount)
        current_training_image = 0
        line = origin_label.readline()
        while line:
            value = -1
            path = line[:line.find("\t")]
            line = line[line.find("\t")+1:]
            if label in re.sub("\n","",line).split(','):
                if current_training_image == number_training_image:
                    current_label_file.close()
                    current_label_file = open("test_label_"+label+".txt", 'w')
                    current_feature_file.close()
                    current_feature_file = open("test_feature_"+label+".txt", 'w')
                value = 1
                current_training_image = current_training_image+1
            current_label_file.write(str(value)+"\t"+path+"\n")
            current_feature_file.write(origin_feature.readline())
            line = origin_label.readline()
           
        current_label_file.close()
        current_feature_file.close()
        origin_label.close()
        origin_feature.close()
        
if __name__ == '__main__':
    labelfile = 'C:\\Users\\Nadine\\git\\Project1\\classes.txt'
    featurefile = 'C:\\Users\\Nadine\\git\\Project1\\features.txt'
    create_data_sets(labelfile,featurefile)

