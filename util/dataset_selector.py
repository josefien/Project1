# -*- coding: utf-8 -*-

import re
import operator

def get_label_count(filename):
    f = open(filename, 'r')
    labels = {}
    line = f.readline()
    while line:
        line =  line[line.find("\t")+1:]
        classes = re.sub("\n|\r","",line).split(',')
        if len(classes) == 1:
            for s in re.sub("\n|\r","",line).split(','):
                val = 1
                if s in labels:
                    val = val + labels[s]
                labels[s] = val
        line = f.readline()
    f.close()
    return labels   
    
def define_training_data(src_features, src_classes, dest_features, dest_classes, num_labels, imgs_per_label):
    dest_feature_file = open(dest_features, 'w')
    dest_class_file = open(dest_classes, 'w')    
    
    labels = get_label_count(src_classes)
    labels = sorted(labels.items(), key=operator.itemgetter(1))
    label_counter = 0
    for label in reversed(labels):
        label_counter = label_counter +1
        if label_counter > num_labels:
            break
        actual_label = label[0].strip()
        img_counter = 0
        while img_counter < imgs_per_label:
            feature_src_file = open(src_features, 'r')
            label_src_file = open(src_classes, 'r')
            label_line = label_src_file.readline()
            feature_line = feature_src_file.readline()
            while label_line:
                src_labels =  label_line[label_line.find("\t")+1:]
                src_labels = re.sub("\n|\r","",src_labels).split(',')
                if len(src_labels) == 1 and src_labels[0] == actual_label:
                    dest_class_file.write(label_line)
                    dest_feature_file.write(feature_line)
                    img_counter = img_counter+1
                    if img_counter >= imgs_per_label:
                        break
                label_line = label_src_file.readline()
                feature_line = feature_src_file.readline()
            feature_src_file.close()
            label_src_file.close()
        
        
    dest_class_file.close()
    dest_feature_file.close()

#creates training set with 5 labels (with highest occurence) and 400 images per label (duplicates if not enough)
#filters only single labelled images
if __name__ == '__main__':
    define_training_data('../standard_features.txt','../standard_classes.txt','../dest_features.txt','../dest_classes.txt',5,400)