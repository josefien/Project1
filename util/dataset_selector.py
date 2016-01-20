'''
this file is used to filter already extracted features for single labels and a certain 
amount of labels and images per label
'''
import re
import operator


'''
returns a dictionary of lables and the number of images in the given feature file
'''
def get_label_count(filename):
    f = open(filename, 'r')
    labels = {}
    line = f.readline()
    while line: #iterate all lines
        line =  line[line.find("\t")+1:] #extract only labels/classes
        classes = re.sub("\n|\r","",line).split(',')
        if len(classes) == 1:   #check if single labeled
            s = classes[0] #label for this line
            val = 1
            if s in labels:
                val = val + labels[s] #add 1 ti the number of images for this label
            labels[s] = val
        line = f.readline()
    f.close()
    return labels   

'''
filters the given feature/class files (src) for a certain amount of labels and images (ordered by frequency)
and writes the resulting files to dest-files
If imgs_per_label = 0, use frequecies as in the original set    
'''
def define_training_data(src_features, src_classes, dest_features, dest_classes, num_labels, imgs_per_label):
    #open files to write in
    dest_feature_file = open(dest_features, 'w')
    dest_class_file = open(dest_classes, 'w')    
    #load lables and sort them by frequency
    labels = get_label_count(src_classes)
    labels = sorted(labels.items(), key=operator.itemgetter(1))
    label_counter = 0
    #iterate in reverse order (label with highest occurence first)
    for label in reversed(labels):
        label_counter = label_counter +1
        #stop if number of labels is reached
        if label_counter > num_labels:
            break
        actual_label = label[0].strip()
        # if img_per_label = 0, use original frequency
        if imgs_per_label == 0:
            #open files to read from
            feature_src_file = open(src_features, 'r')
            label_src_file = open(src_classes, 'r')
            label_line = label_src_file.readline()
            feature_line = feature_src_file.readline()
            #traverse file entries
            while label_line:
                src_labels =  label_line[label_line.find("\t")+1:]
                src_labels = re.sub("\n|\r","",src_labels).split(',')
                #if single label and matches actual label, write to destination
                if len(src_labels) == 1 and src_labels[0] == actual_label:
                    dest_class_file.write(label_line)
                    dest_feature_file.write(feature_line)
                label_line = label_src_file.readline()
                feature_line = feature_src_file.readline()
            feature_src_file.close()
            label_src_file.close()

        if imgs_per_label > 0:
            img_counter = 0
            #traversing input file multiple times may be necessary to generate duplicates
            while img_counter < imgs_per_label:
                #open files to read from
                feature_src_file = open(src_features, 'r')
                label_src_file = open(src_classes, 'r')
                label_line = label_src_file.readline()
                feature_line = feature_src_file.readline()
                #traverse lines
                while label_line:
                    src_labels =  label_line[label_line.find("\t")+1:]
                    src_labels = re.sub("\n|\r","",src_labels).split(',')
                    #if single label and matches actual label, write to destination
                    if len(src_labels) == 1 and src_labels[0] == actual_label:
                        dest_class_file.write(label_line)
                        dest_feature_file.write(feature_line)
                        img_counter = img_counter+1
                        #if reached number of images, stop for this label
                        if img_counter >= imgs_per_label:
                            break
                    label_line = label_src_file.readline()
                    feature_line = feature_src_file.readline()
                feature_src_file.close()
                label_src_file.close()   
        
    dest_class_file.close()
    dest_feature_file.close()

#creates training set with 5 labels (with highest occurence) and 400 images per label (duplicates if not enough)
#filters only single labeled images
#total number of labels: 19
#path to feature and class files
path = 'C:\\Users\\Nadine\\Documents\\University\\Uni 2015\\RPMAI1\\features\\bof\\balanced_100\\'
if __name__ == '__main__':
    #datasets = ['standard','grabcut','hsl_grabcut','hsv_grabcut','rgb_grabcut']
    #prefix = datasets[4]
    datasets = ['standard']
    for prefix in datasets:
        feature_file = prefix+'_features.txt'
        label_file = prefix+'_classes.txt'

        from_dir = ''

        number_of_classes = 3
        occurences = 434

        to_dir = "%s_%d_%d" %('balanced',number_of_classes,occurences)

        f1 = path + from_dir + '/' + feature_file
        f2 = path + from_dir + '/' + label_file
        f3 = path + to_dir + '/' + feature_file
        f4 = path + to_dir + '/' + label_file
  
        define_training_data(f1,f2,f3,f4,number_of_classes,occurences)