20-01-2016

#### Feature extraction ####

In the folder feature_extraction one can find the feature_extraction.py script. Note: in order to use this script succesfully, local paths have to be changed in the scripts to your own suitable paths. This script extract features (any combination of bof, histogram and/or gabor) from the images. The image_loader in the util folder is used to load the images reading the paths from the csv-file. Feature_extraction.py will output 2 files:

	- prefix_features.txt
	- prefix_classes.txt

Every line of this file corresponds to the same image. In feature-file the feature-vectors are saved, in the classes-file all the corresponding labels are saved.

All other scripts in this directory are used for feature extraction.

#### Balancing/reducing the dataset ####

In order to reduce and balance the dataset, one can use the dataset_selector.py in the util-directory. Given the features- and classes-files of the previous step, this script will select the images according to the parameters given in the scripts. Thus if it is given that 6 images will be used with 250 labels per images, dataset_selector.py will select these images and write these two new features- and classes-files.

#### Training and testing the SVM ####

In the experiments-folder the scripts experiments.py is used as framework for the SVM-experiments. Here one can choose any set-up and the SVM will be automatically trained and tested using stratifiedcv.py in the svm-directory. Outputs are written to matrix- and report-files. The confusion matrices can be printed in stratifiedcv.py. In order to use this script succesfully, local paths have to been changed in the script dataloader.py.

#### Classifying new images with the trained SVM ####

Our best models are saved in the svm-directory. With the script classify_new_image.py one can classify any new given image using our trained SVM-model.