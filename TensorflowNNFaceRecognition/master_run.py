###########################################################################
#            TensorFlow and Neural Networks for Face Recogniton           #
###########################################################################
#import os
import facenetmaster.src.align.align_dataset_mtcnn as alignmt
import facenetmaster.src.classifier as classifier
import facenetmaster.src.train_tripletloss as tripletlosstrain
import facenetmaster.src.train_softmax as train_softmax

###########################################################################
#       Parameters for the Alignment of Images of Faces Using MTCNN       #
###########################################################################

##NOTE THAT DIRECTORIES ARE RELATIVE TO THIS master_run.py FILE

##INPUT DIRECTORY: where the raw image files are located##
raw_data_path_train = "LFW-raw-train" # for training
raw_data_path_test = "LFW-raw-test" # for testing/validating

##OUTPUT DIRECTORY: where the processed/aligned images will go##
#As for which files to use for training/testing, it is totally up to you, there are no rules for this#
output_data_path_train = "LFW-train" # for training
output_data_path_test = "LFW-test" # for testing/validating

##IMAGE SIZE PARAMETER: for MTCNN, to tweak the size of processed image to input into the Facial Recognition training##
##Typically should be >= 160##
image_size = 160

##MARGIN PARAMETER##
margin = 32

###########################################################################
#                              Running MTCNN                              #
###########################################################################

user_prompt = input("Do you want to align images? (yes/no) ")
if user_prompt == "yes" or user_prompt == "y":
	input_string_train = [str(raw_data_path_train),str(output_data_path_train),"--image_size",str(image_size),"--margin",str(margin)]
	input_string_test = [str(raw_data_path_test),str(output_data_path_test),"--image_size",str(image_size),"--margin",str(margin)]
	alignmt.master_main(input_string_train)
	alignmt.master_main(input_string_test)

###########################################################################
#               Parameters for Running Classifier with SVN                #
###########################################################################
#Parameters for running the classifier/scikit-learn only
#You can learn more here: http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#And here for face recognition: http://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py

##NOTE THAT DIRECTORIES ARE RELATIVE TO THIS master_run.py FILE
#To run this, you need to have an already trained model in the model_location directory
#You can download such model from: https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk/edit
#Make sure to extract the files into a directory of your choice and update parameter below to point to the .pb file
model_location = "facenetmaster/src/models/20170512-110547/20170512-110547.pb"

#This is the name of the classifier file for the training run to write to
#The testing phase will call this file in order to test the data 
classifier_file = "facenetmaster/my_classifier.pkl" 

#Batch size variable
batch_size = 100

###########################################################################
#                   Running Classifier Training and Testing               #
###########################################################################
#This is a classifier run described in https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images
#I'm purposely switching the testing and training set

user_prompt = input("Do you want to run classifier/scikit-learn training? (yes/no) ")
if user_prompt == "yes" or user_prompt == "y":
	print(" ----------")
	print("| TRAINING |")
	print(" ----------")
	input_string_classifier = ["TRAIN", output_data_path_test, model_location, classifier_file, "--batch_size", str(batch_size)]
	classifier.master_main(input_string_classifier)
	print(" ----------")
	print("| TESTING  |")
	print(" ----------")
	input_string_classifier[0] = "CLASSIFY"
	input_string_classifier[1] = output_data_path_train
	classifier.master_main(input_string_classifier)

###########################################################################
#               Parameters for Running Triplet Loss with CNN              #
###########################################################################

##NOTE THAT DIRECTORIES ARE RELATIVE TO THIS master_run.py FILE

#Where to write the log files to
logs_base_dir = "facenetmaster/logs/facenet/"

#Where to write the model files to
model_base_dir = "facenetmaster/models/facenet/"

#The directory for training data
data_dir = "LFW-train"

#The directory for the testing data
lfw_dir = "LFW-test"

#The directory for the text file containing information on what pairs to test. Examples for format are in report.
lfw_pairs = "facenetmaster/data/pairs-mini.txt"

#Batch Size for the training part
batch_size = 90

#How to fold the pairs from the lfw_pairs file above; probably good idea to keep it a multiple of the total number of pairs in your file.
lfw_nrof_folds = 2

#Image size
image_size_trip = 160

#Optimizer name; the options are 'ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'
#Information on optimizers are on GitHub: https://github.com/davidsandberg/facenet
#Only RMSPROP was tested for this project
optimizer_name = "RMSPROP"

#Weight decay for the inference model
weight_decay = 1e-4

#Total epochs = max_epochs * epoch_size
#The program is set to run epoch_size times before perform a validation to test the current training model. Then it will cycle that max_epoch times.
#My recommendation for small datasets is to keep epochs small for triplet_loss because there is a finited number of triplets that can be generated
#If total epochs is too high, we run the risk of running out of triplets to test and this will cause an infinite loop
#For the small test dataset of 48 images, a total of 50 epochs is safe (you might be able to go to 100, too, even, but keep in mind this doubles training time)
#Of course, with huge datasets, you can go higher
max_epochs = 5
epoch_size = 10

#Designating the batch parameters
#Ony thing is you need to make sure people_per_batch * images_per_person is a multiple of 3
people_per_batch = 3 #3
images_per_person = 5 #5

#embedding size
embedding_size = 128

###########################################################################
#               Running Triplet Loss Training and Testing                 #
###########################################################################

user_prompt = input("Do you want to run triplet loss? (yes/no) ")
if user_prompt == "yes" or user_prompt == "y":
	#applying different Inception ResNet Models in a loop
	for model_file in ["facenetmaster.src.models.inception_resnet_v1", "facenetmaster.src.models.inception_resnet_v2"]:
		#applying different alpha values for the offset of the loss function
		for alpha in [0.90, 0.75]:
			#applying the learning_rate for the model
			for learning_rate in [0.1, 0.01, 0.001]:
				input_trip = ["--logs_base_dir", logs_base_dir, "--models_base_dir", model_base_dir, 
					"--data_dir", data_dir, "--image_size", str(image_size_trip), "--model_def", 
					model_file, "--lfw_dir", lfw_dir, "--lfw_pairs", lfw_pairs, "--optimizer",  optimizer_name, 
					"--learning_rate", str(learning_rate), "--weight_decay", str(weight_decay), "--max_nrof_epochs", 
					str(max_epochs), "--epoch_size", str(epoch_size), "--people_per_batch", str(people_per_batch), 
					"--images_per_person", str(images_per_person), "--alpha", str(alpha),
					"--batch_size", str(batch_size), "--lfw_nrof_folds", str(lfw_nrof_folds), "--embedding_size", str(embedding_size)
					]
	
				tripletlosstrain.master_main(input_trip)

###########################################################################
#                Parameters for Running Classifier with CNN               #
###########################################################################

##NOTE THAT DIRECTORIES ARE RELATIVE TO THIS master_run.py FILE

#The directory for the text file containing information on what pairs to test
lfw_pairs_s = "facenetmaster/data/pairs-mini.txt"

#How to fold the pairs from the lfw_pairs file and batch_size for the validation part; probably good idea to keep it a multiple of the total number of pairs
lfw_batch_size = 12
lfw_nrof_folds = 2

#The directory for training data
data_dir = "LFW-train"

#The directory for testing data
lfw_dir = "LFW-test"

#Batch size for the training part
batch_size_s = 90

#Optimizer name; the options are 'ADAGRAD', 'ADADELTA', 'ADAM', 'RMSPROP', 'MOM'
#Information on optimizers are on GitHub: https://github.com/davidsandberg/facenet
#Only RMSPROP was tested for this project
optimizer_name = "RMSPROP"

#Where to write the log files to
logs_base_dir_s = "facenetmaster/logs/facenet/"

#Where to write the model files to
model_base_dir_s = "facenetmaster/models/facenet/"

#Image size
image_size_soft = 160

#Total epochs = max_epochs * epoch_size
#The program is set to run epoch_size times before perform a validation to test the current training model. Then it will cycle max_epoch times
#I kept this similar to triplet loss for the sake of comparison
max_epochs_s = 5
epoch_size_s = 10

#Parameters for the inference model
keep_probability = 0.8
weight_decay = 1e-4


#Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"(http://ydwen.github.io/papers/WenECCV16.pdf)
center_loss_factor = 1e-2
center_loss_alfa = 0.9

#embedding size
embedding_size_s = 128

###########################################################################
#              Running Classifier/CNN Training and Testing                #
###########################################################################
user_prompt = input("Do you want to run classifier/CNN training? (yes/no) ")
if user_prompt == "yes" or user_prompt == "y":
	#for applying different models
	for model_file_s in ["facenetmaster.src.models.inception_resnet_v1", "facenetmaster.src.models.inception_resnet_v2"]:
		#adjust learning rate
		for learning_rate in [0.1, 0.01, 0.001]:
			input_class = ["--logs_base_dir", logs_base_dir_s, "--models_base_dir", model_base_dir_s,
				"--data_dir", data_dir, "--image_size", str(image_size_soft), "--model_def", 
				model_file_s, "--lfw_dir", lfw_dir, "--lfw_pairs", lfw_pairs_s, "--optimizer",  optimizer_name,
				"--max_nrof_epochs", str(max_epochs_s), "--keep_probability",
				str(keep_probability), "--random_crop", "--random_flip", "--learning_rate", str(learning_rate),
				"--weight_decay", str(weight_decay), "--epoch_size", str(epoch_size_s), "--center_loss_factor", str(center_loss_factor),
				"--center_loss_alfa",	str(center_loss_alfa), "--lfw_batch_size", str(lfw_batch_size), "--lfw_nrof_folds", str(lfw_nrof_folds),
				"--batch_size", str(batch_size_s), "--embedding_size", str(embedding_size_s)
				]
			train_softmax.master_main(input_class)
