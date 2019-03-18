# Big Data Case Study – Tensorflow and Neural Networks for Face Recognition
## Final Project for CSCI E-63: Big Data Analytics

**Purpose**: The goal of this project was to create a program package that demonstrated a specific big data case study, and design an analytical test/experiment that illustrates performance of the tools in question.

**Packages/Dependencies**: Python 3.5, Tensorflow, Scikit-Image, OpenCV3, SciPy, SciKit-Learn, h5py

**Dataset**: The dataset used is from the [“Labeled Faces in the Wild” (LFW)](http://vis-www.cs.umass.edu/lfw/) and my own dataset (in the repository) that I made from googling images.

**Summary**

In this analysis, two versions of the “Inception” CNN model is tested against two different approaches to facial recognition: the classifier and the triplet loss. The main training data was the “Labeled Faces in the Wild” (LFW). This dataset was manually broken into a larger training set and smaller validation testing set for demonstrative purposes (size = 48 images).

A type of Multi-task CNN was used to for pre-processing of images as it was more effective for images with distorted face angles/expressions.

The task of training a facial recognition model is a particularly lengthy and computationally expensive one since the datasets are often large and the CNN’s are very elaborate. My system at the time (Processor: Intel® Core™ i7-6500U CPU @ 2.50GHz 2.59 GHz) was used to train the data on the smaller dataset (LFW, ~13k images), and the training time was about 12+ hours per set.

The parameters were kept similar between running the classifier and triplet loss in order to compare performance. Overall, in smaller datasets and epochs, the triplet loss obtained better accuracy compared to the classifier approach, though at the cost of longer training time. This training time was significantly pronounced when training with larger datasets--there was an additional ~33% of computational time compared to the classifier model, which added up significantly as one can imagine when using a rather weak processor! Interestingly though, with larger datasets, the classifier outperformed the triplet loss approach in the long run. During the training of the larger dataset and in the earlier epochs, the triplet loss appeared to outperform the classifier in accuracy. However, in later epochs, it would appear that the triplet loss did not show further improvement (peaked around ~80%), whereas the classifier's accuracy continue to grow, peaking around ~90%. 

These results suggest to me that the triplet loss is quite effective at handling face recognition because of the principle behind triplets is unique to facial recognition, which is why it would appear to be better than the classifier with smaller datasets and seems to surpass in accuracy in the early training epochs. However, after so many epochs, the effectiveness of the triplets may be “capped” because there are only so many “good quality” triplets that can be generated from a dataset. On the other hand, classifier may be entirely dependent on the size of the dataset, and it merely performs better with the large dataset because it has so many examples to work with. Between the two Inception CNN's chosen (Inception ResNet V1 and V2), there were no major differences in accuracy between the two, apart from the fact that V2 took longer to run due the increased complexity of the CNN.

Overall, these results demonstrated that triplet loss is a pretty good approach to facial recognition training, but it most likely limited by the quality of the training data and how many "effective" triplets they can generate.

**To run**

Clone repository and run _master_run.py_ using Python 3.5
Further details are documented in comments within the script to guide the tweaking of parameters.

**References**
- [FaceNet: Tensorflow GitHub](https://github.com/davidsandberg/facenet) 
- [FaceNet: A Unified Embedding for Face Recognition and Clustering (2015)](https://arxiv.org/abs/1503.03832)
