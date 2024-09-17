---
title: 'Diving Deep Into H2O.ai'
date: 2020-04-18
permalink: https://medium.com/@Msdhoni_07/diving-deep-into-h2o-ai-995921279889
tags:
  - cool posts
---
Diving-Deep-Into-H2O.ai
------
# Diving Deep Into H2O.ai

Artificial Intelligence, Deep Learning, Machine Learning whatever you are doing if you don't understand it, Learn it. Because otherwise, you are going to be a dinosaur within 3 years. - Mark Cuban
![](images/h2oai.gif)
After the rise of GPUs in the 21st century there has been quite significant progress in the world of deep learning, and everyone wants to learn machine learning and deep learning algorithms too. But most of them do not know how to get hands-on experience by implementing machine learning and deep learning models. This is exactly where H2O comes into the picture.

## What is H2O.ai

H2O.ai is the open source leader in AI and machine learning with a mission to democratize AI for everyone. H2O supports the most widely used machine learning algorithms and also has an AutoML functionality which is being used by hundreds of thousands of data scientists in over 20,000 organizations globally. H2O can easily and quickly derive insights from the data through faster and better predictive modeling.

![](images\deepLearning.png) 

## Deep Learning Overview

To begin with, let us have a brief overview of deep neural networks for supervised learning tasks. There are several theoretical frameworks for Deep Learning, but we would be focussing primarily on the Feed Forward architecture used by H2O.
![Feed Forward Neural Network Architecture](images\FFNN.jpg) 

This basic framework of multi-layer neural networks can be used to accomplish Deep Learning tasks. Deep Learning architectures are models of hierarchical feature extraction, typically involving multiple levels of nonlinearity. Deep Learning models can learn useful representations of raw data and have exhibited high performance on complex data such as images, speech, and text.

![](images\decap.png) 

## H2O's Architecture

H2O follows the model of multi-layer, feed-forward neural networks for predictive modeling. In the section, let us take a deep dive into H2O's Deep Learning architecture which provides a variety of options for predictive modeling using feed-forward neural networks by choosing deep learning features, parameter configurations, and computational implementation.

### Deep Learning functionalities in H2O flow framework:

- Supervised training protocol for regression and classification tasks.
- Fast and memory-efficient Java implementations based on columnar compression and fine-grain MapReduce.
- A multi-threaded and distributed parallel computation that can be run on a single or a multi-node cluster.
- Automatic, per-neuron adaptive learning rate for fast convergence.
- Optional specification of the learning rate, annealing and momentum options.
- Regularization options such as L1, L2, dropout, Hogwild and model averaging to prevent model overfitting.
- Elegant and intuitive web interface (Flow).
- Fully scriptable R API from H2O's CRAN package.
- Fully scriptable Python API.
- Grid search for hyperparameter optimization and model selection, automatic early stopping based on the convergence of user-specified metrics to a user-specified tolerance.
- Model checkpointing for reduced run times and model tuning, automatic pre and post-processing for categorical and numerical data.
- Automatic imputation of missing values (optional).
- Automatic tuning of communication vs computation for best performance.
- Model export in plain Java code for deployment in production environments.
- Additional expert parameters for model tuning.
- Deep autoencoders for unsupervised feature learning and anomaly detection.

Let us now apply some of the above-mentioned functionalities to build a deep learning model in the H2O flow framework to perform simple digit classification using the MNIST dataset.

## MNIST Digit Classification

![A survey of handwritten digits](images\handwritten_digits.png) 
Let's start with one of the well known basic deep learning examples of classifying digits using the MNIST dataset to demonstrate how H2O flow can be used to run deep learning models very efficiently. The data consists of 60,000 training images and 10,000 test images. Each image is a standardized 28x28 pixel grayscale image of a single handwritten digit.
Before that, you may have to set up H2O flow in our system. You can download the package from https://www.h2o.ai/download/#h2o and then run the following commands to install H2O dependencies in your system.

![](images\1.png) 

After that open up http://localhost:54321 in your browser and voila! You are there with one of the finest platforms to execute machine learning and deep learning models with ease.

### Loading MNIST Dataset

In this MNIST dataset example, we are using the publicly available training and testing datasets from a public Amazon S3 bucket (available at URLs given below).
Following are the commands to import test and train set in H2O Flow
Train Set:

![](images\2.png) 

Test Set:

![](images\3.png) 

### Parsing the Data
Once the MNIST dataset is imported, the data must be parsed according to the nature of the dataset which includes mentioning the type of parser, what type of separator to use and details about if the first row is a column or not before generating the data frame that is to be used for model building.

![](images\4.png) 

![MNIST train dataframe](images\5.png) 

### Building model

Now that we have our test.hex and train.hex (H2O's data frames) we have multiple options to explore like splitting the dataset into subsets, using AutoML to find out the best model or to build our model on top of this dataset.

![This illustrates how to set our parameter values for our specified model.](images\6.png) 

Here we can use the built-in H2O flow's 'Build a model' feature to build a model with our model parameters for any given dataset, in this case, our MNIST dataset. Below are some of the parameters I used:
Validation_frame list, select the parsed testing test.hex dataset as validation data.
Response list, select the last column (C785).
Hidden field, specify the hidden layer sizes (for this example, enter 128,64).
Epochs field, enter the number of training passes over the dataset (for this example, enter 500).
Activation list, select RectifierWithDropout.
Input_dropout_ratio, specify 0.2 and for Hidden_dropout_ratios, specify 0.3,0.2.
Stopping_metric to misclassification, Stopping_rounds to 3 and Stopping_tolerance to 1e-2 for early stopping based on the convergence of the misclassification rate

![Build a model script with the respective parameters](images\7.png) 

### Model Results:

After successfully building the model it is now time to review our model's performance. H2O.ai provides a variety of parameters to compare and contrast our model's output which is one of the most prominent features available in H2O flow. A budding data scientist can get valuable insights from these comparisons and results to build a better model.

### Log Loss:

For a general classification problem, we use Logarithmic loss as a measure to rate the model's performance. Logarithmic loss measures the performance of a classification model where the prediction input is a probability value between 0 and 1. The goal of our machine learning models is to minimize this value. A perfect model would have a log loss of 0.

![LogLoss scores for train and validation sets vs the number of epochs](images\8.png) 

### Variable Importances:

Variable importance is determined by calculating the relative influence of each variable- if that variable had a bigger impact in classifying the image during the model building process and how much the performance increased or decreased as its result.

![A bar graph showing features that impact the model](images\9.png) 

![Table with features that impact the model](images\10.png) 

### Confusion Matrix

A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. The confusion matrix shows how your classification model is confused when it makes predictions. It gives us insights not only into the errors being made by a classifier but more importantly the types of errors that are being made. Our model, without distortions, convolutions, or other advanced image processing techniques produces almost 0.97–0.99 precision scores on the validation set.

![Confusion matrix for the training set](images\11.png) 

![Confusion Matrix for the validation set](images\12.png) 

### Other Metrics:

There are various other useful metrics and comparisons available such as Status of neuron layers (which has information about layer number, units, type, dropout, L1, L2, mean rate, rate RMS, momentum, mean weight, weight RMS, mean bias, bias RMS), Top-K Hit Ratios (for multi-class classification), etc which can be used for optimizing the model you build.

![Status of neuron layers of the model created for MNIST dataset](images\13.png) 

![Top-10 Hit ratios for respective classes](images\14.png) 

## Model building using Auto ML

AutoML is a function in H2O that automates the process of building a large number of models, intending to find the "best" model without any prior knowledge or effort by the Data Scientist.
The current version of AutoML trains and cross-validates a default Random Forest, an Extremely-Randomized Forest, a random grid of Gradient Boosting Machines (GBMs), a random grid of Deep Neural Nets, a fixed grid of GLMs, and then trains two Stacked Ensemble models at the end. One ensemble contains all the models (optimized for model performance), and the second ensemble contains just the best performing model from each algorithm class/family (optimized for production use).
Here just to give a quick overview of how auto ML works, we are taking the MNIST dataset and will use just two algorithms (Deep learning and GBM) to compare their resulting models.

![AutoML parameters](images\15.png) 

![Choosing algorithms to exclude before AutoML run](images\16.png) 

The Auto ML trains 5 models for each algorithm (deep learning and GBM) and comes up with the best model under each algorithm. After this, the AutoML gives the comparative analysis for each of the models that have been trained from where the end-user can get valuable insights.

![The five model trains for GBM algorithm](images\17.png)

![LogLoss score for best GBM model](images\18.png)

## Conclusion

To summarize, we initially used H2O flow to build a deep learning model which was demonstrated using the MNIST dataset where we compared its results with the benchmark precision scores. We then explored the usage of the AutoML feature of H2O to compare models of various algorithms and to select the best model using the insights generated.


## References

[1]https://github.com/h2oai/h2o-tutorials

[2]https://www.h2o.ai/blog/the-different-flavors-of-automl/?gclid=CjwKCAjwp-X0BRAFEiwAheRuiwM--dObhCzSqNCXgYYSMslEEcfaDwgHFiQwvi8V72Dz3maL8ryf9RoChzQQAvD_BwE

[3]http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/deep-learning.html