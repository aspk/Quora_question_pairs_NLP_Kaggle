Quora_question_pairs_Kaggle
*****************************************************

Introduction
----------------
The goal of this competition is to predict which of the provided pairs of questions contain two questions with the same meaning. The ground truth is the set of labels that have been supplied by human experts. The ground truth labels are inherently subjective, as the true meaning of sentences can never be known with certainty. Human labeling is also a 'noisy' process, and reasonable people will disagree. As a result, the ground truth labels on this dataset should be taken to be 'informed' but not 100% accurate, and may include incorrect labeling. We believe the labels, on the whole, to represent a reasonable consensus, but this may often not be true on a case by case basis for individual items in the dataset.

source:https://www.kaggle.com/c/quora-question-pairs/data

Dependencies
---------------
Tensorflow, Python3, Xgboost, scikit-learn, numpy

Method
---------------
I used word_to_vec embeddings from Google, and scikit-learn and xgboost for training. A machine with 32 GB ram is prefered to run this code.

word_to_vec reference :
https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
