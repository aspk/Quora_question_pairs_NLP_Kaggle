# Duplicate question detection using Word2Vec, XGBoost and autoencoders

In this post, I tackle the problem of classifying questions pairs based on whether they are duplicate or not duplicate. This is important for companies like Quora, or Stack Overflow where multiple questions posted are duplicates of questions already answered. If a duplicate question is spotted by an algorithm, the user can be directed to it and reach the answer faster.

An example of two duplicate questions is 'How do I read and find my YouTube comments?' and 'How can I see all my Youtube comments?', and non duplicate questions is 'What's causing someone to be jealous?' and 'What can I do to avoid being jealous of someone?'.
Two approaches are applied to this problem:

1. Sequence Encoder trained by auto-encoder approach and dynamic pooling for classification
2. Bag of Words model with Logistic Regression and XGBoost classifier

Bag of words model with ngrams = 4 and min_df = 0 achieves an accuracy of 82 % with XGBoost as compared to 89.5% whicch is the best accuracies reported in literature with Bi LSTM and attention. The encoder approach implemented here achieves 63.8% accuracy, which is lower than the other approaches. I found it interesting because of the autoencoder implementation and the approach considers similary between phrases as well as the words for variable length sequences. Perhaphs, the efficiency could be improved by changing the dimentions of dynamically pooled matrix, a different approach in cleaning the data, as well as spelling checks. 

Classifier's can be compared based on three different evaluation metrics, log loss, auc, and accuracy. Log loss or the cross entropy loss is an indicator of how different the probability distribution of the output of the classifier is relative to the true probability distribution of the class labels.  Receiver operating characteristic plots the true positive rate vs the false positive rate and an area under the curve (auc) of 0.5 corresponds to a random classifier. Higher the AUC better the classifier. Accuracy is a simple metric, which calculates the fraction of correct predicted labels.

In this post, I use accuracy as a metric for comparison, as there is specific reason to do otherwise. 

### BOW model
![](https://i.imgur.com/2WqypTt.png)

![](https://i.imgur.com/pbB57pm.png)
As shown in the figure, as min_df is changed from 0 to 600 the accuracy decreases from 80% to 72 % for ngram = 4. min_df thresholds the ngrams appearing in the vocabulary according to count. Any ngram with frequency of appearance below min_df in the corpus is ignored. Ngrams beyond 4 are not used as there is a negligible change in accuracy as ngrams are increased from 3 to 4. Tf-idf vectorizer instead of Count vectorizer is used to speed up computation and it also increases the accuracy by a small amount (less than 1% for one data point). An accuracy of 82% is obtained by running the same input through XGBoost.

For the BOW model parameter sweep vocabulary size ranges from 703912 (n-grams = 4 and min_df =0) to 1018 (ngrams = 1 and min_df = 600).



### Auto-encoder and Dynamic Pooling CNN classifier

![](https://i.imgur.com/XLKpKab.png)

The figure above shows the implemented model, which is similar to Socher et al. Word2Vec embedding is generated with a vocabulary size of 100000 according to Tensorflow Word2Vec opensource release, using the skip gram model. In these embeddings, words which share similar context have smaller cosine distance. The key problem is dealing with questions of different lengths. The information content of a sentence is compressed by training an auto encoder. The main motivation behind this approach is to find similarity between sentences by comparing the entire sentence as well as the phrases in the sentence. The problem of different lengths is circumvented by upsampling and dynamic pooling as described below.

![](https://i.imgur.com/59WshFu.png)

Sentences are encoded using the approach shown in the left figure. The three words and the two encodings are considered as input to generate the similarity matrix. The auto-encoder is trained as shown in the right figure using Tensorflow. The right figure descibes the encoder decoder architecture. I used a single layer Neural Network for the encoder and the decoder, multiple hidden layers could also be considered. Multiple batches of words are concatenated and fed into the encoder and in the ideal case the output of the decoder should match the input. Mean squared error loss of the neural net is minimized with Gradient Descent optimizer with learning rate of 0.1. L2 regularization coefficient of 1e-4 is used for the encoder and decoder weights. 

The autoencoder here uses any two words for training and can be batch trained. It is different from the approach used by Socher et al., where the author encodes the entire sentence and decodes it by unfolding it into a question. Unfolding autoencoder is difficult or maybe even impossible to implement in Tensorflow. Dynamic computational graph construction tools like pytorch could potentially be a better fit to implment the full approach. 

The entire sentence with its intermediate encodings can be used as input to the upsampling and dynamic pooling phase. In the upsampling phase, the smaller vector of the question pair considered is upsampled by repeating the encodings randomly chosen of the vector to match the length of the other question encodings. A pairwise similarity matrix is generated for each phrase vector, and the variable dimention matrix is pooled into a matrix of npool x npool. I used npool = 28. This matrix is fed into a CNN classifier to classify as duplicate or not. A hyper parameter optimization of npool could also increase tha accuracy. The accuracy of this model is 63.8 % .

### Issues
I faced some issued with sklearn's Logistic regression. The model did output right class labels but wrong probabilities. I havent figured out a solution to this problem. There was no such problem with XGBoost.
### References

Best question pair matching method : Wang, Zhiguo, Wael Hamza, and Radu Florian. "Bilateral multi-perspective matching for natural language sentences." arXiv preprint arXiv:1702.03814 (2017).

Understanding Crossentropy loss and visualizing information: http://colah.github.io/posts/2015-09-Visual-Information/

Unfolding recursive autoencoder approach :Socher, Richard, et al. "Dynamic pooling and unfolding recursive autoencoders for paraphrase detection." Advances in neural information processing systems. 2011.

Word2Vec embeddings tensroflow opensource release:
https://github.com/tensorflow/tensorflow/blob/r1.9/tensorflow/examples/tutorials/word2vec/word2vec_basic.py
Tensorflow : https://www.tensorflow.org/


