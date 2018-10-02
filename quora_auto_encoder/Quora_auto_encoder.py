"""
loads Word2Vec embeddings and trains autoencoder
"""
import tensorflow as tf
import numpy as np
import pandas as pd

# load embedding matrix, word to index dictionaries
# and data coprresponds to indices of words in the dictionary
# as they appear in the document
final_embeddings = np.load('final_embeddings.npy')
dictionary = np.load('dictionary.npy').item()
data = np.load('data.npy')
embedding_size = final_embeddings.shape[1]


auto_index = 0
def batch_autoencode(batch_size):
    global auto_index
    batch = np.zeros((batch_size,2*embedding_size),float)
    for i in range(batch_size):
        batch[i] = np.concatenate((final_embeddings[data[auto_index+i]],final_embeddings[data[auto_index+i+1]]))
    auto_index+=batch_size
    return batch
print('auto encoder train data ...')
train_data =  batch_autoencode(batch_size=2000000)
print(train_data.shape)
print('auto encoder valid data ...')
valid_data = batch_autoencode(batch_size=10000)
print(valid_data.shape)

#train_autoencoder
#batch generation for generating autoencoder
beta1 = 1e-4 # regularizer coefficient
beta2 = 1e-4 # regularizer coefficient
batch_auto = 32 # batch size
graph2 = tf.Graph()
with graph2.as_default():
    input_vec = tf.placeholder(tf.float32, shape=[None,2*embedding_size])
    We = tf.Variable(tf.random_uniform([2*embedding_size, embedding_size],-1,1))
    be = tf.Variable(tf.random_uniform([embedding_size],-1,1))
    p = tf.tanh(tf.matmul(input_vec,We)+be)
    Wd = tf.Variable(tf.random_uniform([embedding_size, 2*embedding_size],-1,1))
    bd = tf.Variable(tf.random_uniform([2*embedding_size],-1,1))
    p1 = tf.tanh(tf.matmul(p,Wd)+bd)
    loss_single = tf.squared_difference(p1,input_vec)
    regularizer1 = tf.nn.l2_loss(We)
    regularizer2 = tf.nn.l2_loss(Wd)
    #check1 = tf.reduce_mean(tf.reduce_sum(loss_single,1))
    loss = tf.reduce_mean(tf.reduce_sum(loss_single,1))+beta1*regularizer1+beta2*regularizer2
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    #optimizer = tf.train.AdadeltaOptimizer().minimize(loss)
    # Add variable initializer.
    init = tf.global_variables_initializer()

#Begin training.
num_steps = int(np.floor(len(train_data)/batch_auto))
i = 0
average_loss = 0
num_epoch = 5
with tf.Session(graph = graph2) as sess:
    init.run()
    print('training auto encoder ...')
    for epoch in range(num_epoch):
        print('epoch {}:{}'.format(epoch,num_epoch))
        i=0
        for step in range(num_steps):
            in_auto = train_data[i:i+batch_auto]
            i+=batch_auto
            feed_dict = {input_vec: in_auto}

            # We perform one update step by evaluating the optimizer op (including it
            # in the list of returned values for session.run()
            _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 10000 == 0:
                if step > 0:
                    average_loss /= 10000
                valid_loss = loss.eval(feed_dict = {input_vec: valid_data})

                # The average loss is an estimate of the loss over the last 2000 batches.
                print('Average loss at step ', step, ': ', average_loss)
                print('valid_loss ', step, ': ', valid_loss)
                average_loss = 0
        encoding_mat = We.eval()
        encoding_bias = be.eval()
        decoding_mat = Wd.eval()
        decoding_bias = bd.eval()

np.save('encoding_mat.npy',encoding_mat)
np.save('encoding_bias.npy',encoding_bias)
np.save('decoding_mat.npy',decoding_mat)
np.save('decoding_bias.npy',decoding_bias)
