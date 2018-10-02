"""
trains CNN on the 28x28 upsampled (dynamic pooling matrix) matrix
CNN is trained from tutorial from Tensorflow Mnist
"""
import numpy as np
import pandas as pd
import tensorflow as tf

train_pool = np.load('train_pool.npy')
train_mat = np.load('train_matrix.npy')
np.random.shuffle(train_mat)
train_in, labels = np.transpose(train_mat)
# delete mask
pos = []
count = 0
delete_mask = np.ones((train_mat.shape[0]),bool)
for i in range(len(train_mat)):
    l1,l2 = train_mat[i][0].shape
    if l1 ==0 or l2==0 or l1==1 or l2==1 or l1==2 or l2==2:
        count+=1
        pos.append(i)
        delete_mask[i] = False
print(count)
train_in = train_in[delete_mask]
labels = labels[delete_mask]
#checking if deleting worked
count = 0
for i in range(len(train_in)):
    l1,l2 = train_in[i].shape
    if l1 ==0 or l2==0:
        count+=1
        pos.append(i)
    if l1 ==1 or l2==1:
        count+=1
        pos.append(i)
print(count)

# CNN code
npool = 28
mu = np.mean(train_pool)
sigma = np.std(train_pool)
validate_size = 100000
train_x = train_pool[:len(train_pool)-validate_size]
train_y = labels[:len(train_pool)-validate_size]
valid_x = train_pool[-validate_size:]
valid_y = labels[-validate_size:]
train_x = (train_x.reshape(len(train_x),npool**2)-mu)/sigma
valid_x = (valid_x.reshape(len(valid_x),npool**2)-mu)/sigma
train_y = np.eye(2)[train_y.astype(int)]
valid_y = np.eye(2)[valid_y.astype(int)]


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def train_conv(window=5,nlayer1=64,nlayer2=64,nlayer3=1024,batch_size=32,learning_rate = 0.1):
    npool = 28
    train_pool = np.load('train_pool.npy')
    train_mat = np.load('train_matrix.npy')
    np.random.shuffle(train_mat)
    train_in, labels = np.transpose(train_mat)

    mu = np.mean(train_pool)
    sigma = np.std(train_pool)

    validate_size = 100000
    train_x = train_pool[:len(train_pool)-validate_size]
    train_y = labels[:len(train_pool)-validate_size]
    valid_x = train_pool[-validate_size:]
    valid_y = labels[-validate_size:]
    train_x = (train_x.reshape(len(train_x),npool**2)-mu)/sigma
    valid_x = (valid_x.reshape(len(valid_x),npool**2)-mu)/sigma
    train_y = np.eye(2)[train_y.astype(int)]
    valid_y = np.eye(2)[valid_y.astype(int)]
    graph2 = tf.Graph()
    with graph2.as_default():
        x = tf.placeholder(tf.float32, [None, npool**2])
        y_ = tf.placeholder(tf.float32, [None, 2])
        W_conv1 = weight_variable([window, window, 1, nlayer1])
        b_conv1 = bias_variable([nlayer1])
        x_ = tf.reshape(x, [-1,npool,npool,1])
        h_conv1 = tf.nn.relu(conv2d(x_, W_conv1) + b_conv1)
        h_pool1 = -max_pool_2x2(-h_conv1)

        W_conv2 = weight_variable([window, window, nlayer1, nlayer2])
        b_conv2 = bias_variable([nlayer2])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = -max_pool_2x2(-h_conv2)

        W_fc1 = weight_variable([7 * 7 * nlayer2, nlayer3])
        b_fc1 = bias_variable([nlayer3])

        h_flat = tf.reshape(h_pool2, [-1, 7 * 7 * nlayer2])
        h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

        keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        W_fc2 = weight_variable([nlayer3, nlayer3])
        b_fc2 = bias_variable([nlayer3])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        W_fc3 = weight_variable([nlayer3, 2])
        b_fc3 = bias_variable([2])

        y_conv = tf.matmul(h_fc2, W_fc3) + b_fc3

        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        #train_step = tf.train.AdadeltaOptimizer().minimize(cross_entropy)
        #train_step = tf.train.RMSPropOptimizer(0.01).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        saver = tf.train.Saver()

        random_inds = np.random.choice(np.arange(len(valid_y)),5000)

    train_loss_arr = []
    valid_loss_arr = []
    iter_arr = []
    valid_acc_arr = []
    with tf.Session(graph = graph2) as sess:
        sess.run(tf.global_variables_initializer())
        num_epochs = 2
        epoch =0
        for epoch in range(num_epochs):
            i = 0
            loss = 0
            niter = len(train_x)//batch_size
            for _ in range(niter):
                batch = [train_x[i:i+batch_size],train_y[i:i+batch_size]]
                loss += cross_entropy.eval(feed_dict = {x:batch[0], y_: batch[1], keep_prob: 1})
                if _%100 == 0:
                    loss/=100
                    cross = cross_entropy.eval(feed_dict={
                    x:valid_x[random_inds], y_: valid_y[random_inds], keep_prob: 1.0})
                    print("step %d, training:validation loss is %g:%g"%(_, cross,loss))
                    loss = 0
                    validation_accuracy = accuracy.eval(feed_dict={x:valid_x[random_inds], y_: valid_y[random_inds], keep_prob: 1.0})
                    print('validation accuracy is {}'.format(validation_accuracy))
                    train_loss_arr.append([_+epoch*niter,loss])
                    valid_loss_arr.append([_+epoch*niter,cross])
                    valid_acc_arr.append([_+epoch*niter,validation_accuracy])
                train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                i+=batch_size
            print('validation accuracy')
            validation_accuracy = accuracy.eval(feed_dict={x:valid_x[random_inds], y_: valid_y[random_inds], keep_prob: 1.0})
            print("final step %d, validation accuracy %g"%(_, validation_accuracy))
            #y_predict = y_conv.eval(feed_dict={x:valid_x[:5000], keep_prob: 1.0})
            #y_true = valid_y[:5000]
            #loss = log_loss(y_true[:,1],y_predict[:,1])
            #print('log loss' + str(loss))

        # save model ...
        print("saving model")
        save_path = saver.save(sess, "dynamic_pooling_CNN.ckpt")
        print("Model saved in file: %s" % save_path)
        np.save('train_loss_arr.npy',train_loss_arr)
        np.save('valid_loss_arr.npy',valid_loss_arr)
        np.save('valid_acc_arr.npy',valid_acc_arr)

    return cross

print('training cnn with the best hyperparameters ...')
cross = train_conv(window = 5,nlayer1=32,nlayer2=64,batch_size=100)

"""
#grid search code

import tensorflow as tf
count = 0
nlayer1 = [16,32,64,128,256]
nlayer2 = [32,64,128,256,512]
window_split = [2,3,5]
a = np.ones((len(nlayer1),len(nlayer2)))
for i in range(len(nlayer1)):
    for j in range(len(nlayer2)):
        for k in range(len(window_split)):

            print(count)
            print('window_split '+ str(window_split[k]))
            print('layer1 '+ str(nlayer1[i]))
            print('layer2 '+ str(nlayer2[j]))
            count+=1
            a[i][j] = train_conv(window = window_split[k],nlayer1=nlayer1[i],nlayer2=nlayer2[j],batch_size=25)
np.save('grid_search.npy',a)

"""
