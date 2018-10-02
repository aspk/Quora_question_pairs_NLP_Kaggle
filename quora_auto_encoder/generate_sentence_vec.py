"""
generate sentence from encoder matrix
"""
import numpy as np
import pandas as pd
import tensorflow as tf

# load embedding matrix, word to index dictionaries
# and data coprresponds to indices of words in the dictionary
# as they appear in the document

final_embeddings = np.load('final_embeddings.npy')
dictionary = np.load('dictionary.npy').item()
data = np.load('data.npy')
embedding_size = final_embeddings.shape[1]


# data
a = pd.read_csv('train.csv',delimiter = ',')
a = a.dropna()
q1 = a['question1'].values
q2 = a['question2'].values
specialstr = '?:!/;\|~`%1234567890.,%&$()_{}[]^"'

# load encoder auto_data
encoding_mat = np.load('encoding_mat.npy')
encoding_bias = np.load('encoding_bias.npy')


auto_data = []
def get_word(word):
    if word not in dictionary:
        return 'UNK'
    else:
        return word
def genparent(sentence):
    global auto_data
    n = len(sentence)
    if n>3:
        w1 = genparent(sentence[:int((n+1)/2) + 1])
        w2 = genparent(sentence[int((n+1)/2) + 1:])
        new_w = np.matmul(np.concatenate((w1,w2)),encoding_mat)+encoding_bias
        new_w = new_w/np.linalg.norm(new_w)
        auto_data.append(new_w)
        return new_w
    if n==3:
        w1 = final_embeddings[dictionary[get_word(sentence[0].lower())]]
        w2 = final_embeddings[dictionary[get_word(sentence[1].lower())]]
        w3 = final_embeddings[dictionary[get_word(sentence[2].lower())]]
        new_w = np.matmul(np.concatenate((w1,w2)),encoding_mat)+encoding_bias
        new_w2 = np.matmul(np.concatenate((new_w,w3)),encoding_mat)+encoding_bias
        new_w = new_w/np.linalg.norm(new_w)
        new_w2 = new_w2/np.linalg.norm(new_w2)
        auto_data.append(new_w)
        auto_data.append(new_w2)
        return new_w2
    if n==2:
        w1 = final_embeddings[dictionary[get_word(sentence[0].lower())]]
        w2 = final_embeddings[dictionary[get_word(sentence[1].lower())]]
        new_w = np.matmul(np.concatenate((w1,w2)),encoding_mat)+encoding_bias
        new_w = new_w/np.linalg.norm(new_w)
        auto_data.append(new_w)
        return new_w
    if n==1:
        w1 = final_embeddings[dictionary[get_word(temp_words[0].lower())]]
        return w1


print('create sentence vectors of word and phrases... ')
question1_data = []
question2_data = []
count = 0
parent_q1 = []
parent_q2 = []
for j in range(len(q1)):
    if j%50000==0:
        print(j/(len(q1)))
    auto_data = []
    line = q1[j]
    temp_words = ''.join( c for c in line if  c not in specialstr ).split()
    #adding the leaves
    #print(count)
    for i in temp_words:
        auto_data.append(final_embeddings[dictionary[get_word(i.lower())]])
    #adding parent
    genparent(temp_words)
    #parent_q1.append([auto_data,j])
    auto_data = np.array(auto_data)
    question1_data.append(auto_data)
    count+=1

question1_data =  np.array(question1_data)

for j in range(len(q2)):
    if j%50000==0:
        print(j/(len(q1)))
    auto_data = []
    line = q2[j]
    temp_words = ''.join( c for c in line if  c not in specialstr ).split()
    #adding the leaves
    for i in temp_words:
        auto_data.append(final_embeddings[dictionary[get_word(i.lower())]])
    #adding parent
    genparent(temp_words)
    #parent_q2.append([auto_data,j])
    auto_data = np.array(auto_data)
    question2_data.append(auto_data)

question2_data =  np.array(question2_data)

print('create train_mat with pairwise distances ...')

train_label = a['is_duplicate'].values
train_mat = []
for i in range(len(question1_data)):
    if i%20000==0:
        print(100*i/len(question1_data))
    active1 = question1_data[i]
    active2 = question2_data[i]
    dist_mat = np.zeros((len(active1),len(active2)),float)
    for j in range(len(active1)):
        for k in range(len(active2)):
            dist_mat[j][k] = np.linalg.norm(active1[j]-active2[k])
    train_mat.append(dist_mat)
train_mat = np.array(train_mat)
train_mat = np.transpose([train_mat,train_label])
np.save('train_matrix.npy',train_mat)
