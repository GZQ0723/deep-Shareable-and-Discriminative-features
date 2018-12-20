#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#final code for capturing discriminative features to be written here


# In[6]:


#import all file in this block
import gensim
from gensim.models import Word2Vec
from pprint import pprint
import math
import numpy as np


# In[ ]:


#declaration of global variables


# In[7]:


#readDataFiles : reads the file and return list of list of words, and label if necessary
#path_var: variable for path of file to be read, mode: type of data(test - 1 or trian - 0)
def readDataFiles(path_var, mode):
    data = []
    label = []
    with open(path_var , 'r') as file_obj:
        for line in file_obj:
            temp = line.split(',')
            data.append(temp[:3])
            if(1 != mode):
                label.append(temp[-1][0])
    if(1 != mode):
        return data, label
    else:
        return data


# In[8]:


#generateModel : generates the word2vec model for the data gathered
def generateModel(train_set, val_set, test_set):
    word_set = train_set + val_set + test_set
    word_model = gensim.models.Word2Vec(word_set, min_count = 1, size = 30, window = 5)
    return word_model


# In[ ]:


# main function
#word_model = None
def word_embedding(word_model):
    print('call')
#     global word_model
    train_path = "./training/train.txt"
    train_set, label_train = readDataFiles(train_path, 0)
    #print(train_set)

    for li in train_set:
        word1 = li[0]
        word2 = li[1]
        word3 = li[2]
        
        embd1 = word_model[word1]
        embd2 = word_model[word2]
        embd3 = word_model[word3]

        embd_list1 = embd1.tolist()
        embd_list2 = embd2.tolist()
        embd_list3 = embd3.tolist()
    
        embd_comb_list = embd_list1 + embd_list2 + embd_list3
        embd_comb_arr = np.array(embd_comb_list)
        print(len(embd_comb_list))
        print(embd_comb_arr)
    


# In[ ]:


def predicted_train(x, w):
    z = np.dot(x,w)
    g = 1/(1+np.exp(-z))
    
    return g


# In[ ]:


# get two weight vectors w0 for class 0 , w1 for class 1
def execute_gradient_train(x, y):
    itr = 0
    lr = 0.01 
    #w = np.zeros(x.shape[0])
    w = np.empty(x.shape[0])
    w.fill(1)
    #print(w)
    h = predicted_train(x,w)
    gradient = np.dot(x.T, (y - h))#Loss
    w = w - lr * gradient
        
            
    return w


#wout =  execute_gradient_train(model['Fulton'] , 1)


# In[ ]:


# for class 1 , w1 =m[w10 w11 w12 ...]
def test1(x_test, w1):
    z = np.dot(x_test,w1)
    g = 1/(1+np.exp(-z))
    
    return g

# for class 0 ,  w0 =m[w00 w01 w02 ...]
def test0(x_test, w0):
    z = np.dot(x_test,w0)
    g = 1/(1+np.exp(-z))
    
    return g
    


# In[43]:


def main():
    #global word_model
    #setting of path to take data
    train_path = "./training/train.txt"
    val_path = "./training/validation.txt"
    test_path = "./test/test_triples.txt"
    
    #reading data from the respective files
    train_set, label_train = readDataFiles(train_path, 0)
    val_set, label_val = readDataFiles(val_path, 0)
    test_set = readDataFiles(test_path, 1)
    
    #generating word2vec model for the data
    word_model = generateModel(train_set, val_set, test_set)
    #print(word_model['duck'])
    #print(word_model['goose'])
    print(word_model['sticky'])


    # assert word_model is not None
    word_embedding(word_model)


# In[44]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




