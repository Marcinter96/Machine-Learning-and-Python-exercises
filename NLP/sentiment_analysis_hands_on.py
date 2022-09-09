#!/usr/bin/env python
# coding: utf-8

# ### Welcome to the handson!!!
# - In this hands-on you will be building a sentiment classifier on for movie reviews using word vectors and LSTM.
# - Follow the instruction provided for cell to write the code in each cell.
# - Run the below cell for to import necessary packages to read and visualize data
# - Before submit your notebook. Restart the kernel and run all the cell. Make sure that any cell shouldn't cause any error or problem.
# - Don't forget to run the last cell in the jupyter notebook, failing which your efforts will be invalid.
# - Don't delete any cell given in the notebook.

# ### Import all the necessary packages in the below cell as and when you require

# In[1]:


from keras.datasets import imdb
from keras.datasets.imdb import get_word_index


# #### Downloading the dataset.
# - Keras has a built in function to download movie review available in imdb. 
# - Each words in the review are represented by their unique index and the labels are in binary format representing positive or negative reviews
# - The necessary code to download the dataset has been written for you.
# - The variable **word_to_id** is a dictionary containing words and their corresponding ids
# - Run the below cell to download the dataset

# In[2]:


vocab_size = 5000
(X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=vocab_size)
word_to_id = get_word_index()
print("word to id fist five samples {}".format({key:value for key, value in zip(list(word_to_id.keys())[:5], list(word_to_id.values())[:5])}))
print("\n")
print("sample input\n", X_train[0])
print('\n')
print("target output", Y_train[0])


# #### Each review in the dataset has some special tokens such as
#     - <START> : to identify the start of the sentence
#     - <UNK> : If some words are not identified in the vocabulary
#     - <PAD> : The value to be filled if sequence requires padding
# ### Task 1:
#     - offset the word_to_id dictionary by three values such that 0,1,2 represents START, UNK, PAD respectively
#     - Once you perform the above step reverse the word_to_id dictionary to represent ids as keys and words as values. Assign the resulting dictionary to id_to_word variable

# In[3]:


word_to_id = imdb.get_word_index()
word_to_id = {k:(v+3) for k,v in word_to_id.items()}
word_to_id["PAD"] = 0
word_to_id["START"] = 1
word_to_id["UNK"] = 2
id_to_word = {value:key for key,value in word_to_id.items()}


# ### Run the below code to view the first review in training samples
# 
# ### Expected output
# START this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert UNK is an amazing actor and now the same being director UNK father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for UNK and would recommend it to everyone to watch and the fly UNK was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also UNK to the two little UNK that played the UNK of norman and paul they were just brilliant children are often left out of the UNK list i think because the stars that play them all grown up are such a big UNK for the whole film but these children are amazing and should be UNK for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was UNK with us all

# In[4]:


print(" ".join([id_to_word[i] for i in X_train[0]]))


# #### Since each movie reviews are of variable lengths in terms of number of words, so it is necessay to fix the review lenght to few words say upto first 500 words.
# ### Task 3
#    - For each of the samples of X_train and X_test sample upto first 500 words
#    - If reviews are less than 500 words pad the sequence with zeros in the beginning to make up the length upto 500
#    - Assign the padded sequence to X_train_pad and X_test_pad variables for train and test smaples respectively
#    
#    [   0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    0    0    0    0    0    0    0    0    0    0    0    0
#     0    0    1   14   22   16   43  530  973 1622 1385   65  458 4468
#    66 3941    4  173   36  256    5   25  100   43  838  112   50  670
#     2    9   35  480  284    5  150    4  172  112  167    2  336  385
#    39    4  172 4536 1111   17  546   38   13  447    4  192   50   16
#     6  147 2025   19   14   22    4 1920 4613  469    4   22   71   87
#    12   16   43  530   38   76   15   13 1247    4   22   17  515   17
#    12   16  626   18    2    5   62  386   12    8  316    8  106    5
#     4 2223    2   16  480   66 3785   33    4  130   12   16   38  619
#     5   25  124   51   36  135   48   25 1415   33    6   22   12  215
#    28   77   52    5   14  407   16   82    2    8    4  107  117    2
#    15  256    4    2    7 3766    5  723   36   71   43  530  476   26
#   400  317   46    7    4    2 1029   13  104   88    4  381   15  297
#    98   32 2071   56   26  141    6  194    2   18    4  226   22   21
#   134  476   26  480    5  144   30    2   18   51   36   28  224   92
#    25  104    4  226   65   16   38 1334   88   12   16  283    5   16
#  4472  113  103   32   15   16    2   19  178   32]

# In[5]:


from keras.preprocessing import sequence 
max_review_length = 500
X_train_pad = sequence.pad_sequences(X_train,  maxlen=max_review_length)
X_test_pad =  sequence.pad_sequences(X_test,  maxlen=max_review_length)
print(X_train_pad[0])


# ### Using keras [Sequential](https://keras.io/getting-started/sequential-model-guide/) to build an LSTM model using the below specifications
# - Add an embedding layer( the look up table) such that vacabulary size is 5000 and each word in the vocabulary is 32 dimension vector
# - Add an LSTM layer with 100 hidden nodes
# - Add a final sigmoid activation layer
# - Use adam optimizer and binary cross entropy loss, and metrics as accuracy
# 
# ### Expected Output:
# 
# #### Layer (type)                 Output Shape              Param #   
# =================================================================  
# embedding_5 (Embedding)      (None, None, 32)          160000      
# _________________________________________________________________  
# lstm_5 (LSTM)                (None, 100)               53200     
# _________________________________________________________________  
# #### dense_5 (Dense)              (None, 1)                 101       
# =================================================================    
# Total params: 213,301   
# Trainable params: 213,301   
# Non-trainable params: 0  
# _________________________________________________________________

# In[7]:


from keras.layers import Embedding, LSTM, Dense
from keras import Sequential
embedding_vector_length = 32
model = Sequential() 
model.add(Embedding(5000, embedding_vector_length, input_length=None)) 
model.add(LSTM(100)) 
model.add(Dense(1, activation='sigmoid')) 
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy']) 
print(model.summary()) 


# ### Fit the model with X_train_pad and Y_train as train data and X_test_pad, Y_test as Validation set
#     - set the number of epochs to 3
#     - set batch size to 64

# In[8]:


### Start code here

model.fit(X_train_pad, Y_train, validation_data=(X_test_pad, Y_test), epochs=3, batch_size=64) 

###End code


# ### Run the below cell to run the model prediction on custom samples

# In[10]:


import numpy as np
bad = "this movie was terrible and bad"
good = "i really liked the movie and had fun"
for review in [good,bad]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=500) 
    print("%s. Sentiment: %s" % (review,model.predict(np.array([tmp_padded][0]))[0][0]))

accuracy = model.evaluate(X_test_pad[:1000], Y_test[:1000])[1]


# ### Run the below cells to save your answers
# 

# In[11]:


from test_nlpusingdp_sentimentanalysis import sentimentclassification
sentimentclassification.save_ans1(accuracy)


# In[ ]:




