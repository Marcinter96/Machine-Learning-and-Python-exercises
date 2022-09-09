#!/usr/bin/env python
# coding: utf-8

# ### Welcome to you first hands-on on word embeddings!!!
# 
# - In this hands-on you will be using pretrained word vectors from stanford nlp which you can find [here](https://nlp.stanford.edu/projects/glove/)
# - Follow the instruction provided for cell to write the code in each cell.
# - Before submit your notebook. Restart the kernel and run all the cell. Make sure that any cell shouldn't cause any error or problem.
# - Don't forget to run the last cell in the jupyter notebook, failing which your efforts will be invalid.
# - Don't delete any cell given in the notebook.
# 
# #### Each word vectors is of dimension 50
# #### You will be performing following operations:
#     - Load the pretrained vectors from the text file
#     - Write a function to find cosine similarity between two word vectors
#     - Write an function to find analogy analogy problems such as King : Queen :: Men : __?__

# ### Task1
# - A text file having the trained word vectors is provided for you as word2vec.txt in the same working directory.
# - Each line in the file is space seperated values where first value is the word and the remaing values are its vector representation.
# 
# ### Define a function get_word_vectors()
#     parameters: file_name  
#     returns: word_to_vec: dictionary with key as the word and the value is the corresponding word vectors as 1-d array each element of type float32.  

# In[2]:


import numpy as np

def get_word_vectors(file_name):
    ###Start code here
    word_to_vec = {}
    with open(file_name,'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embedding = np.array(split_line[1:], dtype=np.float64)
            word_to_vec[word] = embedding
    ###End code
    return word_to_vec


# ### Using the function you defined above read the word vectors from the file word_vectors.txt and assign it to variable word_to_vec
# 
# ### Expected output  (showing only first few values of vectors)
#    Father:  [ 0.095496   0.70418   -0.40777   -0.80844    1.256      0.77071 ...]  
#    mother:  [ 0.4336     1.0727    -0.6196    -0.80679    1.2519     1.3767 ....]  
# 
#    

# In[3]:


word_to_vec = get_word_vectors('word2vec.txt')
father = word_to_vec["father"]
mother = word_to_vec["mother"]
print("Father: ", father)
print("mother: ", mother)


# 
# ### Task 2 Determine the cosine similarity between two word vectors
# - The formula for cosine similarity is given by
#   score = $\large \frac{U.V}{\sqrt{||U||.||V||}}$ where ||U|| and ||V|| is the sum of the squares of the elemnts individual vectors
#   
# 
# ### Define a function named cosine_similarity()
#     - parameters u, v are the word vectors whose similarity has to be determined
#     - returns - score: cosine similarity of u and v

# In[4]:


def cosine_similarity(u, v):
    ###Start code here
    dot = np.dot(u,v)
    # Compute the L2 norm of u (≈1 line)
    norm_u = np.sqrt(np.sum(u * u))
    
    # Compute the L2 norm of v (≈1 line)
    norm_v = np.sqrt(np.sum(v * v))
    # Compute the cosine similarity defined by formula (1) (≈1 line)
    score = dot / (norm_u * norm_v)
    
    ###End code
    return score


# #### Run the bellow cell to find the similarity between word vectors paris and rome
# ### Expected output
#    similarity score : 0.7099411

# In[5]:


paris = word_to_vec["paris"]
rome = word_to_vec["rome"]
print("similarity score :", cosine_similarity(paris, rome))


# ### Task 3
# In the word analogy task, we complete the analogy . In detail, we are trying to find a word d, such that the associated word vectors $u_1, v_1, u_2, v_2$ are related in the following manner: $u_1 - v_1 \approx u_2 - v_2$. We will measure the similarity between $u_1 - v_1$ and $u_2 - v_2$ using cosine similarity.
# #### As an example,  to find the best possible word for the analogy King : Queen :: Men : __?_ you will perform following steps:
# - extract word vectors of three words king, queen and men
# - find the element wise difference between the two word vectors king and queen as V1
# - Find the element wise difference between the word vector men and each word vector in word_to_vec ditionary as V2 (while doing so exclude the words of interest ie. king, queen and men)
# - Find the cosine similarity between vector V1 and V2 and choose the word from the word_to_vec ditionary that has maximum similarity between V1 and V2.
# ### Define the function named find_analogy()
#     - parameters: word1 - string corresponding to word vector $u_1$, word2 - string corresponding to word vector $v_1$, word3 - string corresponding to word vector $u_2$, word_to_vec - dictionary of words and their corresponding vectors
#     - returns: best_word -  the word such that $u_1$ - $v_1$ is close to $v\_best\_word$ - $v_c$, as measured by cosine similarity
# 
# 

# In[16]:


def find_analogy(word_1, word_2, word_3, word_to_vec_map):
    ####Start code here
      # convert words to lower case
    word_a, word_b, word_c = word_1.lower(), word_2.lower(), word_3.lower()
    
    ### START CODE HERE ###
    # Get the word embeddings v_a, v_b and v_c (≈1-3 lines)
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    ### END CODE HERE ###
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    # loop over the whole word vector set
    for w in words:        
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_b, word_c] :
            continue
        
        ### START CODE HERE ###
        # Compute cosine similarity between the vector (e_b - e_a) and the vector ((w's vector representation) - e_c)  (≈1 line)
        cosine_sim = cosine_similarity(e_b - e_a, word_to_vec_map[w] - e_c)
        
        # If the cosine_sim is more than the max_cosine_sim seen so far,
            # then: set the new max_cosine_sim to the current cosine_sim and the best_word to the current word (≈3 lines)
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
        ### END CODE HERE ###
        
    return best_word


# ### Run the below  code to check your above defined function
# 
# #### Expected output:
#     father -> son :: mother -> daughter
#     india -> delhi :: japan -> tokyo

# In[17]:


print ('{} -> {} :: {} -> {}'.format('father', 'son', 'mother',find_analogy('father', 'son', 'mother', word_to_vec)))
print ('{} -> {} :: {} -> {}'.format('india', 'delhi', 'japan',find_analogy('india', 'delhi', 'japan', word_to_vec)))


# ### Run the below cells to save your answers

# In[18]:


from test_nlpusingdp_word2vec import word2vec as w2v
w2v.save_ans1(find_analogy, word_to_vec)


# In[ ]:




