#!/usr/bin/env python
# coding: utf-8

# In[52]:


# Importing Packages
import codecs
import os
import re
import time
import pandas as pd
import glob
import nltk
import sklearn.manifold
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('pylab', 'inline')


# In[53]:


#Books present

books = sorted(glob.glob("C:\\Users\\hp\\Downloads\\HarryPotter\\*.txt"))

print("Available Books:\n")
for i in books:
    print(i.split("\\")[-1].split("_")[0])


# In[54]:


# Read data from all books to single corpus variable
temp = ""
chars = []

for book in books:
    print("Reading " + str(book).split("\\")[-1].split("_")[0])
    with codecs.open(book, "r", "utf-8") as infile:
        temp += infile.read()
        chars.append(len(temp))
        print("Characters read so far: " + str(len(temp)))


# In[55]:


lens = []
lens.append(chars[0])
for i in range(1, len(chars)):
    lens.append(chars[i] - chars[i-1])
lens


# In[56]:


y = lens
N = len(y)
x = [i+1 for i in range(N)]
width = 1/1.5

pylab.xlabel("Book")
pylab.ylabel("Length")
plt.bar(x, y, width, color="red", align='center')


# In[57]:


# Split into sentences
sentences = nltk.tokenize.sent_tokenize(temp)
print("Total Sentences are " + str(len(sentences)))


# In[58]:


# sentences to list of words
sent_words = []
total_tokens = 0
for raw_sent in sentences:
    clean = nltk.word_tokenize(re.sub("[^a-zA-Z]"," ", raw_sent.strip().lower()))
    tokens = [i for i in clean if len(i) > 1]
    total_tokens += len(tokens)
    sent_words.append(tokens)

print("Total tokens are " + str(total_tokens))


# In[59]:


import random

# Train the Markov chain model
def train_markov_chain(words):
    markov_model = {}

    for i in range(len(words)-1):
        current_word = words[i]
        next_word = words[i+1]
        
        if current_word in markov_model:
            markov_model[current_word].append(next_word)
        else:
            markov_model[current_word] = [next_word]

    return markov_model


# In[60]:


# Generate text using the Markov chain model
def generate_text(markov_model, num_words=100, initial_word=None):
    if initial_word is None:
        initial_word = random.choice(list(markov_model.keys()))

    current_word = initial_word
    generated_text = current_word

    for _ in range(num_words-1):
        if current_word not in markov_model:
            break

        next_word = random.choice(markov_model[current_word])
        generated_text += " " + next_word
        current_word = next_word

    return generated_text


# In[61]:


# Flatten the list of lists into a single list
words = [word for sublist in sent_words for word in sublist]

# Train the Markov chain model
markov_chain_model = train_markov_chain(words)
generated_text = generate_text(markov_chain_model)
print(generated_text)


# In[62]:


import random

# Train the Markov chain model
def train_markov_chain_second(words):
    markov_model = {}
    order = 2

    for i in range(len(words) - order):
        current_state = tuple(words[i:i+order])
        next_word = words[i+order]

        if current_state in markov_model:
            markov_model[current_state].append(next_word)
        else:
            markov_model[current_state] = [next_word]

    return markov_model

# Generate text using the Markov chain model
def generate_text_second(markov_model, num_words=100, initial_state=None):
    if initial_state is None:
        initial_state = random.choice(list(markov_model.keys()))

    current_state = initial_state
    generated_text = ' '.join(current_state)

    for _ in range(num_words - len(initial_state)):
        if current_state not in markov_model:
            break

        next_word = random.choice(markov_model[current_state])
        generated_text += ' ' + next_word
        current_state = tuple(list(current_state[1:]) + [next_word])

    return generated_text

words = [word for sublist in sent_words for word in sublist]
markov_chain_model = train_markov_chain_second(words)
generated_text = generate_text_second(markov_chain_model)
print(generated_text)


# In[63]:


import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_similarity(text1, text2):
    # Create a CountVectorizer object
    vectorizer = CountVectorizer()

    # Fit and transform the text data
    text_matrix = vectorizer.fit_transform([text1, text2])

    # Calculate cosine similarity
    similarity_matrix = cosine_similarity(text_matrix)

    # Get the similarity score from the similarity matrix
    similarity_score = similarity_matrix[0, 1]

    return similarity_score


# In[64]:


books = sorted(glob.glob("C:\\Users\\hp\\Downloads\\ModelTest\\*.txt"))

print("Available Books:\n")
for i in books:
    print(i.split("\\")[-1].split("_")[0])


# In[65]:


test = ""
chars_test = []

for book in books:
    print("Reading " + str(book).split("\\")[-1].split("_")[0])
    with codecs.open(book, "r", "utf-8") as infile:
        test += infile.read()
        chars_test.append(len(test))
        print("Characters read so far: " + str(len(test)))


# In[66]:


books = sorted(glob.glob("C:\\Users\\hp\\Downloads\\Maths\\*.txt"))

print("Available Books:\n")
for i in books:
    print(i.split("\\")[-1].split("_")[0])


# In[67]:


test2 = ""
chars_test2 = []

for book in books:
    print("Reading " + str(book).split("\\")[-1].split("_")[0])
    with codecs.open(book, "r", "utf-8") as infile:
        test2 += infile.read()
        chars_test2.append(len(test2))
        print("Characters read so far: " + str(len(test2)))


# In[68]:


# Generate text using the Markov chain model based on the first 6 books
generated_text = generate_text_second(markov_chain_model, num_words=100)

# Text from the seventh part of Harry Potter
harry_potter_seventh_text = test

# Text from a random book
random_book_text = test2

# Calculate the similarity between the generated text and the text from the seventh part of Harry Potter
similarity_score_harry_potter = calculate_similarity(generated_text, harry_potter_seventh_text)

# Calculate the similarity between the generated text and the text from the random book
similarity_score_random_book = calculate_similarity(generated_text, random_book_text)

print(similarity_score_harry_potter)
print(similarity_score_random_book)
# Compare the similarity scores
if similarity_score_harry_potter > similarity_score_random_book:
    print("The seventh part of Harry Potter is more similar to the generated text.")
else:
    print("The random book is more similar to the generated text.")


# In[50]:


import random

# Train the Markov chain model
def train_markov_chain_zero(words):
    markov_model = {}
    
    for word in words:
        if word in markov_model:
            markov_model[word] += 1
        else:
            markov_model[word] = 1
    
    return markov_model

# Generate text using the Markov chain model
def generate_text_zero(markov_model, num_words=100):
    words = list(markov_model.keys())
    probabilities = list(markov_model.values())
    total_words = sum(probabilities)
    
    generated_text = ""
    
    for _ in range(num_words):
        random_index = random.choices(range(len(words)), weights=probabilities)[0]
        generated_word = words[random_index]
        generated_text += generated_word + " "
    
    return generated_text

# Example usage
words = [word for sublist in sent_words for word in sublist]
markov_chain_model = train_markov_chain_zero(words)
generated_text_zero = generate_text_zero(markov_chain_model)
print(generated_text)


# In[51]:


# Generate text using the Markov chain model based on the first 6 books
generated_text_zero = generate_text_zero(markov_chain_model, num_words=100)

# Text from the seventh part of Harry Potter
harry_potter_seventh_text = test

# Text from a random book
random_book_text = test2

# Calculate the similarity between the generated text and the text from the seventh part of Harry Potter
similarity_score_harry_potter = calculate_similarity(generated_text_zero, harry_potter_seventh_text)

# Calculate the similarity between the generated text and the text from the random book
similarity_score_random_book = calculate_similarity(generated_text_zero, random_book_text)

print(similarity_score_harry_potter)
print(similarity_score_random_book)
# Compare the similarity scores
if similarity_score_harry_potter > similarity_score_random_book:
    print("The seventh part of Harry Potter is more similar to the generated text.")
else:
    print("The random book is more similar to the generated text.")


# In[ ]:




