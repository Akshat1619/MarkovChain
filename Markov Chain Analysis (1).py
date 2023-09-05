#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


#Books present

books = sorted(glob.glob("C:\\Users\\hp\\Downloads\\HarryPotter\\*.txt"))

print("Available Books:\n")
for i in books:
    print(i.split("\\")[-1].split("_")[0])


# In[3]:


# Read data from all books to single corpus variable
temp = ""
chars = []

for book in books:
    print("Reading " + str(book).split("\\")[-1].split("_")[0])
    with codecs.open(book, "r", "utf-8") as infile:
        temp += infile.read()
        chars.append(len(temp))
        print("Characters read so far: " + str(len(temp)))


# In[4]:


lens = []
lens.append(chars[0])
for i in range(1, len(chars)):
    lens.append(chars[i] - chars[i-1])
lens


# In[5]:


y = lens
N = len(y)
x = [i+1 for i in range(N)]
width = 1/1.5

pylab.xlabel("Book")
pylab.ylabel("Length")
plt.bar(x, y, width, color="red", align='center')


# In[6]:


# Split into sentences
sentences = nltk.tokenize.sent_tokenize(temp)
print("Total Sentences are " + str(len(sentences)))


# In[7]:


# sentences to list of words
sent_words = []
total_tokens = 0
for raw_sent in sentences:
    clean = nltk.word_tokenize(re.sub("[^a-zA-Z]"," ", raw_sent.strip().lower()))
    tokens = [i for i in clean if len(i) > 1]
    total_tokens += len(tokens)
    sent_words.append(tokens)

print("Total tokens are " + str(total_tokens))


# In[8]:


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


# In[9]:


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


# In[10]:


# Flatten the list of lists into a single list
words = [word for sublist in sent_words for word in sublist]

# Train the Markov chain model
markov_chain_model = train_markov_chain(words)
generated_text = generate_text(markov_chain_model)
print(generated_text)


# In[11]:


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


# In[12]:


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


# In[130]:


books = sorted(glob.glob("C:\\Users\\hp\\Downloads\\ModelTest\\*.txt"))

print("Available Books:\n")
for i in books:
    print(i.split("\\")[-1].split("_")[0])


# In[131]:


test = ""
chars_test = []

for book in books:
    print("Reading " + str(book).split("\\")[-1].split("_")[0])
    with codecs.open(book, "r", "utf-8") as infile:
        test += infile.read()
        chars_test.append(len(test))
        print("Characters read so far: " + str(len(test)))


# In[156]:


books = sorted(glob.glob("C:\\Users\\hp\\Downloads\\ModelCompare\\*.txt"))

print("Available Books:\n")
for i in books:
    print(i.split("\\")[-1].split("_")[0])


# In[157]:


test2 = ""
chars_test2 = []

for book in books:
    print("Reading " + str(book).split("\\")[-1].split("_")[0])
    with codecs.open(book, "r", "utf-8") as infile:
        test2 += infile.read()
        chars_test2.append(len(test2))
        print("Characters read so far: " + str(len(test2)))


# In[17]:


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


# In[18]:


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


# In[19]:


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


# In[20]:


def train_markov_chain(words):
    markov_model = {}

    for i in range(len(words)-1):
        current_word = words[i]
        next_word = words[i+1]

        if current_word in markov_model:
            if next_word in markov_model[current_word]:
                markov_model[current_word][next_word] += 1
            else:
                markov_model[current_word][next_word] = 1
        else:
            markov_model[current_word] = {next_word: 1}

    return markov_model


# In[34]:


from collections import defaultdict
import nltk

# Tokenize the sentences in the corpus
sentences = nltk.sent_tokenize(temp)
sent_words = [nltk.word_tokenize(re.sub("[^a-zA-Z]", " ", raw_sent.strip().lower())) for raw_sent in sentences]

# Create a dictionary to store word transitions
word_transitions = defaultdict(lambda: defaultdict(int))

# Iterate through sentences and words to populate the transition dictionary
for sentence in sent_words:
    for i in range(len(sentence) - 1):
        current_word = sentence[i]
        next_word = sentence[i + 1]
        word_transitions[current_word][next_word] += 1

# Create a dictionary to store transition probabilities
transition_matrix = {}

# Calculate probabilities for each word's transitions
for current_word, next_word_count in word_transitions.items():
    total_count = sum(next_word_count.values())
    probabilities = {next_word: count / total_count for next_word, count in next_word_count.items()}
    transition_matrix[current_word] = probabilities


# In[29]:


import matplotlib.pyplot as plt
from nltk.corpus import stopwords
# Sort the transition matrix for each current word by probability in descending order
sorted_transition_matrix = {current_word: {k: v for k, v in sorted(next_words.items(), key=lambda item: item[1], reverse=True)}
                            for current_word, next_words in transition_matrix.items()}

# Get the most occurring word with the highest probability for each current word
most_probable_words = {current_word: next_words_list[0]
                       for current_word, next_words_list in sorted_transition_matrix.items()}

# Extract the word and its probability
words = list(most_probable_words.keys())
probabilities = [transition_matrix[current_word][most_probable_words[current_word]] for current_word in words]

# Plot the bar chart
plt.figure(figsize=(12, 6))
plt.bar(words, probabilities)
plt.xlabel('Word')
plt.ylabel('Probability')
plt.title('Most Occurring Words with Highest Probability')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# In[36]:


# Importing necessary libraries
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

# Assuming you have already tokenized the text and stored it in a variable called 'words_list'
# For example, words_list = ["word1", "word2", "word1", "word3", ...]
# sentences to list of words
sent_words = []
total_tokens = 0
for raw_sent in sentences:
    clean = nltk.word_tokenize(re.sub("[^a-zA-Z]"," ", raw_sent.strip().lower()))
    tokens = [word for word in tokens if not word in stopwords.words('english')]
    tokens = [i for i in clean if len(i) > 1]    
    total_tokens += len(tokens)
    sent_words.append(tokens)

# Flatten the list of lists into a single list
words_list = [word for sublist in sent_words for word in sublist]

# Count the occurrences of each word
word_freq = nltk.FreqDist(words_list)

# Select the top N most occurring words
top_n = 30
most_common_words = word_freq.most_common(top_n)

# Extract words and their frequencies
words = [word for word, freq in most_common_words]
frequencies = [freq for word, freq in most_common_words]

# Plot the bar chart
plt.figure(figsize=(10, 6))
plt.bar(words, frequencies)
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Top {} Most Occurring Words'.format(top_n))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt

# Function to create a transition matrix from the Markov chain model
def create_transition_matrix(markov_model):
    words = list(markov_model.keys())
    transition_matrix = np.zeros((len(words), len(words)))

    for i, current_word in enumerate(words):
        next_words = markov_model[current_word]
        total_next_words = sum(next_words.values())
        
        for j, next_word in enumerate(words):
            if next_word in next_words:
                transition_matrix[i, j] = next_words[next_word] / total_next_words
    
    return words, transition_matrix

# Create the transition matrix
words, transition_matrix = create_transition_matrix(markov_chain_model)

# Presenting Findings and Transition Matrix
print("Insights and Patterns from Transition Matrix Analysis:")
print("-----------------------------------------------------")
print("The transition matrix reveals the probabilities of transitioning from one word to another.")
print("It highlights the patterns and relationships in how words follow each other in the text.")
print("By analyzing the transition matrix, we can observe the following:")

# Additional findings and insights can be added here based on your analysis

# Visualizing the Transition Matrix
plt.figure(figsize=(12, 10))
plt.imshow(transition_matrix, cmap='viridis', interpolation='nearest')
plt.title("Transition Matrix Visualization")
plt.xlabel("Next Word Index")
plt.ylabel("Current Word Index")
plt.colorbar(label="Transition Probability")
plt.xticks(np.arange(len(words)), words, rotation=90)
plt.yticks(np.arange(len(words)), words)
plt.tight_layout()
plt.show()


# In[41]:


from collections import Counter
from nltk.util import ngrams

n = 2  # Change to the desired n-gram length
ngram_freq = Counter(ngrams(words_list, n))

# Plot the most common n-grams
plt.figure(figsize=(10, 6))
plt.bar([str(ngram) for ngram, freq in ngram_freq.most_common(20)],
        [freq for ngram, freq in ngram_freq.most_common(20)])
plt.xlabel("{}-gram".format(n))
plt.ylabel("Frequency")
plt.title("Top {}-grams".format(n))
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[42]:


pip install textblob


# In[43]:


from textblob import TextBlob

# Assuming you have 'sentences' as a list of sentences
sentiment_scores = []

for sentence in sentences:
    # Create a TextBlob object
    blob = TextBlob(sentence)
    
    # Calculate the polarity of the sentence (ranges from -1 to 1)
    polarity = blob.sentiment.polarity
    
    # Append the sentiment score to the list
    sentiment_scores.append(polarity)

# Plot the sentiment scores
plt.figure(figsize=(10, 6))
plt.plot(sentiment_scores)
plt.xlabel("Sentence Index")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Analysis of Sentences")
plt.tight_layout()
plt.show()


# In[44]:


# Assuming you have sentiment scores for each sentence

plt.figure(figsize=(10, 6))
plt.plot(range(len(sentiment_scores)), sentiment_scores, color='green')
plt.xlabel("Sentence Index")
plt.ylabel("Sentiment Score")
plt.title("Sentiment Analysis")
plt.grid(True)
plt.show()


# In[45]:


# Sort words by frequency
sorted_word_freq = sorted(word_freq.items(), key=lambda item: item[1], reverse=True)

# Extract word ranks and frequencies
ranks = list(range(1, len(sorted_word_freq) + 1))
frequencies = [freq for word, freq in sorted_word_freq]

# Plot the scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(ranks, frequencies, s=10)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Word Rank (log scale)')
plt.ylabel('Word Frequency (log scale)')
plt.title('Word Frequency Distribution (Zipf\'s Law)')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[46]:


digraph g {
    node [shape=circle]
    A -> B
    B -> C
    C -> B
    C -> C
    B -> D
}


# In[47]:


pip install networkx


# In[48]:


import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from(['A', 'B', 'C', 'D'])

# Add edges
edges = [('A', 'B'), ('B', 'C'), ('C', 'B'), ('C', 'C'), ('B', 'D')]
G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_size=1000, font_size=12, node_color='skyblue', edge_color='gray')
plt.title("Directed Graph")
plt.show()


# In[50]:


pip install pygraphviz


# In[52]:


import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes
G.add_nodes_from(['A', 'B', 'C'])

# Add edges
G.add_edges_from([('A', 'B'), ('B', 'C'), ('C', 'A'), ('C', 'C')])

# Set layout using Circo layout
pos = nx.circular_layout(G)

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, font_color='black')

# Show the graph
plt.title('Directed Graph with Circo Layout')
plt.axis('off')
plt.show()


# In[54]:


import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add nodes with labels
G.add_node('R', label='Raleigh')
G.add_node('C', label='Chapel Hill')
G.add_node('D', label='Durham')

# Add labeled edges
G.add_edge('R', 'R', label='0.9')
G.add_edge('R', 'C', label='0.05')
G.add_edge('R', 'D', label='0.05')

G.add_edge('C', 'R', label='0.1')
G.add_edge('C', 'C', label='0.8')
G.add_edge('C', 'D', label='0.1')

G.add_edge('D', 'R', label='0.04')
G.add_edge('D', 'C', label='0.01')
G.add_edge('D', 'D', label='0.95')

# Define positions for the nodes
pos = {'R': (0, 1), 'C': (1, 0), 'D': (2, 1)}

# Draw the graph
nx.draw(G, pos, with_labels=True, node_size=1000, font_size=10, node_color='lightblue', arrows=True)

# Draw edge labels
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

plt.title('Directed Graph with Labeled Edges')
plt.show()


# In[62]:


import numpy as np
from scipy.sparse import lil_matrix

words = [word for sublist in sent_words for word in sublist]
word_indices = {word: i for i, word in enumerate(words)}

# Initialize a sparse transition matrix
num_words = len(words)
transition_matrix = lil_matrix((num_words, num_words), dtype=np.float64)

# Calculate transition counts
for i in range(len(words) - 1):
    current_word = words[i]
    next_word = words[i + 1]
    
    current_index = word_indices[current_word]
    next_index = word_indices[next_word]
    
    transition_matrix[current_index, next_index] += 1

# Convert to a normalized sparse matrix
transition_matrix = transition_matrix.tocsr()
row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix / row_sums

print("Transition Matrix:")
print(transition_matrix.toarray())  # Convert to a regular NumPy array for display


# In[56]:


word_indices


# In[57]:


words


# In[58]:


words_list


# In[63]:


import numpy as np
from scipy.sparse import lil_matrix

words = [word for sublist in sent_words for word in sublist]
word_indices = {word: i for i, word in enumerate(words)}

# Initialize a sparse transition matrix
num_words = len(words)
transition_matrix = lil_matrix((num_words, num_words), dtype=np.float64)

# Calculate transition counts
for i in range(len(words) - 1):
    current_word = words[i]
    next_word = words[i + 1]
    
    current_index = word_indices[current_word]
    next_index = word_indices[next_word]
    
    transition_matrix[current_index, next_index] += 1

# Convert to a normalized sparse matrix
transition_matrix = transition_matrix.tocsr()
row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix / row_sums

print("Transition Matrix:")
print(transition_matrix)


# In[64]:


import numpy as np
from scipy.sparse import lil_matrix

# Extract unique alphabets from all sentences
alphabets = set()
for sentence in sentences:
    for word in sentence:
        alphabets.update(set(word))

# Create a mapping of alphabets to indices
alphabet_indices = {alphabet: i for i, alphabet in enumerate(alphabets)}

# Initialize a sparse transition matrix
num_alphabets = len(alphabets)
transition_matrix = lil_matrix((num_alphabets, num_alphabets), dtype=np.float64)

# Calculate transition counts
for sentence in sentences:
    for i in range(len(sentence) - 1):
        current_word = sentence[i]
        next_word = sentence[i + 1]
        
        current_index = alphabet_indices[current_word]
        next_index = alphabet_indices[next_word]
        
        transition_matrix[current_index, next_index] += 1

# Convert to a normalized sparse matrix
transition_matrix = transition_matrix.tocsr()
row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix / row_sums

print("Transition Matrix:")
print(transition_matrix)


# In[143]:


sent_words


# In[75]:


lowercased_sentences = [sentence.lower() for sentence in sentences]
cleaned_sentences = []
for sentence in lowercased_sentences:
    cleaned_sentence = re.sub(r'[^\w\s]', '', sentence)  # Remove punctuation
    cleaned_sentence = re.sub(r'\d+', '', cleaned_sentence)  # Remove numbers
    cleaned_sentence = cleaned_sentence.replace('\n', '')
    cleaned_sentence = ' '.join(cleaned_sentence.split())
    cleaned_sentences.append(cleaned_sentence)
cleaned_sentences


# In[76]:


import numpy as np
from scipy.sparse import lil_matrix

# Extract unique alphabets from all sentences
alphabets = set()
for sentence in cleaned_sentences:
    for word in sentence:
        alphabets.update(set(word))

# Create a mapping of alphabets to indices
alphabet_indices = {alphabet: i for i, alphabet in enumerate(alphabets)}

# Initialize a sparse transition matrix
num_alphabets = len(alphabets)
transition_matrix = lil_matrix((num_alphabets, num_alphabets), dtype=np.float64)

# Calculate transition counts
for sentence in sentences:
    for i in range(len(sentence) - 1):
        current_word = sentence[i]
        next_word = sentence[i + 1]
        
        current_index = alphabet_indices[current_word]
        next_index = alphabet_indices[next_word]
        
        transition_matrix[current_index, next_index] += 1

# Convert to a normalized sparse matrix
transition_matrix = transition_matrix.tocsr()
row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix / row_sums

print("Transition Matrix:")
print(transition_matrix)


# In[74]:


alphabets


# In[353]:


import numpy as np
from scipy.sparse import lil_matrix

# Extract unique alphabets from all sentences
alphabets = set()
for sentence in cleaned_sentences:
    for word in sentence.split():  # Split sentence into words
        alphabets.update(set(word.lower()))  # Convert word to lowercase

# Include space as an alphabet
alphabets.add(' ')

# Create a mapping of alphabets to indices
alphabet_indices = {alphabet: i for i, alphabet in enumerate(alphabets)}

# Initialize a sparse transition matrix
num_alphabets = len(alphabets)
transition_matrix = lil_matrix((num_alphabets, num_alphabets), dtype=np.float64)

# Calculate transition counts
for sentence in cleaned_sentences:
    for word in sentence.split():  # Split sentence into words
        word = word.lower()  # Convert word to lowercase
        for i in range(len(word)):
            current_alphabet = word[i]
            current_index = alphabet_indices[current_alphabet]
            
            # Handle the case when the current alphabet is not the last one in the word
            if i < len(word) - 1:
                next_alphabet = word[i + 1]
                next_index = alphabet_indices[next_alphabet]
                transition_matrix[current_index, next_index] += 1
            
            # Handle the case when the current alphabet is followed by a space
            else:
                space_index = alphabet_indices[' ']
                transition_matrix[current_index, space_index] += 1

# Convert to a normalized sparse matrix
transition_matrix = transition_matrix.tocsr()
row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix / row_sums

print("Transition Matrix:")
for i in range(num_alphabets):
    for j in range(num_alphabets):
        print(transition_matrix[i, j], end="\t")
    print()  # Print a new line after each row


# In[363]:



alphabet_indices


# In[364]:


print("Transition Matrix Dimensions:", transition_matrix.shape)


# In[365]:


import numpy as np

# Define the alphabets (you should replace this with your actual set of alphabets)
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']

# Convert the matrix to a NumPy array
transition_array = np.array(transition_matrix)

# Print the transition matrix with exponential notation and corresponding alphabets
for i in range(transition_array.shape[0]):
    print("Transition probabilities from", alphabets[i])
    
    # Get the probabilities and corresponding alphabet indices
    probabilities = transition_array[i]
    sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
    
    for j in sorted_indices:
        prob = probabilities[j]
        if prob > 0:
            print(f"To {alphabets[j]}: {prob:.2e}")
    print()


# In[366]:


cleaned_sentences


# In[217]:


alphabet_indices


# In[367]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Define node labels
node_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']

# Create a directed graph
G = nx.DiGraph()

# Add nodes and edges
for i in range(len(node_labels)):
    for j in range(len(node_labels)):
        if transition_matrix[i, j] > 0:
            G.add_edge(node_labels[i], node_labels[j], weight=transition_matrix[i, j])

# Position nodes using spring layout
pos = nx.spring_layout(G)

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=500)

# Draw edges
edges = G.edges()
weights = [G[u][v]['weight'] * 10 for u, v in edges]
nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, edge_color='gray')

# Draw labels
nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

# Show the plot
plt.axis('off')
plt.title('Transition Probability Network Diagram')
plt.show()


# In[219]:


import pandas as pd

# Define row and column labels
row_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']
col_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']

# Create a DataFrame for the transition matrix
df = pd.DataFrame(transition_matrix, index=row_labels, columns=col_labels)

# Display the DataFrame as a table
print(df)


# In[99]:





# In[104]:


import subprocess

# Install the required packages using pip
subprocess.call(['pip', 'install', 'matplotlib', 'matplotlib-venn'])


import matplotlib.pyplot as plt
from matplotlib_venn import venn2_circles, venn2

# Sample data (replace with your data)
set1 = set(['a', 'b', 'c', 'd', 'e'])
set2 = set(['b', 'c', 'd', 'e'])

# Create a Venn diagram
venn = venn2([set1, set2], ('Set 1', 'Set 2'))

# Add annotations
venn.get_label_by_id('10').set_text('\n'.join(set1 - set2))
venn.get_label_by_id('01').set_text('\n'.join(set2 - set1))
venn.get_label_by_id('11').set_text('\n'.join(set1 & set2))

# Customize circle styles
venn2_circles(subsets=(len(set1), len(set2), len(set1 & set2)))

# Add title
plt.title("Chord Diagram")

# Display the diagram
plt.show()


# In[113]:


seventh_hp_book = None
random_book = None
for book_path in books:
    with open(book_path, 'r', encoding='utf-8') as file:
        text = file.read()
    if "harry potter and the deathly hallows" in text.lower():
        seventh_hp_book = text
    else:
        random_book = text

# Preprocess the texts


# In[110]:


print(random_book)


# In[123]:


sevenbook = "C:\\Users\\hp\\Downloads\\ModelTest\\Book7.txt"
randombook = "C:\\Users\\hp\\Downloads\\ModelTest\\maths.txt"


# In[220]:


sentencestest = nltk.tokenize.sent_tokenize(test)
print("Total Sentences are " + str(len(sentencestest)))


# In[221]:


sentencestest2 = nltk.tokenize.sent_tokenize(test2)
print("Total Sentences are " + str(len(sentencestest2)))


# In[222]:


sent_words_test = []
total_tokens_test = 0
for raw_sent in sentencestest:
    clean = nltk.word_tokenize(re.sub("[^a-zA-Z]"," ", raw_sent.strip().lower()))
    tokens_test = [i for i in clean if len(i) > 1]
    total_tokens_test += len(tokens_test)
    sent_words_test.append(tokens_test)

print("Total tokens are " + str(total_tokens_test))


# In[223]:


sent_words_test2 = []
total_tokens_test2 = 0
for raw_sent in sentencestest2:
    clean2 = nltk.word_tokenize(re.sub("[^a-zA-Z]"," ", raw_sent.strip().lower()))
    tokens_test2 = [i for i in clean2 if len(i) > 1]
    total_tokens_test2 += len(tokens_test2)
    sent_words_test2.append(tokens_test2)

print("Total tokens are " + str(total_tokens_test2))


# In[162]:


sent_words_test2


# In[224]:


import numpy as np
from scipy.sparse import lil_matrix

# Extract unique alphabets from all sentences
alphabets = set()
for sentence in cleaned_sentences:
    for word in sentence.split():  # Split sentence into words
        alphabets.update(set(word.lower()))  # Convert word to lowercase

# Include space as an alphabet
alphabets.add(' ')

# Create a mapping of alphabets to indices
alphabet_indices = {alphabet: i for i, alphabet in enumerate(alphabets)}

# Initialize a sparse transition matrix
num_alphabets = len(alphabets)
transition_matrix = lil_matrix((num_alphabets, num_alphabets), dtype=np.float64)

# Calculate transition counts
for sentence in cleaned_sentences:
    for word in sentence.split():  # Split sentence into words
        word = word.lower()  # Convert word to lowercase
        for i in range(len(word)):
            current_alphabet = word[i]
            current_index = alphabet_indices[current_alphabet]
            
            # Handle the case when the current alphabet is not the last one in the word
            if i < len(word) - 1:
                next_alphabet = word[i + 1]
                next_index = alphabet_indices[next_alphabet]
                transition_matrix[current_index, next_index] += 1
            
            # Handle the case when the current alphabet is followed by a space
            else:
                space_index = alphabet_indices[' ']
                transition_matrix[current_index, space_index] += 1

# Convert to a normalized sparse matrix
transition_matrix = transition_matrix.tocsr()
row_sums = transition_matrix.sum(axis=1)
transition_matrix = transition_matrix / row_sums

print("Transition Matrix:")
for i in range(num_alphabets):
    for j in range(num_alphabets):
        print(transition_matrix[i, j], end="\t")
    print()  # Print a new line after each row


# In[246]:


import numpy as np

# Define the alphabets (you should replace this with your actual set of alphabets)
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']

# Convert the matrix to a NumPy array
transition_array = np.array(transition_matrix)

# Print the transition matrix with exponential notation and corresponding alphabets
for i in range(transition_array.shape[0]):
    print("Transition probabilities from", alphabets[i])
    
    # Get the probabilities and corresponding alphabet indices
    probabilities = transition_array[i]
    sorted_indices = np.argsort(probabilities)[::-1]  # Sort in descending order
    
    for j in sorted_indices:
        prob = probabilities[j]
        if prob > 0:
            print(f"To {alphabets[j]}: {prob:.2e}")
    print()


# In[257]:


import numpy as np
from scipy.sparse import lil_matrix

# Extract unique alphabets from all sentences
alphabets = set()
for sentence in sent_words_test:
    for word in sentence:  # Split sentence into words
        alphabets.update(set(word.lower()))  # Convert word to lowercase

# Include space as an alphabet
alphabets.add(' ')

# Create a mapping of alphabets to indices
alphabet_indices_test = {alphabet: i for i, alphabet in enumerate(alphabets)}

# Initialize a sparse transition matrix
num_alphabets = len(alphabets)
transition_matrix_test = lil_matrix((num_alphabets, num_alphabets), dtype=np.float64)

# Calculate transition counts
for sentence in sent_words_test:
    for word in sentence:  # Split sentence into words
        word = word.lower()  # Convert word to lowercase
        for i in range(len(word)):
            current_alphabet = word[i]
            current_index = alphabet_indices[current_alphabet]
            
            # Handle the case when the current alphabet is not the last one in the word
            if i < len(word) - 1:
                next_alphabet = word[i + 1]
                next_index = alphabet_indices[next_alphabet]
                transition_matrix_test[current_index, next_index] += 1
            
            # Handle the case when the current alphabet is followed by a space
            else:
                space_index = alphabet_indices[' ']
                transition_matrix_test[current_index, space_index] += 1

# Convert to a normalized sparse matrix
transition_matrix_test = transition_matrix_test.tocsr()
row_sums = transition_matrix_test.sum(axis=1)
transition_matrix_test = transition_matrix_test / row_sums

print("Transition Matrix:")
for i in range(num_alphabets):
    for j in range(num_alphabets):
        print(transition_matrix_test[i, j], end="\t")
    print()  # Print a new line after each row


# In[166]:


import numpy as np

# Define the alphabets (you should replace this with your actual set of alphabets)
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Convert the matrix to a NumPy array
transition_array_test = np.array(transition_matrix_test)

# Print the transition matrix with exponential notation and corresponding alphabets
for i in range(transition_array_test.shape[0]):
    print("Transition probabilities from", alphabets[i])
    
    # Get the probabilities and corresponding alphabet indices
    probabilities_test = transition_array_test[i]
    sorted_indices_test = np.argsort(probabilities_test)[::-1]  # Sort in descending order
    
    for j in sorted_indices_test:
        prob = probabilities_test[j]
        if prob > 0:
            print(f"To {alphabets[j]}: {prob:.2e}")
    print()


# In[251]:


import numpy as np
from scipy.sparse import lil_matrix

# Extract unique alphabets from all sentences
alphabets = set()
for sentence in sent_words_test2:
    for word in sentence:  # Split sentence into words
        alphabets.update(set(word.lower()))  # Convert word to lowercase

alphabets.add(' ')
# Create a mapping of alphabets to indices
alphabet_indices_test2 = {alphabet: i for i, alphabet in enumerate(alphabets)}

# Initialize a sparse transition matrix
num_alphabets_test2 = len(alphabets)
transition_matrix_test2 = lil_matrix((num_alphabets_test2, num_alphabets_test2), dtype=np.float64)

# Calculate transition counts
for sentence in sent_words_test2:
    for word in sentence:  # Split sentence into words
        word = word.lower()  # Convert word to lowercase
        for i in range(len(word) - 1):
            current_alphabet_test2 = word[i]
            next_alphabet_test2 = word[i + 1]
            
            current_index_test2 = alphabet_indices_test2[current_alphabet_test2]
            next_index_test2 = alphabet_indices_test2[next_alphabet_test2]
            
            transition_matrix_test2[current_index_test2, next_index_test2] += 1

# Convert to a normalized sparse matrix
transition_matrix_test2 = transition_matrix_test2.tocsr()
row_sums_test2 = transition_matrix_test2.sum(axis=1)
transition_matrix_test2 = transition_matrix_test2 / row_sums_test2

print("Transition Matrix_test2:")
for i in range(num_alphabets_test2):
    for j in range(num_alphabets_test2):
        print(transition_matrix_test2[i, j], end="\t")
    print()  # Print a new line after each row


# In[252]:


import numpy as np

# Define the alphabets (you should replace this with your actual set of alphabets)
alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',' ']

# Convert the matrix to a NumPy array
transition_array_test2 = np.array(transition_matrix_test2)

# Print the transition matrix with exponential notation and corresponding alphabets
for i in range(transition_array_test2.shape[0]):
    print("Transition probabilities from", alphabets[i])
    
    # Get the probabilities and corresponding alphabet indices
    probabilities_test2 = transition_array_test2[i]
    sorted_indices_test2 = np.argsort(probabilities_test2)[::-1]  # Sort in descending order
    
    for j in sorted_indices_test2:
        prob = probabilities_test2[j]
        if prob > 0:
            print(f"To {alphabets[j]}: {prob:.2e}")
    print()


# In[169]:


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Assuming you have the transition matrices A, B, and C
# Replace these matrices with your actual data
matrix_A = transition_matrix
matrix_B = transition_matrix_test
matrix_C = transition_matrix_test2

# Flatten the matrices into 1D arrays
flat_matrix_A = matrix_A.flatten()
flat_matrix_B = matrix_B.flatten()
flat_matrix_C = matrix_C.flatten()

# Reshape the arrays to have a shape of (1, N) where N is the total number of elements
flat_matrix_A = flat_matrix_A.reshape(1, -1)
flat_matrix_B = flat_matrix_B.reshape(1, -1)
flat_matrix_C = flat_matrix_C.reshape(1, -1)

# Calculate cosine similarities
similarity_A_B = cosine_similarity(flat_matrix_A, flat_matrix_B)
similarity_A_C = cosine_similarity(flat_matrix_A, flat_matrix_C)

print("Cosine Similarity between Matrix A and Matrix B:", similarity_A_B[0][0])
print("Cosine Similarity between Matrix A and Matrix C:", similarity_A_C[0][0])

if similarity_A_B > similarity_A_C:
    print("Matrix B is more similar to Matrix A.")
else:
    print("Matrix C is more similar to Matrix A.")


# In[ ]:





# In[176]:


# Read the text content from the 7th part of Harry Potter
with open('C:\\Users\\hp\\Downloads\\ModelTest\\Book7.txt', 'r', encoding='utf-8') as file:
    text_content = file.read()
    
with open('C:\\Users\\hp\\Downloads\\Maths\\maths.txt', 'r', encoding='utf-8') as file:
    text_content1 = file.read()

# Filter out non-alphabetic characters and convert to lowercase
alphabets_harry_potter_7th = ''.join(filter(str.isalpha, text_content)).lower()
alphabets_random = ''.join(filter(str.isalpha, text_content1)).lower()

# Now you have the sequence of alphabets from the 7th part of Harry Potter
sequence_harry_potter_7th = alphabets_harry_potter_7th
sequence_random_book = alphabets_random


# In[182]:


import numpy as np

# Given transition matrix from the first six books and sequences of alphabets
transition_matrix_first_six = np.array(transition_matrix)  # Convert to NumPy array
sequence_harry_potter_7th = "sequence_from_7th_part"  # Replace with the 7th part of Harry Potter sequence
sequence_random_book = "sequence_from_random_book"  # Replace with the sequence from the random book

# Tokenize sequences into individual alphabets
alphabets_harry_potter_7th = list(sequence_harry_potter_7th)
alphabets_random_book = list(sequence_random_book)

# Print the types of the alphabets sequences
print("Type of alphabets_harry_potter_7th:", type(alphabets_harry_potter_7th))
print("Type of alphabets_random_book:", type(alphabets_random_book))

# Calculate log likelihood for a given transition matrix and sequence
def calculate_log_likelihood(transition_matrix, alphabets):
    log_likelihood = 0
    for i in range(len(alphabets) - 1):
        from_alphabet = alphabets[i]
        to_alphabet = alphabets[i + 1]
        from_index = ord(from_alphabet) - ord('a')  # Convert alphabet to index
        to_index = ord(to_alphabet) - ord('a')
        transition_prob = transition_matrix[from_index, to_index]
        log_likelihood += np.log(transition_prob)
    return log_likelihood

# Calculate log likelihoods for both sequences
log_likelihood_harry_potter_7th = calculate_log_likelihood(transition_matrix_first_six, alphabets_harry_potter_7th)
log_likelihood_random_book = calculate_log_likelihood(transition_matrix_first_six, alphabets_random_book)

# Compare log likelihoods
if log_likelihood_harry_potter_7th > log_likelihood_random_book:
    print("The 7th part of Harry Potter is a better fit to the transition matrix.")
else:
    print("The random book is a better fit to the transition matrix.")


# In[181]:


# Assuming you have already tokenized sentences in sent_words_test
sequence_harry_potter_7th = []

for sentence_words in sent_words_test:
    sequence_harry_potter_7th.extend(sentence_words)

# Now sequence_harry_potter_7th contains the sequence of words from your sentences
print(sequence_harry_potter_7th)


# In[188]:


import numpy as np

# Given transition matrix from the first six books and sequences of alphabets
transition_matrix_first_six = np.array(transition_matrix)  # Convert to NumPy array
sequence_harry_potter_7th = cleaned_test  # Replace with the 7th part of Harry Potter sequence
sequence_random_book = cleaned_test  # Replace with the sequence from the random book

# Calculate log likelihood for a given transition matrix and sequence
def calculate_log_likelihood(transition_matrix, alphabets):
    log_likelihood = 0
    for i in range(len(alphabets) - 1):
        from_alphabet = alphabets[i]
        to_alphabet = alphabets[i + 1]
        from_index = ord(from_alphabet) - ord('a')  # Convert alphabet to index
        to_index = ord(to_alphabet) - ord('a')
        transition_prob = transition_matrix[from_index, to_index]
        log_likelihood += np.log(transition_prob)
    return log_likelihood

# Calculate log likelihoods for both sequences
log_likelihood_harry_potter_7th = calculate_log_likelihood(transition_matrix_first_six, sequence_harry_potter_7th)
log_likelihood_random_book = calculate_log_likelihood(transition_matrix_first_six, sequence_random_book)

# Compare log likelihoods
if log_likelihood_harry_potter_7th > log_likelihood_random_book:
    print("The 7th part of Harry Potter aligns more closely with the transition matrix.")
else:
    print("The random book aligns more closely with the transition matrix.")


# In[359]:


test
cleaned_test = re.sub(r'[^\w\s]', '', test)  # Remove punctuation
cleaned_test = re.sub(r'\d+', '', cleaned_test)  # Remove numbers
cleaned_test = cleaned_test.replace('\n', '')
cleaned_test = ' '.join(cleaned_test.split())
cleaned_test=cleaned_test.lower()
cleaned_test


# In[239]:


test2
cleaned_test2 = re.sub(r'[^\w\s]', '', test2)  # Remove punctuation
cleaned_test2 = re.sub(r'\d+', '', cleaned_test2)# Remove numbers
cleaned_test2 = cleaned_test2.replace('\n', '')
cleaned_test2 = ' '.join(cleaned_test2.split())
cleaned_test2=cleaned_test2.lower()
cleaned_test2


# In[243]:


english_words = re.findall(r'\b[a-zA-Z]+\b', cleaned_test2)

# Join the extracted words into a single string
cleaned_test2 = ' '.join(english_words)


# In[360]:


unique_characters = set(cleaned_test)
unique_characters2 = set(cleaned_test2)

# Print the unique characters
print("Unique characters in the string:", unique_characters)
print("Unique characters in the string2:", unique_characters2)


# In[230]:


import numpy as np
from scipy.sparse import lil_matrix

# ... (your code to create the transition matrix) ...

# Apply Laplace smoothing
smoothed_transition_matrix = transition_matrix + 1

# Normalize the smoothed transition matrix
row_sums = smoothed_transition_matrix.sum(axis=1)
row_sums = np.asarray(row_sums).flatten()  # Convert row_sums to a 1D array
row_sums[row_sums == 0] = 1  # Avoid division by zero
smoothed_transition_matrix = smoothed_transition_matrix / row_sums[:, np.newaxis]

# Print the smoothed transition matrix
print("Smoothed Transition Matrix:")
for i in range(num_alphabets):
    for j in range(num_alphabets):
        print(smoothed_transition_matrix[i, j], end="\t")
    print()  # Print a new line after each row


# In[233]:


transition_matrix[np.isnan(transition_matrix)] = 0
transition_matrix


# In[383]:


books = sorted(glob.glob("C:\\Users\\hp\\Downloads\\ModelCompare\\Book1.txt"))

print("Available Books:\n")
for i in books:
    print(i.split("\\")[-1].split("_")[0])

test2 = ""
chars_test2 = []

for book in books:
    print("Reading " + str(book).split("\\")[-1].split("_")[0])
    with codecs.open(book, "r", "utf-8") as infile:
        test2 += infile.read()
        chars_test2.append(len(test2))
        print("Characters read so far: " + str(len(test2)))
        
cleaned_test2 = re.sub(r'[^\w\s]', '', test2)  # Remove punctuation
cleaned_test2 = re.sub(r'\d+', '', cleaned_test2)# Remove numbers
cleaned_test2 = cleaned_test2.replace('\n', '')
cleaned_test2 = ' '.join(cleaned_test2.split())
cleaned_test2=cleaned_test2.lower()
english_words = re.findall(r'\b[a-zA-Z]+\b', cleaned_test2)
# Join the extracted words into a single string
cleaned_test2 = ' '.join(english_words)
cleaned_test2


# In[384]:


epsilon = 1e-6  # You can adjust this value as needed

transition_matrix[np.isnan(transition_matrix)] = epsilon
transition_matrix_first_six[np.isnan(transition_matrix_first_six)] = epsilon
transition_matrix_first_six


# In[385]:


import numpy as np

# Given transition matrix from the first six books and sequences of alphabets
transition_matrix_first_six = np.array(transition_matrix)  # Convert to NumPy array
sequence_harry_potter_7th = cleaned_test  # Replace with the 7th part of Harry Potter sequence
sequence_random_book = cleaned_test2  # Replace with the sequence from the random book

# Create a mapping of characters to indices
char_to_index = {char: index for index, char in enumerate(['h', 'f', ' ', 'u', 'y', 'e', 'z', 'k', 'p', 'r', 'm', 'q', 'd', 'o', 'b', 'x', 'n', 's', 't', 'l', 'i', 'g', 'c', 'v', 'a', 'j', 'w'])}

# Calculate log likelihood for a given transition matrix and sequence
def calculate_log_likelihood(transition_matrix, sequence):
    log_likelihood = 0
    epsilon = 1e-10  # Small epsilon value to prevent division by zero
    for i in range(len(sequence) - 1):
        from_char = sequence[i]
        to_char = sequence[i + 1]
        from_index = char_to_index[from_char]
        to_index = char_to_index[to_char]
        transition_prob = transition_matrix[from_index, to_index]
        log_likelihood += np.log(transition_prob + epsilon)  # Add epsilon to prevent division by zero
    return log_likelihood

# Calculate log likelihoods for both sequences
log_likelihood_harry_potter_7th = calculate_log_likelihood(transition_matrix_first_six, sequence_harry_potter_7th)
log_likelihood_random_book = calculate_log_likelihood(transition_matrix_first_six, sequence_random_book)
print(log_likelihood_harry_potter_7th)
print(log_likelihood_random_book)
# Compare log likelihoods
if -log_likelihood_harry_potter_7th > -log_likelihood_random_book:
    print("The 7th part of Harry Potter aligns more closely with the transition matrix.")
else:
    print("The random book aligns more closely with the transition matrix.")


# In[ ]:





# In[386]:


import matplotlib.pyplot as plt

# Your log likelihood values
log_likelihood_1 = -log_likelihood_harry_potter_7th
log_likelihood_2 = -log_likelihood_random_book

# Labels for the bars
labels = ['Likelihood of seventh part', 'Likelihood of wikipedia page']

# Values for the bars
values = [log_likelihood_1, log_likelihood_2]

# Create a bar chart
plt.bar(labels, values, color=['blue', 'green'])

# Add labels and a title
plt.xlabel('Log Likelihood')
plt.ylabel('Value')
plt.title('Comparison of Log Likelihood Values')

# Show the chart
plt.show()


# In[262]:


print("First Six Matrix Shape:", transition_matrix_first_six.shape)
print("Harry Potter 7th Matrix Shape:", transition_matrix_test.shape)
print("Random Book Matrix Shape:", transition_matrix_test2.shape)

# Flatten the matrices into vectors
vector_first_six = transition_matrix_first_six.flatten()
vector_harry_potter_7th = transition_matrix_test.flatten()
vector_random_book = transition_matrix_test2.flatten()

# Replace NaN values with zeros
vector_first_six = np.nan_to_num(vector_first_six)
vector_harry_potter_7th = np.nan_to_num(vector_harry_potter_7th)
vector_random_book = np.nan_to_num(vector_random_book)

# Reshape the vectors to have shape (1, -1) for cosine_similarity function
vector_first_six = vector_first_six.reshape(1, -1)
vector_harry_potter_7th = vector_harry_potter_7th.reshape(1, -1)
vector_random_book = vector_random_book.reshape(1, -1)

# Compute the cosine similarities
cosine_similarity_harry_potter = cosine_similarity(vector_first_six, vector_harry_potter_7th)
cosine_similarity_random = cosine_similarity(vector_first_six, vector_random_book)

# The result will be a similarity score between -1 and 1, where higher values indicate closer alignment.
print("Cosine Similarity with Harry Potter 7th Part:", cosine_similarity_harry_potter[0][0])
print("Cosine Similarity with Random:", cosine_similarity_random[0][0])

if cosine_similarity_harry_potter > cosine_similarity_random:
    print("The 7th part of Harry Potter aligns more closely with the transition matrix.")
else:
    print("The random book aligns more closely with the transition matrix.")


# In[330]:


import matplotlib.pyplot as plt

# Your log likelihood values
log_likelihood_1 = cosine_similarity_harry_potter[0][0]
log_likelihood_2 = cosine_similarity_random[0][0]

# Labels for the bars
labels = ['cosine similarity 7th part', 'cosine similarity random']

# Values for the bars
values = [log_likelihood_1, log_likelihood_2]

# Create a bar chart
plt.bar(labels, values, color=['blue', 'green'])

# Add labels and a title
plt.xlabel('Cosine similarity')
plt.ylabel('Value')
plt.title('Comparison of Cosine Similarity Values')

# Show the chart
plt.show()


# In[264]:


import numpy as np
from scipy.stats import chisquare

# Assuming you have your transition matrices already loaded as:
# transition_matrix_first_six, transition_matrix_harry_potter_7th, and transition_matrix_random_book

# Flatten the matrices into 1D arrays
expected_frequencies = transition_matrix_first_six.flatten()
observed_harry_potter_7th = transition_matrix_harry_potter_7th.flatten()
observed_random_book = transition_matrix_random_book.flatten()

# Add a small constant to expected frequencies to avoid division by zero
epsilon = 1e-6
expected_frequencies += epsilon

# Perform the chi-square test
chi2_stat_harry_potter = chisquare(f_obs=observed_harry_potter_7th, f_exp=expected_frequencies)
chi2_stat_random_book = chisquare(f_obs=observed_random_book, f_exp=expected_frequencies)

# The result will include the chi-square statistic and the p-value.
# A higher p-value suggests closer alignment.
print("Chi-Square Statistic and p-value for Harry Potter 7th Part:")
print("Chi-Square Statistic:", chi2_stat_harry_potter.statistic)
print("p-value:", chi2_stat_harry_potter.pvalue)

print("\nChi-Square Statistic and p-value for Random Book:")
print("Chi-Square Statistic:", chi2_stat_random_book.statistic)
print("p-value:", chi2_stat_random_book.pvalue)


# In[266]:


# Check the shape of your observed arrays
print(observed_harry_potter_7th.shape)
print(observed_random_book.shape)
print(expected_frequencies.shape)

# Check for missing or unexpected categories
# Ensure both observed arrays have the same categories as expected_frequencies

# Verify your expected_frequencies calculations

# Handle missing or infinite values if necessary

from scipy.stats import chisquare

# Add a small constant to avoid division by zero
epsilon = 1e-10
expected_frequencies = expected_frequencies + epsilon

# Retry the chi-squared test
chi2_stat_harry_potter = chisquare(f_obs=observed_harry_potter_7th, f_exp=expected_frequencies)
chi2_stat_random_book = chisquare(f_obs=observed_random_book, f_exp=expected_frequencies)

print("Chi-squared statistic for 'Harry Potter and the Deathly Hallows':", chi2_stat_harry_potter)
print("Chi-squared statistic for a random book:", chi2_stat_random_book)


# In[267]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have transition probability matrices for each book
transition_matrix_harry_potter_7th = transition_matrix_test
transition_matrix_random_book = transition_matrix_test2
transition_matrix_first_six_books = transition_matrix

# Create a list of book names
book_names = ["Harry Potter 7th", "Random Book", "First Six Books"]

# Create subplots for heatmaps
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot heatmaps for each book
for i, transition_matrix in enumerate([transition_matrix_harry_potter_7th, transition_matrix_random_book, transition_matrix_first_six_books]):
    sns.heatmap(transition_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[i])
    axes[i].set_title(book_names[i])
    axes[i].set_xlabel("Current Chapter")
    axes[i].set_ylabel("Next Chapter")

plt.tight_layout()
plt.show()


# In[283]:


import numpy as np
from scipy import stats

# Assuming you have transition probability matrices for each book
transition_matrix_harry_potter_7th = transition_matrix_test
transition_matrix_random_book = transition_matrix_test2
transition_matrix_first_six_books = transition_matrix

# Flatten the matrices to get transition probabilities as 1D arrays
transition_probs_harry_potter_7th = transition_matrix_harry_potter_7th.flatten()
transition_probs_random_book = transition_matrix_random_book.flatten()
transition_probs_first_six_books = transition_matrix_first_six_books.flatten()

transition_probs_harry_potter_7th=np.nan_to_num(transition_probs_harry_potter_7th, nan=0)
transition_probs_random_book=np.nan_to_num(transition_probs_random_book, nan=0)
transition_probs_first_six_books=np.nan_to_num(transition_probs_first_six_books, nan=0)
# Assuming you have the transition probability matrices for the three cases
# Let's call them matrix_seventh_harry_potter, matrix_random_book, matrix_first_six_books

# Add a small constant to avoid zero variance
epsilon = 1e-10  # Small constant
transition_probs_harry_potter_7th += epsilon
transition_probs_random_book += epsilon
transition_probs_first_six_books += epsilon

# Perform t-tests for each transition probability value
p_values = np.empty_like(transition_probs_harry_potter_7th)

for i in range(transition_probs_harry_potter_7th.shape[0]):
    for j in range(transition_probs_harry_potter_7th.shape[1]):
        # Perform t-test for the (i, j) element of the matrices
        _, p_value = stats.ttest_ind(transition_probs_harry_potter_7th[i, j], transition_probs_random_book[i, j])
        print(stats.ttest_ind(transition_probs_harry_potter_7th[i, j], transition_probs_random_book[i, j]))
        p_values[i, j] = p_value

# Set the significance level (alpha)
alpha = 0.05
print(p_values)
# Check if any p-value is less than alpha
if np.any(p_values < alpha):
    print("There is a statistically significant difference between the seventh part of Harry Potter and the random book.")
else:
    print("There is no statistically significant difference between the seventh part of Harry Potter and the random book.")


# In[288]:


np.var(transition_probs_random_book)


# In[289]:


np.var(transition_probs_harry_potter_7th)


# In[290]:


p_values = np.empty_like(transition_probs_harry_potter_7th)


# In[291]:


p_values


# In[293]:


transition_probs_harry_potter_7th.shape[1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[311]:


transition_probs_harry_potter_7th[0,3]


# In[312]:


transition_probs_random_book[0,3]


# In[349]:


from scipy.stats import chi2_contingency

p_values = np.zeros_like(transition_probs_random_book)

for i in range(transition_probs_random_book.shape[0]):
    for j in range(transition_probs_harry_potter_7th.shape[1]):
        # Perform t-test for the (i, j) element of the matrices
        observed = np.array([[transition_probs_harry_potter_7th[i, j], transition_probs_first_six_books[i, j]],
                             [transition_probs_random_book[i, j], transition_probs_first_six_books[i, j]]])
        
        # Perform the chi-squared test for independence
        chi2, _, _, p_value_array = chi2_contingency(observed, correction=False)
        
        p_value = p_value_array[0, 0]
        # Store the p-value in the corresponding position
        p_values[i, j] = p_value


# In[350]:


p_value


# In[351]:


alpha = 0.05

# Check if any p-value is less than alpha
if np.any(p_values < alpha):
    print("There is a statistically significant difference between the seventh part of Harry Potter and the random book.")
else:
    print("There is no statistically significant difference between the seventh part of Harry Potter and the random book.")


# In[352]:


import matplotlib.pyplot as plt

# Assuming you have a single p-value
alpha = 0.00001 # Set your desired alpha threshold

# Create a bar chart with a colored bar for the single p-value
color = 'red' if p_value < alpha else 'blue'

plt.figure(figsize=(6, 4))
plt.bar(['Hypothesis'], [p_value], color=color)
plt.xlabel('Hypothesis')
plt.ylabel('p-value')
plt.title('p-value Visualization')
plt.axhline(y=alpha, color='r', linestyle='--', label=f'Alpha = {alpha}')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:




