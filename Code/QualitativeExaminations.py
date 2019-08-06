#!/usr/bin/env python
# coding: utf-8

# # Qualitative Examination

# 
# This script train a word and entity model and do a qualitative examination afterwards.
# 
# Therefore the following code reduces dimensionality of word embeddings with PCA(https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) or t-SNE(https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). The two-dimensional representation of the words can be plotted by using [pythons matplotlib](https://matplotlib.org).
# 
# In addition, the most_similar() function(https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar) of Gensim is used to show related words to a given word, e.g. cucumber.
# 

# # Import

# 
import gensim
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import argparse
from gensim.models import Word2Vec
import time
import multiprocessing as mp


# ## Functions

# With following function the parameter setting is read. It returns a list of lists, in which each list is a parameter setting:


#read parameter settings and get list of lists in which each list is a parameter setting
def readParaSetting(paraList):
    finallist = []
    lines = [line.rstrip('\n') for line in open(paraList)]
    #lines[-1] = lines[-1][0:len(lines[-1])-1]
    for e in lines:
        if len(e) > 0:
            list =[]
            listelements = e.split(' , ')
            for i in listelements:
                list.append(i)
            finallist.append(list)
    return finallist


# The plot-function reduces the dimensionaltiy of given words. Afterwards it draws the words into a plot. 
# 
# Required parameters are:
# - **saveName** name of the file to save the plot to
# - **model** which is the trained Word2Vec model
# - **wordList** a list of words that should be drawn
# - **pca** PCA is used if TRUE and t-SNE if FALSE
# - **byTurns** a different label align and colour is used for every second word
# - **connect** items which are next to each other in the word list are connected with arrows when this parameter is set
# - **xfrom** start of x axis 
# - **xto** end of x axis 
# - **yfrom** start of y axis 
# - **yto** end of y axis 
# - **heading** heading of the plot

def plot(saveName, model, wordList, pca=False, byTurns=True, connect=True, xfrom=3, xto=3, yfrom=3, yto=3, heading=''):
    
    # get all embeddings for the given words in the list 
    embeddings = [model[word] for w in words]

    #use pca to get two-dimensional presentation of the embeddings
    if pca:
        pca = PCA(n_components=2, whiten=True)
        newEmbeddings = pca.fit(embeddings).transform(embeddings)
    #use t-SNE
    else:
        tsn = TSNE(n_components=2, random_state=0)
        newEmbeddings = tsne.fit_transform(embeddings)

    # draw plot
    plt.figure(figsize=(6,6))
    if reducingMethod:
        plt.axis([xfrom, xto, yfrom, yto])

    first = True # colour byTurns
    for point, w in zip(newEmbeddings , wordList):
        # plot points
        plt.scatter(point[0], point[1], c='r' if first else 'g')
        # plot word byTurns
        plt.annotate(
            w, 
            xy = (point[0], point[1]),
            xytext = (-7, -6) if first else (7, -6),
            textcoords = 'offset points',
            ha = 'right' if first else 'left',
            va = 'bottom',
            size = "x-large"
        )
        first = not first if byTurns else first

    # arrows
    if connect:
        for i in range(0, len(wordList)-1, 2):
            a = newEmbeddings[i][0] + 0.04
            b = newEmbeddings[i][1]
            c = newEmbeddings[i+1][0] - 0.04
            d = newEmbeddings[i+1][1]
            plt.arrow(a, b, c-a, d-b,
                shape='full',
                lw=0.1,
                edgecolor='#bbbbbb',
                facecolor='#bbbbbb',
                length_includes_head=True,
                head_width=0.08,
                width=0.01
            )

    # heading
    if heading:
        plt.title(heading)

    plt.tight_layout()
    
    #save diagram into saveName-file
    plt.savefig(saveName, format ='png')


# ## Configuration

#start record time
startTime = time.time()

parser = argparse.ArgumentParser(description='Script for training word embeddings')
parser.add_argument('source', type=str, help='source folder with preprocessed input data')
parser.add_argument('paraList', type=str, help='source folder of paraList')
parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='number of worker threads to train the model')
args = parser.parse_args()


# ## Load input corpus for training and read parameters

#load input corpus with PathLineSentences
sentences = gensim.models.word2vec.PathLineSentences(args.source)

#set parameter (read one parameter setting from a textfile)
paraList = readParaSetting(args.paraList)


# ## Train model with Word2Vec


#train model with word2vec
#
#for each parameter setting 
#note that here only one setting should be in list; readParaSetting is used to read the Parameter Setting from a text file)
for e in paraList:
    sgI = int(e[3])
    cbowmeanI = int(e[6])
    sizeI = int(e[0]) 
    windowI = int(e[1])
    min_countI = int(e[2])
    hsI = int(e[4])
    negativeI = int(e[5]) 
        
    model  = Word2Vec(sentences,
                        size=sizeI,
                        window=windowI,
                        min_count=min_countI,
                        workers= mp.cpu_count(),
                        sg=sgI,
                        hs=hsI,
                        negative=negativeI,
                        cbow_mean=cbowmeanI)

#calculate training time and print it
trainingTime = time.time()    
print('model trained and it took:' , trainingTime  - startTime)  


# ## Plotting 

# The trained model have high-dimensional word vectors and with the plot-function a list of words can be plottet into a graph with 2 dimensions.
# 
# In the following two classes of words are given, which are put by turns in a list of words. Note that the arrows parameter is set to TRUE in order to produce arrows:



# plot food and their corresponding countries and save it into figures/food
wordList = ["sauerkraut","Germany","pizza","Italy","baguette","France","doner","Turkey"]
plot('figures/currency', model, wordList, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ Food$')

# plot countries and their corresponding capitals and save it into figures/capitals
wordList  = ["Athens","Greece","Berlin","Germany","Paris","France","Bern","Switzerland","Vienna","Austria","Lisbon","Portugal","Moscow","Russia","Rome","Italy","Tokyo","Japan","London","England"]
plot('figures/capitals', model, wordList, True, True, True, -3, 3, -2, 2.2, r'$PCA\ Visualisierung:\ Capitals$')

# plot countries and their corresponding language and save it into figures/languages
wordList = ["Germany","German","Italy","Italian","France","French","Greece","Greek","Spain","Spanish","Sweden","Swedish"]
plot('figures/language', model, wordList, True, True, True, -3, 3, -2, 1.7, r'$PCA\ Visualisierung:\ Language$')

# plot countries and their corresponding currencies and save it into figures/currency
wordList = ["Switzerland","franc","Germany","Euro","England","pound","Japan","yen"]
plot('figures/currency', model, wordList, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ Currencies$')

# plot countries and their corresponding head of government and save it into figures/government
wordList = ["Germany","Merkel","Russia","Putin","France","Macron","Austria","Kurz"]
plot('figures/currency', model, wordList, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ Head of government$')


# The next 3 examples shows related words to a given word (tiger, cucumber and car) by using the [most_similar() function](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar) of Gensim:


# plot related words to 'tiger' with similarity function from Gensim
similar = model.most_similar(positive=["tiger"], negative=[], topn=10)
wordList = [sim[0] for sim in similar]
plot('figures/tiger',model, wordList, True, False, False, -3, 2, -2, 2, r'$PCA\ Visualisierung:\ tiger$')



# plot related words to 'cucumber' with similarity function from Gensim
similar = model.most_similar(positive=["cucumber"], negative=[], topn=10)
wordList = [sim[0] for sim in similar]
plot('figures/cucumber', model, wordList, True, False, False, -3, 2, -2, 2, r'$PCA\ Visualisierung:\ cucumber$')



# plot related words to 'car' with similarity function from Gensim
similar = model.most_similar(positive=["car"], negative=[], topn=10)
wordList = [sim[0] for sim in similar]
plot('figures/cars', model, wordList, True, False, False, -3, 2, -2, 2, r'$PCA\ Visualisierung:\ car$')


# In the following the correct gender of a given name and the correct food category (fruit or vegetable) of a given food should be captured:


# plot name to get correct gender
wordList = ["Annika","Anton","Andrea","Andreas","Emily","Charles","Erica","Florian","Fiona","Johannes","Judith","Lars","Julia","Leon","Lisa","Linus","Lucia","Mira","Nicole","Nico","Simona", "Tom"]
plot('figures/gender', model, wordList, True, True, False, -3, 3, -1.5, 2.5, r'$PCA\ Visualisierung:\ Name\ according to \ gender$')

#plot foods
wordList = ["banana", "cucumber", "orange", "tomato", "cherry", "garlic", "apple", "carrot"]
plot('figures/foodCategory', model, wordList, True, True, False, -3, 3, -1.5, 2.5, r'$PCA\ Visualisierung:\ Food category$')


# ## Total run-time


endTime = time.time()
print("total run-time", endTime - startTime)


