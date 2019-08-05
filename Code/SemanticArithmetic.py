#!/usr/bin/env python
# coding: utf-8

# # Qualitative Examination


# This script train a word and entity model and do a qualitative examination afterwards.
# 
 

# # Import

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


# The drawing_words-function reduces the dimensionaltiy of given words either with PCA(https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) or t-SNE(https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) and draws the words into a diagram. 
# 
# Required parameters:
# - **model** which is used to visualize vectors from
# - **words** a list of words, which should be visualized
# - **pca** use PCA, if TRUE and t-SNE otherwise
# - **alternate** different color and label align is used for every second word
# - **arrows** arrows are used to connect related words (i.e. items that are next to each other in the list)
# - **x1** x axis range from
# - **x2** x axis range to
# - **y1** y axis range from
# - **y2** y axis range to
# - **title** title of the diagram


def drawing_words(saveName, model, words, pca=False, alternate=True, arrows=True, x1=3, x2=3, y1=3, y2=3, title=''):
    # get vectors for given words from model
    vectors = [model[word] for word in words]

    #use pca
    if pca:
        pca = PCA(n_components=2, whiten=True)
        vectors2d = pca.fit(vectors).transform(vectors)
    #use t-SNE
    else:
        tsne = TSNE(n_components=2, random_state=0)
        vectors2d = tsne.fit_transform(vectors)

    # draw image
    plt.figure(figsize=(6,6))
    if pca:
        plt.axis([x1, x2, y1, y2])

    first = True # color alternation to divide given groups
    for point, word in zip(vectors2d , words):
        # plot points
        plt.scatter(point[0], point[1], c='r' if first else 'g')
        # plot word annotations
        plt.annotate(
            word, 
            xy = (point[0], point[1]),
            xytext = (-7, -6) if first else (7, -6),
            textcoords = 'offset points',
            ha = 'right' if first else 'left',
            va = 'bottom',
            size = "x-large"
        )
        first = not first if alternate else first

    # draw arrows
    if arrows:
        for i in range(0, len(words)-1, 2):
            a = vectors2d[i][0] + 0.04
            b = vectors2d[i][1]
            c = vectors2d[i+1][0] - 0.04
            d = vectors2d[i+1][1]
            plt.arrow(a, b, c-a, d-b,
                shape='full',
                lw=0.1,
                edgecolor='#bbbbbb',
                facecolor='#bbbbbb',
                length_includes_head=True,
                head_width=0.08,
                width=0.01
            )

    # draw diagram title
    if title:
        plt.title(title)

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


# ## Load input corpora for training and read parameters


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

# The trained model have a high dimensional word vectors and with the drawing_words-function a list of words can be plottet.
# 
# In the following 2 word classes are given, which are alternately put in a list and the alternate parameter of the function is set to TRUE in order to produce arrows. Countries and their corresponding currencies are plotted and saved into figures/currency file:


# plot currencies
words = ["Switzerland","franc","Germany","Euro","England","pound","Japan","yen"]
drawing_words('figures/currency', model, words, True, True, True, -3, 3, -2, 2, r'$PCA\ Visualisierung:\ Currencies$')


# In the following 2 word classes are given, which are alternately put in a list and the alternate parameter of the function is set to TRUE in order to produce arrows. Countries and their corresponding capitals are plotted and saved into figures/capitals file:

# plot capitals
words  = ["Athens","Greece","Berlin","Germany","Paris","France","Bern","Switzerland","Vienna","Austria","Lisbon","Portugal","Moscow","Russia","Rome","Italy","Tokyo","Japan","London","England"]
drawing_words('figures/capitals', model, words, True, True, True, -3, 3, -2, 2.2, r'$PCA\ Visualisierung:\ Capitals$')


# In the following 2 word classes are given, which are alternately put in a list and the alternate parameter of the function is set to TRUE in order to produce arrows. Countries and their corresponding languages are plotted and saved into figures/language file:

# plot language
words = ["Germany","German","Italy","Italian","France","French","Greece","Greek","Spain","Spanish","Sweden","Swedish"]
drawing_words('figures/language', model, words, True, True, True, -3, 3, -2, 1.7, r'$PCA\ Visualisierung:\ Language$')


# The next 3 examples shows related words to a given word (tiger, cucumber and car) by using the most_similar() function(https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar) of Gensim:


# plot related words to 'tiger' with similarity function from gensim
matches = model.most_similar(positive=["tiger"], negative=[], topn=10)
words = [match[0] for match in matches]
drawing_words('figures/tiger',model, words, True, False, False, -3, 2, -2, 2, r'$PCA\ Visualisierung:\ tiger$')



# plot related words to 'cucumber' with similarity function from gensim
matches = model.most_similar(positive=["cucumber"], negative=[], topn=10)
words = [match[0] for match in matches]
drawing_words('figures/cucumber', model, words, True, False, False, -3, 2, -2, 2, r'$PCA\ Visualisierung:\ cucumber$')



# plot related words to 'car' with similarity function from gensim
matches = model.most_similar(positive=["car"], negative=[], topn=10)
words = [match[0] for match in matches]
drawing_words('figures/cars', model, words, True, False, False, -3, 2, -2, 2, r'$PCA\ Visualisierung:\ car$')


# In the following the correct gender of a given name should be captured:

# plot name
words = ["Annika","Anton","Andrea","Charlotte","Charles","Emily","Eric","Florian","Felix","Johanna","Judith","Lara","Julian","Lea","Lisa","Lina","Lukas","Mia","Nico","Sophie","Simon", "Tom"]
drawing_words('figures/gender', model, words, True, True, False, -3, 3, -1.5, 2.5, r'$PCA\ Visualisierung:\ Name\ according to \ gender$')


# ## Total run-time

endTime = time.time()
print("total run-time", endTime - startTime)

