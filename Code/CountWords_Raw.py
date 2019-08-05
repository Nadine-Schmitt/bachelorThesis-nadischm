#!/usr/bin/env python
# coding: utf-8

# # Frequeny of words from KORE dataset of the raw input corpus


# 
# For each entity from the KORE datatset the number of occurence in the raw input corpus is calculated and printed as output.

# ## Import

import pickle
import argparse
import os
import gensim
from gensim.models import Word2Vec


# ## Functions

# Read KORE dataset into list and split entities which multiple tokens( e.g. Apple Inc. into Apple and Inc.) in order to count for each word seperately:

def readKore(source):
    list = [line.rstrip('\n') for line in open(source, encoding="UTF-8")]
   
    lenght = len(list)
    #list[0] = 'Apple Inc.'
    #list[195] = 'Golden Globe Award for Best Actor - Motion Picture Drama' #instead of Golden Globe Award for Best Actor â€“ Motion Picture Drama
    #list[299]= 'Kärtsy Hatakka' #instead of KÃ¤rtsy Hatakka
    #list[302]= 'Ragnarök' #instead of RagnarÃ¶k
    #print(list)

    for index in range(lenght):
        if list[index][0] == "\t":
            list[index] = list[index][1:len(list[index])]
    #raw: single words
    listSingle = []
    i=0
    for e in list:
        eNew = e.split(' ')
        for ele in eNew:
            listSingle.insert(i, ele)
            i=i+1
    
    return listSingle


# The following function reads the inputList raw file by file and for each word from the KORE dataset the number of occurence is calculated:

def loadInputList(dirname, KoreList):
    counterList = []
    i = 0
    while i < len(KoreList):
        counterList.append(0)
        i = i+1
    loadSentence = []
    filenames = [f.path for f in os.scandir(dirname) ]
    for fname in filenames:
        print("read sentences from ", fname)
        loadSentence = []
        with open(fname, 'r') as f:
            line = f.readline()
            while line:
                words = line.split()
                for w in words:
                    loadSentence.append(w)
                line = f.readline()
            #print(loadSentence)
            #count words
            iterator = 0
            for e in KoreList:
                #print(e)
                counterList[iterator] += loadSentence.count(e)
                iterator = iterator +1
                
    return counterList               
    


# ## Configuration and Main


parser = argparse.ArgumentParser(description='Script for translating wikipedia page names')
parser.add_argument('koreSource', type=str, help='source folder with kore dataset')
parser.add_argument('inputSource', type=str, help='source folder with inputlist raw')

args = parser.parse_args()


#read Kore data
KoreList = readKore(args.koreSource)
#KoreList[0] ='Apple'
for e in KoreList:
    print(e) #print all words from Kore
count = loadInputList(args.inputSource,KoreList)
for e in count:
    print(e) #print afterwords their occurence
