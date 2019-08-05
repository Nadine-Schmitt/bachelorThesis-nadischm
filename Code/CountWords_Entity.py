#!/usr/bin/env python
# coding: utf-8

# # Frequeny of words from KORE dataset of the entity input corpus

# 
# For each entity from the KORE datatset the number of occurence in the entity input corpus is calculated and printed as output.

# ## Import


import pickle
import argparse
import os
import gensim
from gensim.models import Word2Vec


#
# ## Functions

# The underscore_creator() take an entity (e.g. Apple Inc.) and transform it into its id (Apple_Inc.)


def underscore_creator(s): 
    w = ''
    l = s.split(' ')
    for index in range(len(l) -1): 
        w += l[index] + '_'
    w += l[-1]
    return w 


# Read KORE dataset into list and transform entities into their id with the underscore-creator():


def readKore(source):
    
    list = [line.rstrip('\n') for line in open(source, encoding="UTF-8")]
   
    lenght = len(list)
        for index in range(lenght):
        if list[index][0] == "\t":
            list[index] = list[index][1:len(list[index])]
    #entity: words with underscore
    listUnderscore = []
    for e in list:
        eNew = underscore_creator(e)
        listUnderscore.append(eNew)
            
    return listUnderscore



# The following function reads the inputList entity file by file and for each word from the KORE dataset the number of occurence is calculated:


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
parser.add_argument('inputSource', type=str, help='source folder with inputlist entity')

args = parser.parse_args()


#read Kore data
KoreList = readKore(args.koreSource)
KoreList[0] ='Apple'
for e in KoreList:
    print(e) #print all words from Kore
count = loadInputList(args.inputSource,KoreList)
for e in count:
    print(e) #print afterwards their occurence
#print(KoreList[0])