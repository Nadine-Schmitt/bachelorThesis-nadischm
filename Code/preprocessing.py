#!/usr/bin/env python
# coding: utf-8

# # Preprocessing

# The preprocessing is done for each downloaded and extracted Wikipedia dump (i.e. for each language) twice in order to get a ``raw input corpus`` and an ``entity annotated corpus``.

# ## Import

import argparse
import base64
import io
import json
import os
from pathlib import Path
import pickle
import nltk
import re
import time
#
# our global variables
acronymList = []    #list of acronyms


# ## Functions 

# readAcronymList-function reads the acronyms with dots into a list:

# In[ ]:


def readAcronymList(fName):
    acronymList.clear()
    with open(fName, errors = 'ignore') as f:
        for words in f:
            testw = words.rstrip("\n")
            if len(testw) > 0 :
                acronymList.append(testw)


# When producing an entity annotated input corpus (-e is set), then each entity in the text is substituted with its ID. If there is for example the sentence _Yesterday Obama was in Paris_ with entity _(11,15) (Obama, Barack Obama)_, the character position from 11 to 15 (with characters _Obama_) is taken and substituted with the entity id Barack_Obama. 
# 
# Note that sometimes the returning char offset from the extracted Wikipedia dump is false and have to be corrected. For example in AA/wiki_00 there is such an entity with a wrong char offset: _(1631, 1637) archon archon_. As archon beginns at position 1634 in the text and without handling this _hon_ would be printed aditionally. Therefore the char offset for each entity is verified (if doVerify) and if it is wrong, it is corrected.
# 
# 



def Wiki_Extractor(fileName, tt): 
    new_text = ''
    with io.open(fileName, errors = 'ignore') as f:
        for wiki_article in f.readlines():
            # each line is a dict with keys ['id', 'url', 'title', 'text', 'internal_links']
            wiki_article = json.loads(wiki_article)
            
            #getting raw text, when tt is False
            if not tt:
                #return wiki_article['text']
                new_text += wiki_article['text']
                #print(new_text)
            #getting annotated text
            else:
                beginn = 0
                doVerify = False
                searchWindow = 10   # search up to 10 characters behind the char_offsets
                textShift = 0       #how many characters are needed to shift the mention
                
                # The dictionary in 'internal_links' has begin and end char offsets as keys and 
                # then a tuple of mention (i.e. the link name) and entity (i.e. Wikipedia page 
                # name). Notice, f.ex. the line with 
                # 
                # (2317, 2328) ('Love Affair', 'Love Affair (1994 film)')
                #
                # where text 'Love Affair' was linking to the Wikipedia page 'Love Affair (1994 film)'
                for (char_offsets, (mention, wiki_page_name)) in pickle.loads(base64.b64decode(wiki_article['internal_links'].encode('utf-8'))).items():
                    #print(char_offsets, mention, wiki_page_name)
                    entity_name = ''
                    #the new text contains the whole text until the word beginns
                    if wiki_article['text'][beginn] == " ":
                        new_text += wiki_article['text'][beginn:(char_offsets[0]+textShift)]
                    else:
                        new_text += " " + wiki_article['text'][beginn:(char_offsets[0]+textShift)]
                    #create the word with underscores
                    entity_list = wiki_page_name.split()
                    if len(entity_list) >0: 
                        for index in range(len(entity_list)-1):
                            entity_name += entity_list[index] + '_'
                        entity_name += entity_list[-1]
                    #new text gets the word with underscores
                    if new_text[-1] != " ":
                        new_text += " "
                        doVerify = True
                    new_text += entity_name
                    
                    #verify if char offeset is correct
                    if doVerify:
                        sStart = char_offsets[0]
                        sEnde = char_offsets[1] + searchWindow
                        if sEnde > len(wiki_article['text']):
                            sEnde = len(wiki_article['text']) -1
                        iFound = wiki_article['text'][sStart:sEnde].find(mention)
                        if iFound < 1:
                            beginn = char_offsets[1] + textShift
                        else:
                            beginn = char_offsets[1] + iFound
                            textShift = iFound
                            searchWindow = 10 + iFound
                    else:
                        #the next step in the loop beginns at the index where the word ends 
                        beginn = char_offsets[1]                   
    #return raw or entity annoated corpus
    return new_text


# Following function removes the title and category lines as well as unwanted newlines:


def trim2article(inText):
    #
    # as python starts with 0 itLen as pointer in the text is 1 to large
    itLen = len(inText) - 1
    outText = ""
    first = True
    i = 0
    #print("Textlen:" + str(itLen))
    
    while i<itLen:
        #print("next loop at position " + str(i))
        if first:
            cPointer = inText.find("Category:",i)
            nlPointer = inText.find('\n', i)
            i = nlPointer + 1
            first = False
        else:
            if inText[i+1] == "\n": 
                i += 1
            else:
                nlPointer = inText.find('\n',i)
                if nlPointer != -1 and nlPointer < cPointer:
                    outText += inText[i:nlPointer] + " "
                    i = nlPointer +1
                else:
                    if nlPointer == -1:
                        outText += inText[i:cPointer]
                        i = itLen
                    else:
                        outText += inText[i:cPointer]
                        cPointer = inText.find("Category:",nlPointer)
                        nlPointer = inText.find('\n',nlPointer+1)
                    
                        while cPointer != -1 and nlPointer != -1 and not first:
                            if nlPointer < cPointer:
                                first = True
                                i = nlPointer-1
                            else:
                                cPointer = inText.find("Category:",cPointer+1)
                                nlPointer = inText.find('\n',nlPointer+1)
                        if not first:
                            i = itLen
    return outText


# processText-function removes special characters, such as %, \ and extra spaces from text. 
# 
# Important: German, Italian, Spanish and French language have special characters, e.g. the German Ä, which the WikiExtractor.py can not handle and replace it with Ã„. As the input corpora should contain Ä instead of Ã„ latter is replaced with Ä by using the [UTF-8 enconding cheatsheet](https://bueltge.de/wp-content/download/wk/utf-8\_kodierungen.pdf):



def processText(inText, german, italian, french, spanish):
    # remove header...
    outText = trim2article(inText)
    #change to lowercase if argument l is given
    #remove all the digits, special characters
    outText = outText.replace('$' , '')
    outText = outText.replace('%' , '')
    
    #german
    if german:
        outText =outText.replace('Ã¼','ü')
        outText =outText.replace('Ã¶','ö')
        outText =outText.replace('Ã¤','ä')
        outText =outText.replace('ÃŸ','ß')
        outText =outText.replace('â€™','’')
        outText =outText.replace('Ã–','Ö')
        outText =outText.replace('Ãœ','Ü')
        outText =outText.replace('Ã„','Ä')
        outText =outText.replace('â€“','–')  #langer Gedankenstrich
        outText =outText.replace('Ã“','Ó') 
        
    #italian
    if italian:
        outText =outText.replace('Ã¨','è')
        outText =outText.replace('Ã¹','ù')
        outText =outText.replace('Ã©','é')
        outText =outText.replace('â€™','’')
        outText =outText.replace('Ã','à')
        outText =outText.replace('Ã¬','ì')
        outText =outText.replace('Ã²','ò')
        outText =outText.replace('Ã§','ç')
        outText =outText.replace('Ã‰','É')
        outText =outText.replace('Å”','À')
        outText =outText.replace('ÄŒ','È')
        outText =outText.replace('Äš','Ì')
        outText =outText.replace('Ä›','ì')
        outText =outText.replace('Ã?','Í')
        outText =outText.replace('Ã-','í')
        outText =outText.replace('ÄŽ','Ï')
        outText =outText.replace('Ä?','ï')
        outText =outText.replace('Å‡','Ò')
        outText =outText.replace('Ã“','Ó') 
        outText =outText.replace('Ã³','ó')
        outText =outText.replace('Å®','Ù')       
        outText =outText.replace('Ãš','Ú') 
        outText =outText.replace('Ãº','ú')       
                
    #French
    if french:
        outText =outText.replace('Å”','À')
        outText =outText.replace('Ã','à')
        outText =outText.replace('Ã','Â')
        outText =outText.replace('Ã¢','â')
        outText =outText.replace('Ä†','Æ')
        outText =outText.replace('Ä‡','æ')
        outText =outText.replace('Ã‡','Ç')      
        outText =outText.replace('Ã§','ç')
        outText =outText.replace('ÄŒ','È')
        outText =outText.replace('Ã¨','è')
        outText =outText.replace('Ã‰','É')
        outText =outText.replace('Ã©','é')
        outText =outText.replace('Ä˜','Ê')
        outText =outText.replace('Ä™','ê')
        outText =outText.replace('Ã‹','Ë')
        outText =outText.replace('Ã«','ë')
        outText =outText.replace('ÃŽ','Î')
        outText =outText.replace('Ã®','î')
        outText =outText.replace('ÄŽ','Ï')
        outText =outText.replace('Ä?','ï')
        outText =outText.replace('Ã”','Ô')
        outText =outText.replace('Ã´','ô')
        outText =outText.replace('Åš','Œ')
        outText =outText.replace('Å›','œ')
        outText =outText.replace('Å®','Ù') 
        outText =outText.replace('Ã¹','ù')
        outText =outText.replace('Å°','Û')
        outText =outText.replace('Å±','û')
        outText =outText.replace('Åº','Ÿ')
        outText =outText.replace('Ë™','ÿ')
    
    #Spanish
    if spanish:
        outText =outText.replace('Ã?','Á')
        outText =outText.replace('Ã¡','á')
        outText =outText.replace('Ã§','ç')
        outText =outText.replace('Ã‰','É')
        outText =outText.replace('Ã©','é')
        outText =outText.replace('Ã?','Í')
        outText =outText.replace('Ã-','í')
        outText =outText.replace('Åƒ','Ñ')
        outText =outText.replace('Å„','ñ')
        outText =outText.replace('Ã“','Ó') 
        outText =outText.replace('Ã³','ó')
        outText =outText.replace('Ãš','Ú') 
        outText =outText.replace('Ãº','ú')
        outText =outText.replace('Ã¼','ü')

    outText = re.sub('[^a-zA-Z0-9ÄÖÜäöüßèùéàìòçÉÓÀÈÌìÍíÏïÒóÙÚúÂâÆæÇÊêËëÎîÔôŒœÛûŸÿÁáÑñ._–()’\'-:&]', ' ', outText)
    outText = outText.replace(',' , '')
    
    
    #remove all extra spaces from the text
    outText = re.sub(r'\s+', ' ', outText)
    return outText


# Following function checks if a sentence has an acronym with dot (is in AcronymList) as last word:

def isAcronym(sentence):
    words = sentence.split()
    test = (words[-1] in acronymList)
    return test


# Sometimes sentences are unwanted broken by [nltk](https://www.nltk.org/api/nltk.tokenize.html), since there can be a dot without the sentence ending, when there is an acronym (Inc. for example). 
# 
# If there is a dot and the next sentence starts with an uppercase letter, then it is a new sentence. In the other case, the next sentence is appended to the current sentence. However there are examples like _i.e. Germany_, where after a dot there is an uppercase letter, but the text should not be split. Therefore an AcronymList, which contains all acronyms with a dot of an given language, is used. If _i.e_  is for example in the list, then the unwanted broken sentences are combined:

#
# combine sentences which where unwanted broken
# sList : list of sentences
#
# return fixed list of sentences
#
def fixUnwantedBreak(sList):
    # 
    outList = []
    if len(sList) > 0:
        eI = 1       #pointer in the input-list
        oI = 0       #pointer in the out-list
        outList.append(sList[0])
        while eI < len(sList):
            sentence = sList[eI].strip()
            if sentence[0].isupper():
                if isAcronym(outList[oI]):
                    outList[oI] = outList[oI] + ' ' + sentence
                else:
                    outList.append(sentence)
                    oI += 1
            else:
                outList[oI] = outList[oI] + ' ' + sentence
            eI += 1
    return outList


# Preprocessed inputLists are saved:


#save the finallist in the target file
# filename - where to save the finallist
# flist    - finallist which is saved
def saveModelList(filename, flist):
    print("saving the final list for training in " + filename)
    with open(filename, 'w', encoding="UTF-8") as f:
        #pickle.dump(flist, f)
        for sentence in flist:
            f.writelines("%s " % w for w in sentence)
            f.writelines("\n")


# Following function does the preprocessing by using the functions defined above.
# 
# With [nltk](https://www.nltk.org/api/nltk.tokenize.html) the text is split into a list of individual sentences.
# 
# In order to learn a word or entity embedding from text with [Gensim's Word2Vec libary](https://radimrehurek.com/gensim/models/word2vec.html), the text is needed to be loaded and organized into sentences. [PathLineSentence](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.PathLineSentence) is used. PathLineSentence processes all files in a directory in alphabetical order by filename and each file contains one sentence per line. 
# 
# That is why the preprocessed input text is stored in the inputList raw or inputList entity dictionary, in which each file contains a single sentence per line. 


# return the final inputlist
def buildModelList(scr, texttype, lowercase, targetdir, german, italian, french, spanish):
    #
    #
    #
    sentence_list = []
    subfolders = [f.path for f in os.scandir(scr) if f.is_dir() ]
    for d in subfolders:
        finallist = []
        print("processing files in path " + d)
        files = [f.path for f in os.scandir(d) if not f.is_dir() ]
        for f in files:
            print(f)
            # get text from article
            original_output_text = Wiki_Extractor(f, texttype)
            # prepare the text for further processing
            output_text = processText(original_output_text, german, italian, french, spanish)
            
            # split text to individual sentences
            sentences_list = nltk.sent_tokenize(output_text)
            #sentences_list = output_text.split('.')  #split the raw text into a list of sentences
            
            sentences_list = fixUnwantedBreak(sentences_list)
            
            for sentence in sentences_list:
                if lowercase:
                    sentence = sentence.lower()
                s = sentence.strip()
                l = len(s)
                if l > 0:
                    if s[l-1] == ".":
                        s = s[0:l-1]
                    #append only words with lenght > 0
                    tmpList = []
                    for w in s.split():
                        if len(w) > 0 :
                            tmpList.append(w)
                    finallist.append(tmpList)
        path = d.split("\\")
        partlistname = targetdir + "\\" + path[-1] + "_list"
        print("Save partlist " + partlistname)
        saveModelList(partlistname, finallist)


# ## Configuartion and Main 

# configuration
parser = argparse.ArgumentParser(description='Script for preprocessing wikipedia corpus')
parser.add_argument('source', type=str, help='source file')
parser.add_argument('target', type=str, help='target directory name to store corpus in')
parser.add_argument('acListName', type=str, help="name of a file holding acronyms")
parser.add_argument('-e', '--entity', action='store_true', help='process entity text')
parser.add_argument('-l', '--lowercase', action='store_true', help='lower casing text')
parser.add_argument('-ger', '--german', action='store_true', help='preproccesing german wikipedia')
parser.add_argument('-it', '--italian', action='store_true', help='preproccesing italian wikipedia')
parser.add_argument('-fr', '--french', action='store_true', help='preproccesing french wikipedia')
parser.add_argument('-es', '--spanish', action='store_true', help='preproccesing spanish wikipedia')


args = parser.parse_args()

#start recording time
startTime = time.time()

#read AcronymList
readAcronymList(args.acListName)

# if  no -e then args.entity= false and raw is done
buildModelList(args.source, args.entity, args.lowercase, args.target, args.german, args.italian, args.french, args.spanish)

#print building model time
flTime = time.time()
print("building model input takes ", flTime-startTime, " seconds")


