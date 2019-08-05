#!/usr/bin/env python
# 
coding: utf-8

# 
# Translation of the KORE dataset




# 
# This script translate automatically the original English KORE dataset into other languages.

# 
# To translate a single entity from the English KORE dataset the MediaWkiki Action API(https://www.mediawiki.org/wiki/API:Search) is used. For each English entity in the KORE dataset (which has an English Wikipedia page), the corresponding Wikipedia page in the target language is searched and then taken as translated entity.
# 



# ## Import

# 



import requests

import argparse




 ## Functions

# 

With following function the original English KORE datatset is loaded:




def readFile(source):
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
    return list




# The writeFile-function writes the translated entities (which are in the list) to the target file:





def writeFile(target, list):
    
    with open(target, 'w', encoding="UTF-8") as f:
        for name in list:
            f.writelines(name)
            f.writelines("\n")




# ## Configuration and Main
# 




parser = argparse.ArgumentParser(description='Script for translating wikipedia page names')
parser.add_argument('source', type=str, help='source folder with the dataset')
parser.add_argument('target', type=str, help='target folder in which to store the translated data')
parser.add_argument('lang', type=str, help='language in which to translate to')

args = parser.parse_args()

#read file into list
pageList = readFile(args.source)
#print(pageList)

#set language in which the datatset should be translated to
LANG = str(args.lang)
#print(LANG)

#empty list, where the translated entities will be stored in
pageTranslated = []

for e in pageList:
    
    S = requests.Session()
    
    #set URL to the wikipedia in which language on like to translate to; here german: de
    #it: italian, es: spanish, fr: french
    URL = "https://" + LANG + ".wikipedia.org/w/api.php"
    #print(URL)
    
    SEARCHPAGE = LANG + ":" + e

    PARAMS = {
        'action':"query",
        'list':"search",
        'srsearch': SEARCHPAGE,
        'format':"json"
    }

    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    
    
    pageTranslated.append(str((DATA['query']['search'][0]['title'])))
    
#write list into file
writeFile(args.target, pageTranslated)
print("Translation successfully saved into " + args.target)





# ## Example 
# Translate English entity _European Commission_ into German entity _Europäische Kommission_:





#test

S = requests.Session()
    
#set URL to the wikipedia in which language on like to translate to: here german: de
URL = "https://de.wikipedia.org/w/api.php"
    
SEARCHPAGE = 'de:European Commission'
    
    
PARAMS = {
        'action':"query",
        'list':"search",
        'srsearch': SEARCHPAGE,
        'format':"json"
    }

R = S.get(url=URL, params=PARAMS)
DATA = R.json()
    

print(str((DATA['query']['search'][0]['title'])))


