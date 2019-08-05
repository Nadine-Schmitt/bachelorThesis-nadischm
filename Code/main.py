
import gensim
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd
import pickle
import time

from gensim.models import Word2Vec
from gensim.models import FastText
from scipy.stats.stats import pearsonr
from scipy.stats import spearmanr


def cosine_similarity(x, y):
    temp = x / np.linalg.norm(x, ord=2)
    temp2 = y / np.linalg.norm(y, ord=2)
    return np.dot(temp, temp2)

def pairwise_accuracy(golds, preds):
    count_good = 0.0
    count_all = 0.0
    for i in range(len(golds) - 1):
        for j in range(i+1, len(golds)):
            count_all += 1.0
            diff_gold = golds[i] - golds[j]
            diff_pred = preds[i] - preds[j]
            if (diff_gold * diff_pred >= 0):
                count_good += 1.0
    return count_good / count_all


def underscore_creator(s): 
    w = ''
    l = s.split(' ')
    for index in range(len(l) -1): 
        w += l[index] + '_'
    w += l[-1]
    return w    

def removeBrackets(word):
    if word[0] == '(':
        word = word[1: len(word)]
    if word[len(word)-1] == ')':
        word = word[:len(word)-1]
    #print(word)
    return word

def word2vec_cosine_calculator(x, y, trained_model): 
    if x in trained_model.wv.vocab and y in trained_model.wv.vocab:
        vector_1 = trained_model[x]
        vector_2 = trained_model[y]
        cosine = cosine_similarity(vector_1, vector_2)
    else:
        cosine = 0
    
    return cosine

# get KORE-data in dictionary
def getKoreData(datafile):
    #

    lines = [line.rstrip('\n') for line in open(datafile)]
    lines[0] = 'Apple Inc.'
    lines[195] = '\tGolden Globe Award for Best Actor - Motion Picture Drama' #instead of Golden Globe Award for Best Actor â€“ Motion Picture Drama
    lines[299]= '\tKärtsy Hatakka' #instead of KÃ¤rtsy Hatakka
    lines[302]= '\tRagnarök' #instead of RagnarÃ¶k

    Kore_dict = dict()
    start_parameter = 1
    end_parameter = 21
    for i in range(21): 
        word = lines[start_parameter -1]
        Kore_dict[word] = []
    
        for k in range(start_parameter,end_parameter): 
            w = lines[k][1:]
            Kore_dict[word].append(w)
        start_parameter += 21
        end_parameter += 21
    return Kore_dict
#

### Similarity-353
def evaSim353(sim353_data, model):
    
    #read file
    data_sim353 = pd.read_table(sim353_data, header = None, names=('Col1', 'Col2', 'Col3'))

    #add cosine similarity
    data_sim353['Word2Vec_Cosine'] = data_sim353[['Col1','Col2']].apply(lambda row: word2vec_cosine_calculator(*row, model), axis=1)
    data_return = data_sim353['Word2Vec_Cosine']

    #calculate pearson, spearman
    pearson_sim353 = pearsonr(data_sim353['Col3'], data_sim353['Word2Vec_Cosine'])
    spearman_sim353, p_value_sim353 = spearmanr(data_sim353['Col3'], data_sim353['Word2Vec_Cosine'])
    #print("Pearson Sim-353: ", pearson_sim353)
    #print("Spearman Sim-353: " , spearman_sim353, p_value_sim353)
    return [pearson_sim353[0], pearson_sim353[1], spearman_sim353, p_value_sim353], data_return


### Relatedness-353
def evaRel353(rel353_data, model):
    
    #read file
    data_rel353 = pd.read_table(rel353_data, header = None, names=('Col1', 'Col2', 'Col3'))
    #data_rel353.head()

    #add cosine similarity
    data_rel353['Word2Vec_Cosine'] = data_rel353[['Col1','Col2']].apply(lambda row: word2vec_cosine_calculator(*row, model), axis=1)
    data_return = data_rel353['Word2Vec_Cosine']   

    #calculate pearson, spearman
    pearson_rel353 = pearsonr(data_rel353['Col3'], data_rel353['Word2Vec_Cosine'])
    spearman_rel353, p_value_rel353 = spearmanr(data_rel353['Col3'], data_rel353['Word2Vec_Cosine'])
    #print("Pearson Rel-353: ", pearson_rel353)
    #print("Spearman Rel-353: " , spearman_rel353, p_value_rel353)
    return [pearson_rel353[0], pearson_rel353[1], spearman_rel353, p_value_rel353], data_return


### MEN
def evaMen(men_data, model):
    
    #read file
    data_men = pd.read_table(men_data, sep = " ", header = None, names=('Col1', 'Col2', 'Col3'))
    #print(data_men.head())

    #add cosine similarity
    data_men['Word2Vec_Cosine'] = data_men[['Col1','Col2']].apply(lambda row: word2vec_cosine_calculator(*row, model), axis=1)
    data_return = data_men['Word2Vec_Cosine']
    #data_men

    #calculate pearson, spearman
    pearson_men = pearsonr(data_men['Col3'], data_men['Word2Vec_Cosine'])
    spearman_men, p_value_men = spearmanr(data_men['Col3'], data_men['Word2Vec_Cosine'])
    #print("Pearson MEN: ", pearson_men)
    #print("Spearman MEN: " , spearman_men, p_value_men)
    return [pearson_men[0], pearson_men[1], spearman_men, p_value_men], data_return

###RG65
def evaRG65(RG65_data, model):

    #read file
    data_rg65 = pd.read_table(RG65_data, sep = ";", header = None, names=('Col1', 'Col2', 'Col3'))
    #print(data_rg65)

    #add cosine similarity
    data_rg65['Word2Vec_Cosine'] = data_rg65[['Col1','Col2']].apply(lambda row: word2vec_cosine_calculator(*row, model), axis=1)
    data_return = data_rg65['Word2Vec_Cosine']
    #data_men

    #calculate pearson, spearman
    pearson_rg65 = pearsonr(data_rg65['Col3'], data_rg65['Word2Vec_Cosine'])
    spearman_rg65, p_value_rg65 = spearmanr(data_rg65['Col3'], data_rg65['Word2Vec_Cosine'])
    #print("Pearson RG65: ", pearson_rg65)
    #print("Spearman RG65: " , spearman_rg65, p_value_rg65)
    return [pearson_rg65[0], pearson_rg65[1], spearman_rg65, p_value_rg65], data_return


### MTurk
def evaMTurk(MTurk_data, model):

    #read file
    data_mturk = pd.read_table(MTurk_data, sep = ",", header= None, names=('Col1', 'Col2', 'Col3'))
    #print(data_mturk.head())

    #add cosine similarity
    data_mturk['Word2Vec_Cosine'] = data_mturk[['Col1','Col2']].apply(lambda row: word2vec_cosine_calculator(*row, model), axis=1)
    data_return = data_mturk['Word2Vec_Cosine']
    #data_mturk

    #calculate pearson, spearman
    pearson_mturk = pearsonr(data_mturk['Col3'], data_mturk['Word2Vec_Cosine'])
    spearman_mturk, p_value_mturk = spearmanr(data_mturk['Col3'], data_mturk['Word2Vec_Cosine'])
    #print("Pearson MTurk: ", pearson_mturk)
    #print("Spearman MTurk: " , spearman_mturk, p_value_mturk)
    return [pearson_mturk[0], pearson_mturk[1], spearman_mturk, p_value_mturk], data_return

###SimLex999
def evaSimLex999(SimLex999_data, model):
    
    #read file
    data_simlex = pd.read_table(SimLex999_data, sep = "\t", header = None, names=('word1', 'word2', 'POS', 'SimLex999', 'conc(w1)', 'conc(w2)', 'concQ', 'Assoc(USF)', 'SimAssoc333', 'SD(SimLex)'))
    #print(data_simlex)

    #add cosine similarity
    data_simlex['Word2Vec_Cosine'] = data_simlex[['word1','word2']].apply(lambda row: word2vec_cosine_calculator(*row, model), axis=1)
    data_return = data_simlex['Word2Vec_Cosine']
    #print(data_simlex)

    #calculate pearson, spearman
    pearson_simlex= pearsonr(data_simlex['SimLex999'], data_simlex['Word2Vec_Cosine'])
    spearman_simlex, p_value_simlex = spearmanr(data_simlex['SimLex999'], data_simlex['Word2Vec_Cosine'])
    #print("Pearson SimLex999: ", pearson_simlex)
    #print("Spearman Simlex999: " , spearman_simlex, p_value_simlex)
    return [pearson_simlex[0], pearson_simlex[1], spearman_simlex, p_value_simlex], data_return

###Rareword
def evaRareWord(RareWord_data, model):
    
    #read file
    data_rare = pd.read_table(RareWord_data, sep = "\t", header = None, names=('word1', 'word2', 'average', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
    #print(data_rare)

    #add cosine similarity
    data_rare['Word2Vec_Cosine'] = data_rare[['word1','word2']].apply(lambda row: word2vec_cosine_calculator(*row, model), axis=1)
    data_return = data_rare['Word2Vec_Cosine']
    #print(data_rare)

    #calculate pearson, spearman
    pearson_rare= pearsonr(data_rare['average'], data_rare['Word2Vec_Cosine'])
    spearman_rare, p_value_rare = spearmanr(data_rare['average'], data_rare['Word2Vec_Cosine'])
    #print("Pearson RareWord: ", pearson_rare)
    #print("Spearman Rareword: " , spearman_rare, p_value_rare)
    return [pearson_rare[0], pearson_rare[1], spearman_rare, p_value_rare], data_return

### evaluate KORE for RAW corpus
def evaKoreRaw(kore_dict, model, modelSize):
    #
    #
    standard_ranking = [i for i in range(1, 21)] #create list with ranking from 1 to 20
    counter = 0
    correlation_spearman = 0
    correlation_pairwise = 0
    for key, value in kore_dict.items(): #loop through dictionary
        key_list = key.split(' ')
        main_vector = np.zeros(modelSize)
        for index in range(len(key_list)): 
            key_list[index] = removeBrackets(key_list[index])
            main_vector += model[key_list[index]]

        vector_dict = dict()
        vector_list = []
        final_list= []
        for word in value: #loop through the list of words  
            w_list = word.split(' ') 
            vector = np.zeros(modelSize)
            for index in range(len(w_list)):     
                w_list[index] = removeBrackets(w_list[index]) #remove brackets
                if w_list[index] in model.wv.vocab: #get the vector of the main entitiy
                    vector += model[w_list[index]]
                else: 
                    vector += np.zeros(modelSize)
                    #print(w_list[index])
            cosine = cosine_similarity(main_vector, vector) #calculate the cosine similarity between main word and connected word 
            vector_dict[word] = cosine #store similarity in a dictionary: keys like Steve Jobs and cosine to Apple as value
            vector_list.append(cosine) #list of cosine values (size 20)
        vector_list.sort() #sort list
        if len(vector_list) != len(set(vector_list)): #check if the list has duplicates, then a warning is printed 
            print('WARNING!!!!!!')
        for word in value: 
            k = vector_dict[word] #getting the cosine value according to the ranking
            rank = vector_list.index(k) +1 #get rank in the list  
            final_list.append(rank) #append rank to fianal list
        spearman, p_value = spearmanr(standard_ranking, final_list) #calculate spearman
        pairwise = pairwise_accuracy(standard_ranking, final_list) #calculate pairwise accuracy
        #print('Spearman:', spearman, p_value, 'Pairwise Accuracy', pairwise)
        correlation_spearman += spearman
        correlation_pairwise += pairwise
        counter += 1
    final_corr_spearman = correlation_spearman/counter  
    final_corr_pairwise = correlation_pairwise/counter  
    #print('Spearman', final_corr_spearman, 'Pairwise Accuracy', final_corr_pairwise)
    return [final_corr_spearman, final_corr_pairwise]
    

### evaluate KORE for Entity corpus
def evaKoreEntity(kore_dict, model, modelSize):
    #
    #
    #
    standard_ranking = [i for i in range(1, 21)] #create list with ranking from 1 to 20
    counter = 0
    correlation_spearman = 0
    correlation_pairwise = 0
    for key, value in kore_dict.items(): #loop through dictionary
        main_vector = model[underscore_creator(key)]
        #print("entity: " ,underscore_creator(key))
        
        vector_dict = dict()
        vector_list = []
        final_list= []
        for word in value: #loop through the list of words  
            w = underscore_creator(word) #connect words with underscores
            if w in model.wv.vocab: #get vector if each word 
                vector = model[w]
            else: 
                vector = np.zeros(modelSize)
                #print(w)
            cosine = cosine_similarity(main_vector, vector) #calculate the cosine similarity between main word and connected word 
            vector_dict[w] = cosine #store similarity in a dictionary: keys like Steve Jobs and cosine to Apple as value
            vector_list.append(cosine) #list of cosine values (size 20)
        vector_list.sort() #sirt list
        if len(vector_list) != len(set(vector_list)): #check if the list has duplicates, then a warning is printed 
            print('WARNING!!!!!!')
        for word in value: 
            k = vector_dict[underscore_creator(word)] #getting the cosine value according to the ranking
            rank = vector_list.index(k)+1 #get rank in the list  
            final_list.append(rank) #append rank to fianal list
        spearman, p_value = spearmanr(standard_ranking, final_list) #calculate spearman
        pairwise = pairwise_accuracy(standard_ranking, final_list) #calculate pairwise accuracy
        #print('Spearman:', spearman, p_value, 'Pairwise Accuracy', pairwise)
        correlation_spearman += spearman
        correlation_pairwise += pairwise
        counter += 1
    final_corr_spearman = correlation_spearman/counter  
    final_corr_pairwise = correlation_pairwise/counter  
    #print('Spearman', final_corr_spearman, 'Pairwise Accuracy', final_corr_pairwise)
    return [final_corr_spearman, final_corr_pairwise]


### calculate Pearson between RAW/Entity list
def evaRawEntity(rList, eList):
    pearson = pearsonr(rList, eList)
    return pearson[0]

def para2text(sizeI, windowI, min_countI, sgI, hsI, negativeI, cbowmeanI):
    para = str(sizeI) + "/" + str(windowI) + "/" + str(min_countI) + "/" + str(sgI) + "/" + str(hsI) + "/" + str(negativeI) + "/" + str(cbowmeanI)
    return para

def result2text(result, pearsonRE):
    rText = "{0:18.16f};{1:18.16e};{2:18.16f};{3:18.16e};{4:18.16f}".format(*result, pearsonRE)
    return rText

def result2TextShort(result):
    rText = "{0:18.16f};{1:18.16f}".format(*result)
    return rText


# print results
def printResult(modelType, modelP, results1, results2, results3, result4, result5, result6, result7, result8):
    print(modelType,  modelP , results1, results2, results3, result4, result5, result6, result7, result8, sep=";")

#read parameter settings and get list of lists in which each list is a parameter set
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

# configuration
startTime = time.time()

parser = argparse.ArgumentParser(description='Script for training word embeddings')
parser.add_argument('sourceRaw', type=str, help='source folder with preprocessed raw data')
parser.add_argument('sourceEntity', type=str, help='source folder with preprocessed entity data')
parser.add_argument('goldData', type=str, help='directory where to find the gold lists for the evaluation models')
parser.add_argument('iterations', type=int, default=1, help='how often train model for each parameter set')
parser.add_argument('paraList', type=str, help='source folder of paraList')

parser.add_argument('-t', '--threads', type=int, default=mp.cpu_count(), help='number of worker threads to train the model')
args = parser.parse_args()


# Load gold standards for word related tasks
print(" define goldstandard and load model input ")
sim353_datafile = args.goldData + r"\wordsim_similarity_goldstandard.txt"
rel353_datafile = args.goldData + r"\wordsim_relatedness_goldstandard.txt"
men_datafile = args.goldData + r"\MEN_dataset_natural_form_full.txt"
rg65_datafile = args.goldData + r"\RG65.txt"
mturk_datafile = args.goldData + r"\MTurk.csv"
simlex999_datafile = args.goldData + r"\SimLex-999.txt"
rareword_datafile = args.goldData + r"\RareWord.txt"
kore_datafile = args.goldData + r"\Kore_entity_relatedness_processed.txt"

#using PathLineSentence
sentencesRaw = gensim.models.word2vec.PathLineSentences(args.sourceRaw)
sentencesEntity = gensim.models.word2vec.PathLineSentences(args.sourceEntity)

#Loading Kore dataset for the entity related task
kore_dict = getKoreData(kore_datafile)


#set parameter
paraList = readParaSetting(args.paraList)

#set number of iterations
iterations = args.iterations


result = []

for e in paraList:
    sgI = int(e[3])
    if sgI == 0:
        cbowmeanP = [0, 1]
    else:
        cbowmeanP = [1]
    sizeI = int(e[0]) 
    windowI = int(e[1])
    min_countI = int(e[2])
    hsI = int(e[4])
    negativeI = int(e[5])
    for cbowmeanI in cbowmeanP:
        lauf = 0
        sim353ResultR = [0, 0, 0, 0]
        sim353ResultE = [0, 0, 0, 0]
        sim353RE = 0
        rel353ResultR = [0, 0, 0, 0]
        rel353ResultE = [0, 0, 0, 0]
        rel353RE = 0
        menResultR = [0, 0, 0, 0]
        menResultE = [0, 0, 0, 0]
        menRE = 0
        rg65ResultR = [0, 0, 0, 0]
        rg65ResultE = [0, 0, 0, 0]
        rg65RE = 0
        MTurkResultR = [0, 0, 0, 0]
        MTurkResultE = [0, 0, 0, 0]
        MTurkRE = 0
        simlex999ResultR = [0, 0, 0, 0]
        simlex999ResultE = [0, 0, 0, 0]
        simlex999RE = 0
        RareWordResultR = [0, 0, 0, 0]
        RareWordResultE = [0, 0, 0, 0]
        RareWordRE = 0
                            
        KoreResultR = [0, 0]
        KoreResultE = [0, 0]
        # training and evaluation is done #iteration-times and performance is averaged
        while lauf < iterations:
            # build the model
            print("Lauf ", lauf)
            lauf += 1
            #mRStart = time.time()
            modelRaw  = Word2Vec(sentencesRaw,
                                size=sizeI,
                                window=windowI,
                                min_count=min_countI,
                                workers=args.threads,
                                sg=sgI,
                                hs=hsI,
                                negative=negativeI,
                                cbow_mean=cbowmeanI)

            #mREnd = time.time()

            #mEStart = time.time()
            modelEntity  = Word2Vec(sentencesEntity,
                                size=sizeI,
                                window=windowI,
                                min_count=min_countI,
                                workers=args.threads,
                                sg=sgI,
                                hs=hsI,
                                negative=negativeI,
                                cbow_mean=cbowmeanI)
            #mEEnd = time.time()
                            
            #print("Raw    Model training time : ", mREnd - mRStart)
            #print("Entity Model training time : ", mEEnd - mEStart)

            #Kore Test
            result = evaKoreRaw(kore_dict, modelRaw, sizeI)
            for i in range(len(KoreResultR)):
                KoreResultR[i] += result[i]
            result = evaKoreEntity(kore_dict, modelEntity, sizeI)
            for i in range(len(KoreResultE)):
                KoreResultE[i] += result[i]
                               
            result, cosineRaw  = evaSim353(sim353_datafile, modelRaw)
            for i in range(len(sim353ResultR)):
                sim353ResultR[i] += result[i]
            result, cosineEntity = evaSim353(sim353_datafile, modelEntity)
            for i in range(len(sim353ResultE)):
                sim353ResultE[i] += result[i]
            sim353RE =+ evaRawEntity(cosineRaw, cosineEntity)

            result, cosineRaw  = evaRel353(rel353_datafile, modelRaw)
            for i in range(len(rel353ResultR)):
                rel353ResultR[i] += result[i]
            result, cosineEntity = evaRel353(rel353_datafile, modelEntity)
            for i in range(len(rel353ResultE)):
                 rel353ResultE[i] += result[i]
            rel353RE =+ evaRawEntity(cosineRaw, cosineEntity)
                                
            result, cosineRaw  = evaMen(men_datafile, modelRaw)
            for i in range(len(menResultR)):
                menResultR[i] += result[i]
            result, cosineEntity = evaMen(men_datafile, modelEntity)
            for i in range(len(menResultE)):
                 menResultE[i] += result[i]
            menRE =+ evaRawEntity(cosineRaw, cosineEntity)
                                
            result, cosineRaw  = evaRG65(rg65_datafile, modelRaw)
            for i in range(len(rg65ResultR)):
                rg65ResultR[i] += result[i]
            result, cosineEntity = evaRG65(rg65_datafile, modelEntity)
            for i in range(len(rg65ResultE)):
                rg65ResultE[i] += result[i]
            rg65RE =+ evaRawEntity(cosineRaw, cosineEntity)
                                
            result, cosineRaw  = evaMTurk(mturk_datafile, modelRaw)
            for i in range(len(MTurkResultR)):
                MTurkResultR[i] += result[i]
            result, cosineEntity = evaMTurk(mturk_datafile, modelEntity)
            for i in range(len(MTurkResultE)):
                MTurkResultE[i] += result[i]
            MTurkRE =+ evaRawEntity(cosineRaw, cosineEntity)

            result, cosineRaw  = evaSimLex999(simlex999_datafile, modelRaw)
            for i in range(len(simlex999ResultR)):
                simlex999ResultR[i] += result[i]
            result, cosineEntity = evaSimLex999(simlex999_datafile, modelEntity)
            for i in range(len(simlex999ResultE)):
                simlex999ResultE[i] += result[i]
            simlex999RE =+ evaRawEntity(cosineRaw, cosineEntity)

            result, cosineRaw = evaRareWord(rareword_datafile, modelRaw)
            for i in range(len(RareWordResultR)):
                RareWordResultR[i] += result[i]
            result, cosineEntity = evaRareWord(rareword_datafile, modelEntity)
            for i in range(len(RareWordResultE)):
                RareWordResultE[i] += result[i]
            RareWordRE =+ evaRawEntity(cosineRaw, cosineEntity)
                                    
        #calc average results
        for i in range(4):
            sim353ResultR[i] = sim353ResultR[i] / iterations
            sim353ResultE[i] = sim353ResultE[i] / iterations
            sim353RE = sim353RE / iterations
                                
            rel353ResultR[i] = rel353ResultR[i] / iterations
            rel353ResultE[i] = rel353ResultE[i] / iterations
            rel353RE = rel353RE / iterations
                                
            menResultR[i] = menResultR[i]  / iterations
            menResultE[i] = menResultE[i]  / iterations
            menRE = menRE / iterations
                                
            rg65ResultR[i] = rg65ResultR[i] / iterations
            rg65ResultE[i] = rg65ResultE[i] / iterations
            rg65RE = rg65RE / iterations
                                
            MTurkResultR[i] = MTurkResultR[i] / iterations
            MTurkResultE[i] = MTurkResultE[i] / iterations
            MTurkRE = MTurkRE / iterations
                                
            simlex999ResultR[i] = simlex999ResultR[i] / iterations
            simlex999ResultE[i] = simlex999ResultE[i] / iterations
            simlex999RE = simlex999RE / iterations
                                
            RareWordResultR[i] = RareWordResultR[i] / iterations
            RareWordResultE[i] = RareWordResultE[i] / iterations
            RareWordRE = RareWordRE / iterations
        for i in range(2):  
            KoreResultR[i] = KoreResultR[i] / iterations
            KoreResultE[i] = KoreResultE[i] / iterations
                                
        #prepare printout
        mPara = para2text(sizeI, windowI, min_countI, sgI, hsI, negativeI, cbowmeanI)
        r1 = result2text(sim353ResultR, sim353RE)
        r2 = result2text(rel353ResultR, rel353RE)
        r3 = result2text(menResultR, menRE)
        r4 = result2text(rg65ResultR, rg65RE)
        r5 = result2text(MTurkResultR, MTurkRE)
        r6 = result2text(simlex999ResultR, simlex999RE)
        r7 = result2text(RareWordResultR, RareWordRE)
        r8 = result2TextShort(KoreResultR)
        e1 = result2text(sim353ResultE, sim353RE)
        e2 = result2text(rel353ResultE, rel353RE)
        e3 = result2text(menResultE, menRE)
        e4 = result2text(rg65ResultE, rg65RE)
        e5 = result2text(MTurkResultE, MTurkRE)
        e6 = result2text(simlex999ResultE, simlex999RE)
        e7 = result2text(RareWordResultE, RareWordRE)
        e8 = result2TextShort(KoreResultE)
        #print it
        printResult("Raw", mPara, r1,r2,r3,r4,r5,r6,r7,r8)
        printResult("Entity", mPara, e1,e2,e3,e4,e5,e6,e7,e8)
endTime = time.time()
print("total run-time", endTime - startTime)



