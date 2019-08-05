# The impact of entity annotations on the word embedding training process

In my bachelor thesis I trained embeddings from raw text (word embeddings) and from entity annotated text (entity embeddings) with [Gensim's Word2Vec libary](https://radimrehurek.com/gensim/models/word2vec.html) and evaluated them with word related tasks and entity tasks afterwards in order to answer following question:

                     Do entity annotations have an impact on the word embedding training process?
                     
An extensive parameter tuning is peformed and for the best parameters the results are checked with other algorithms of word embeddings using [Gensim's FastText libary](https://radimrehurek.com/gensim/models/fasttext.html) and other languages (German, Italian, Spanish and French).

In the following figure an overview of the implementation is given:

![flow2](https://user-images.githubusercontent.com/48829194/62204597-ccd85100-b38d-11e9-97df-d09e76e18ba1.PNG)

For training, the entire [English Wikipedia dump](https://dumps.wikimedia.org/enwiki/) (and [German Wikipedia dump](https://dumps.wikimedia.org/dewiki/), [Italian Wikipedia dump](https://dumps.wikimedia.org/itwiki/), [Spanish Wikipedia dump](https://dumps.wikimedia.org/eswiki/), [French Wikipedia dump](https://dumps.wikimedia.org/frwiki/) respectively) is used.
Firstly the Wikipedia dump is downloaded and then the [WikiExtractor for WikiMentions](https://github.com/samuelbroscheit/wikiextractor-wikimentions) is used in order to convert the downloaded Bz2-files into several files of similar size in a given directory. Each file contains several documents in a given document format. These are the input for the preprocessing, in which the text is prepared for the training of the word and entity embeddings.  An ``inputList_raw`` and ``inputList_entity`` is outputed, which are files, in which each row contains one sentence. They are used as input corpuses for the training of the word and entity embeddings with [Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html). Afterwards, the models are evaluated with different evaluation tasks (by using Pearson correlation, Spearman correlation and Pairwise Accuracy as evaluation metrics).The results are compared with each other and to find out, if the Pearson and Spearman correlations are statistical significant, the p-value is calculated. To compare two Pearson correlations the [cocor package in R](https://cran.r-project.org/web/packages/cocor/cocor.pdf) is used. The dataset for the entity evaluation task (KORE dataset) is only available in English, and therefore it is translated into the other languages.

## Download Wikipedia dump
To download the Wikipedia dump, a directory have to be created with
```markdown
mkdir wikidump
```
and then the dump is downloaded with following command (where the URL is the filename of the wikidump):
```markdown
wget "https://dumps.wikimedia.org/enwiki/20190201/enwiki-20190201-pages-articles-multistream.xml.bz2"
```

## Extract Wikipedia dump
Downloading the entire Wikipedia dump just gives a Bz2-file and the texts from the Wikipedia database dump has to be extracted and cleaned by the [WikiExtractor.py](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/WikiExtractor.py), which is a [Python script](https://github.com/attardi/wikiextractor). The extraction is done with the [WikiExtractor for WikiMentions](https://github.com/samuelbroscheit/wikiextractor-wikimentions) from Samuel Broscheit, which is a modified version of the WikiExtractor with the additional option to extract the internal Wikipedia links from an article. To do so, the [WikiExtractor.py](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/WikiExtractor.py) is downloaded. Then following command
```markdown
python ~/bin/WikiExtractor.py --json --filter_disambig_pages --processes 4 --collect_links /data/wikidump/enwiki-20190201-pages-articles-multistream.xml.bz2 -o /data/wikiExtracted
```
is run in order that each articles dictionary contains an additional field ``internal_links``. Running the command for the [English Wikipedia dump](https://dumps.wikimedia.org/enwiki/), 5,669,083 articles are extracted. As result the /data/wikiExtracted directory has a size of 21GB and 213 subfolders (from AA to IE), which each has a size of 98MB and contains 100 files (from wiki_00 to wiki_99).

## Preprocessing

The preprocessing is done for each downloaded and extracted Wikipedia dump (i.e. for each language) twice in order to get a ``raw input corpus`` and an ``entity annotated corpus``. The [preprocessing.py](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/preprocessing.py) script can be run by calling the command
```markdown
python preproccesing.py WikiExtracted inputList_raw AcronymList -ger
```
for example for the ``german raw inputList`` and
```markdown
python preproccesing.py WikiExtracted inputList_entity AcronymList -e -ger
```
for the ``german entity inputList``. So, the ``extracted Wikipedia dump`` is needed as input, as well as an [AcronomyList](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/AcronymLists). Moreover, it must also be specified if a need arises for a raw or entity corpus (with parameter -e) and the decision of which language is used (``ger`` for German, ``it`` for Italian, ``es`` for Spanish and ``fr`` for French}. Besides, the output is stored in the ``inputList_raw`` or ``inputList_entity`` directory. Optionally it is also possible to lower casing the input text (setting ``-l`` flag), however, better word and entity embeddings are always achived when not lower casing the input corpora, why it is not applied.

Note, that for splitting the text into a list of individual sentences (Word2Vec requires text, which is organized into sentences, as input) [nltk.sent_tokenize utility](https://www.nltk.org/api/nltk.tokenize.html) is used and have to be imported beforehand by running following commands:
```markdown
mkdir nltk_data
python
import nltk
nltk.download('punkt')
```
Besides, sometimes sentences are unwanted broken by [nltk.sent_tokenize utility](https://www.nltk.org/api/nltk.tokenize.html), since there can be a dot without the sentence ending, when there is an acronym (Inc. for instance). If there is a dot and the next sentence starts with an uppercase letter, then it is a new sentence. In the other case, the next sentence is appended to the current sentence. However, there are examples like _i.e. Germany_, where after a dot there is an uppercase letter, but the sentence should not be split. Therefore an [AcronomyList](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/AcronymLists), which contains all acronyms with a dot of a given language, is used. If _i.e_  is for example in the [AcronomyList](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/AcronymLists) then the unwanted broken sentences are combined.

See the [preprocessing ipython notebook](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/Preprocessing.ipynb) for more details.


## Training and Evaluation

###

For training word and entity embeddings and evaluating them the [main.py](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/main.py) script is used (see the [main ipython notebook](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/main.ipynb)). It can be run by typing following command:
```markdown
python main.py inputList_raw inputList_entity goldstandard 3 Parameter.txt -t 16
```
where the ``inputList_raw`` and  ``inputList_entity`` are the preprocessed input corpuses, ``goldstandard`` is the directory with all the [gold standard datastes](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/data) for evaluation (e.g. Similarity353 dataset),  ``3`` specifies that three training iterations are done. ``Parameter.txt`` is the parameter setting, which should be applied and ``16 threads`` are used for training.
Note, that for each parameter setting a raw and an entity model is trained and evaluated directly afterwords, because the models are to big to be saved on a disk and reloaded on a later point in time.

### Training with Word2Vec
In order to use [Gensim's Word2Vec libary](https://radimrehurek.com/gensim/models/word2vec.html), Gensim have to be installed with following commands:
```markdown
sudo chmod -R 777 bin
easy_install --upgrade gensim
```
For working with the Word2Vec model a ``Word2Vec class`` is provided by Gensim.  In order to learn a word embedding from text, the text is needed to be loaded and organised into sentences and provided to the constructor of a new ``Word2Vec() instance``. [PathLineSentence](https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.PathLineSentence) is applied and the preprocessed input corpus is loaded as following:
```markdown
sentences = gensim.models.word2vec.PathLineSentences(inputList)
```
where ``inputList`` is the output dictionary from the [preprocessing.py](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/preprocessing.py).
Besides the training input corpus, there are also other parameters on the Word2Vec-constructor. An extensive parameter tuning is performed. The English models are trained with 151 different parameter settings, taking following into account (the number in the brackets are the settings, which are applied):

- **size**: Dimensionality of the word vectors (50,100,200,300)
- **window size**: Maximum distance between the current and predicted word within a sentence (3,5)
- **min count**: Ingnores all words with a lower frequency than this (2,5)
- **workers**: Number of how many worker threads are ued to train the model(the servers, on which the training is done, have 16 VCPUs, hence this parameter is always set to 16}
- **sg**: Training algorithm: 1 for skip-gram and 0 for CBOW (0,1)
- **hs**: If 1, hierarchical softmax will be used for model training. If 0 and negative sampling is non-zero, negative sampling will be used (0,1)
- **negative sampling**: If >0, negative sampling will be used. The number specifies how many noise words should be drawn (0,8,16)
- **CBOWMean**: Only applies, when CBOW is used. If 0, use the sum of the context word vectors and if 1 use the mean (0,1)


A new Word2Vec model is created straightforward by following python code (note that the parameter setting is the setting that leads to the best model):
```markdown
from gensim.models import Word2Vec

model = Word2Vec(sentences, size=300, window=3, min_count=5, workers=16, sg=1, hs=0, negative=16, cbow_mean=1)
````

### Training with other algorithms: FastText
Word and entity embeddings are not only trained with Word2Vec, but also with FastText. There is also a [Python Gensim implementation of FastText](https://radimrehurek.com/gensim/models/fasttext.html).
Instead of using the ``Word2Vec`` class, the ``FastText`` class is used to train a FastText model. The parameters are still the same:
```markdown
from gensim.models import FastText

model = FastText(sentences, size=300, window=3, min_count=5, workers=16, sg=1, hs=0, negative=16, cbow_mean=1)
````

### Training with other languages 
All subjects discussed above, are also applied for the training in other languages. Everything is done as for the English models, just other ``input corpuses`` and the ``translated gold standard datasets`` are used. These models are only trained for the best parameter settings from the English models, hence, no intensive parameter tuning is done for the cross-lingual languages.  

### Evaluation 
After training, the word and entity embeddings are directly evaluated with following evaluation tasks: 

#### Word related evaluation task
The word related task is based on the idea that the similarity between two words can be measured with the cosine similarity of their word embeddings. A list of word pairs along with their similarity rating, which is judged by human annotators, is used by this task and the following [gold standards](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/data) are used. Note that not all datasets are available in all languages:

- Similarity353 (English, German, Italian)
- Relatedness353 (English, German, Italian)
- MEN (English)
- RG65 (English, German, Italian, Spanish, French)
- MTurk (English)
- SimLex999 (English, German, Italian, Spanish, French)
- RareWord (English)

The evaluation task is to measure how well the notion of word similarity according to human annotators is captured by the word embeddings. In other words, the distances between words in an embedding space can be evaluated through the human judgments on the actual semantic distances between these words. Once the cosine similarity between the words is computed, the two obtained distances are then compared with each other using Pearson or Spearman correlation. The more similar they are (i.e. Pearson or Spearman score is close to 1), the better are the embeddings. 

#### Entity evaluation task

The [KORE dataset](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/) is used as entity task. The dataset contain a total of 441 entities. There are 21 seed entities and for each seed there is a ranking of 20 candidate entities, which are linked to by the Wikipedia article of the seed. Example seed entities and KORE gold standard ranks of related entities are shown in the following table:

| Seed | Related entity (rank) |
| ---- | --------------------- |
| Apple Inc. | Steve Jobs (1), Steve Wozniak (2) ... NeXT (10), Safari (web browser) (11) ...Ford Motor Company (20) |
| Johnny Depp | Pirates of the Carribbean (1), Jack Sparrow (2) ...  Into the Great Wide Open (10), ... Mad Love (20) |
| GTA IV | Niko Bellic (1), Liberty City (2) ...  New York (10), Bosnian War (11) ... Mothers Against Drunk Driving (20) |
| The Sopranos | Tony Soprano (1), David Chase (2) ...  Golden Globe Award (10), The Kings (11) ... Big Love (20) |
| Chuck Norris | Chuck Norris facts (1), Aaaron Norris (2) ... Northrop Corporation (10), ... Priscilla Presley (20) |

To measure how good the produced word and entity embeddings capture the semantic relatedness between entities, the following is done: For each seed entity of the KORE dataset a ranking of the 20 candidate entities is produced by using the ``word embeddings`` and the ``entity embeddings``. When using the ``word embeddings`` the similarity between the word embedding of the seed entity and the sum of the word embeddings of the single words is measured by using cosine similarity. When using entity embeddings only the entity embeddings of the given entities have to be considered. To illustarte this, a short example is given:
Let Google be the seed entity with three candidate entities ranked as follows:
```markdown
Google
	Larry Page (1)
	Sergey Brin (2)
	Google Maps (3)
```
For the word embeddings, the similarity of the word embedding ``Google`` with the word embedding of ``Larry`` plus the word embedding of ``Page`` is measured. When considering entity embeddings, the similarity of the entity embedding ``Google`` with the entity embedding of ``Larry_Page`` is measured. Afterwards, a ranking based on the similarity score is produced, for instance
\begin{lstlisting}[language=Python]
```markdown
Google	
	Google Maps (1)
	Larry Page (2)
	Sergey Brin (3)
```
Then the quality of the correlation between the gold ranking and the produced ranking is measured in terms of Spearman correlation and Pairwise Accuracy. As result, for each entity seed a Spearman and Pairwise Accuracy score is provided. Finally, it is averaged and for each method a final value is reported.

Due to the reason that the KoORE dataset is only available in English, it is translated into German, Italian, Spanish and French. 

## Translation of KORE dataset

The Kore dataset is automatically translated into German, Italian, Spanish and French. It is done by the [TranslateWikipageNames.py](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/TranslateWikipageNames.py) script and can be run by calling the command:
```markdown 
python translationWikipageNames.py KoreDataset.txt KoreDataSetTranslated.txt de
```
where the ``KoreDataset.txt`` is the source folder of the original English KORE dataset, ``KoreDataSetTranslated.txt`` is the target folder in which to store the translated data and ``de`` is the language in which one would like to translate to (``de``: German ``it``: Italian, ``es``: Spanish and ``fr``: French).

The resulting [new datasets](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/data) are available and can be downloaded. 

To translate a single entity from the English KORE dataset the [MediaWkiki Action API](https://www.mediawiki.org/wiki/API:Search) is used. For each English entity in the KORE dataset (which has an English Wikipedia page), the corresponding Wikipedia page in the target language has to be searched and then taken as translated entity. In the following code snippet the English entity _Google_ is translated into the German entity _Google+_:

![TranslationKore](https://user-images.githubusercontent.com/48829194/62262835-6e0ce900-b41a-11e9-8408-448e33bc640b.PNG)

See the [TranslateWikipageNames ipython notebook](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/TranslateWikipageNames.ipynb) for more examples.

In case that there is no Wikipedia page in the target language available the English Wikipage name is used. 
Furthermore, some special characters (/, +, ", !, ...) are removed from the original KORE dataset, e.g. _Game of Thrones/ Staffel 7_ is set to _Game of Thrones Staffel 7_. It is done because these characters are also removed in the preprocessing process.

Remark, that there are entities in the datatset, where no corresponding embedding exists (only few occurencies of it in the input corpora for training). If it is a candidate entity, then the embedding vector is set to zero. However, the seed entity _Terremoti del XXI secolo_ is not available in the Italian entity embedding model and therefore it is removed with its 20 cadidate entities from the original dataset. Moreover, the seed entity _Deus Ex: MankindDivided_ is not available in the Spanish entity embedding model and therefore it is also removed with its 20 cadidate entities from the original dataset. Finally, as the seed entity _Sur écoute_ is not availble in the French raw embedding model and the seed entities _Quake Champions_ and _Saison 7 de Futurama_ are not available in the French entity embedding model, they are removed with their 20 candidate entities from the original dataset.


## Compare two Pearson correlations
In order to compare two Pearson correlations the [cocor package in R](https://cran.r-project.org/web/packages/cocor/cocor.pdf) is used. It can be downloaded from the [project's homepage](https://CRAN.R-project.org/package=cocor). 
The follwoing command is typed into the R console to install the cocor package in R:
```markdown
install.packages("cocor", lib= "/my/own/R-packages/")
library("cocor")
``` 
Subsequent to the above steps, the cocor package can be used to compare two Pearson correlations. It is done for a _dependent overlapping group_ by using following function in R (see the [R script](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Code/cocor.Rmd)):
```markdown
cocor.dep.groups.overlap(r.jk, r.jh, r.kh, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, data.name = NULL, var.labels = NULL, return.htest = FALSE)
```
where following arguments as input are required: 
- **r.jk** is a number specifying the correlation between j and k (this correlation is used for comparison). 
- **r.jh** is a number specifying the correlation between j and h (this correlation is used for comparison).
- **r.kh** is a number specifying the correlation between k and h.
- **n** is an integer defining the size of the group.
- **alternative** is a character string specifying whether the alternative hypothesis is two-sided ("two.sided"; default) or one-sided ("greater" or "less", depending on the direction).
- **test** is a vector of character strings specifying the tests (pearson1898, hotelling1940, hendrickson1970, williams1959, olkin1967, dunn1969, steiger1980, meng1992, hittner2003, or zou2007) to be used. With "all" all tests are applied (default).
- **alpha** is a number defining the alpha level for the hypothesis test. The default value is 0.05.
- **conf.level** is a number defining the level of confidence for the confidence interval (if test meng1992 or zou2007 is used). The default value is 0.95.
- **null.value** is a number defining the hypothesized difference between the two correlations used for testing the null hypothesis. The default value is 0. If the value is other than 0, only the test zou2007 that uses a confidence interval is available.
- **data.name** is a character string giving the name of the data/group.
- **var.labels** is a vector of three character strings specifying the labels for j, k, and h (in this order).
- **return.htest** is a logical indicating whether the result should be returned as a list containing a list of class htest for each test. The default value is FALSE.

Illustrating this, an example of the comparison between the two Pearson scores for Similarity353 for the best models with parameter setting (300,3,5,1,0,16) is shown in the following. As output from the training and evaluation a Pearson score of 0.786 for the raw model and 0.793 for the entity embedding is the result. As also the intercorrelation between the two correlations is needed as input parameter, the correlation between the cosine similarities of the raw model with the cosine similarities of the entity model is computed and given as 0.012. Besides, the Similaritym353 dataset has a size of 203 instances. Therefore following need to be typed in to the R command line in order to compare the two Pearson correlations:
```markdown
cocor.dep.groups.overlap(r.jk= 0.786, r.jh= 0.793, r.kh= 0.012, n=203, alternative="two.sided", alpha=0.05, conf.level=0.95, null.value=0)
````
As output all results of the tests are shown and the null hypothesis is for this example always retained:

![OutputCocot](https://user-images.githubusercontent.com/48829194/62342257-86e2d080-b4e6-11e9-8685-94fb930be027.PNG)

All the calculated results can be seen on the [excel files].

## Further analysis

### Frequency of words
In order to calculate for the entities of the Kore dataset the number of occurence in the ``raw`` and ``entity input corpus`` the CountWords_Raw.py and CountWords_entity.py scripts are used and can be run with following command:
```markdown
python CountWords_Raw.py Kore.txt inputList_raw
```
for the ``raw model`` and 
```markdown
python CountWords_Entity.py Kore.txt inputList_entity
```
for the ``entity model``. ``Kore.txt`` is the Kore dataset in the corresponding language and the second argument is the ``input corpus``. The fuctionality is quite simple: For each entity from the Kore datatset the number of occurence is calculated and printed as output. The results can be seen in the [excel sheets](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/Results/FrequencyWords).

### Qualitative examination

A qualitative examination is done by using the SemanticArithmetic.py script. It can be run with following command:
```markdown
python SemanticArithmetic.py inputList_raw Parameter.txt -t 16
``` 
where `inputList_raw` is the input corpus for training, `Parameter.txt` the parameters for which the training should be done and `16 threads` are used. 
The [Principal Component Analysis](https://en.wikipedia.org/wiki/Principal_component_analysis) is used in this script and it is a method to reduce the number of dimensions of highdimensional vectors, while keeping main features (= the principal components). Therefore firstly the model is trained with the specified parameters from `Parameter.txt` and afterwards, the high dimensions of the vectors (e.g. 300) are reduced to a two-dimensional representation and plotted with [pythons matplotlib](https://matplotlib.org) for some word classes, e.g. countries and their their correspondig languages:

![PCALanguage](https://user-images.githubusercontent.com/48829194/62262257-52084800-b418-11e9-9f79-1116f4e69eb9.png)

In the figure above, the countries and languages are grouped correctly. The connecting lines are approximately parallel and of the same length. So the concept of capitals and languages is understood by the word embedding model.

In addition, the [most_similar() function](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.WordEmbeddingsKeyedVectors.most_similar) of gensim is used to show related words to a given word, e.g. cucumber:

![SemanticArithmetic_Cucumber](https://user-images.githubusercontent.com/48829194/62340643-8d6e4980-b4e0-11e9-827e-84cd27e2b8f2.PNG)


## Results

### Training times
Downloading the ``English wikipedia dump`` took about 2 hours and 2.22 hours to extract it. 2.13 hours were taken by preprocessing the English Wikipedia Dump for the ``raw inputList``, and 2.12 hours for the ``entity inputList``. More interesting, 269.43 days in total were taken by the training of the ``English raw`` and ``entity models`` and their evaluation.

151 different parameter settings were used for training and all the results can be seen in [Results.xlsx](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/Results.xlsx).

### Best Models and best entity relatedness models
When taking the average of all spearman scores, the best score is achived with following parameter setting:
- size = 300
- item windowSize = 3
- minCount = 5
- sg =1
- hs = 0
- negative sampling = 16

The models with this parameter setting are called **best models**.
Following results are achieved (Pearson and Spearman correlations of best models with parameter setting 300/3/5/1/0/16 and the corresponding p-values in brackets. The better one is highlighted in green):

![BestModelResult](https://user-images.githubusercontent.com/48829194/62346708-62432480-b4f7-11e9-8012-dce25a7715f5.PNG)

On the ``raw model`` a better average score is achieved than on the ``entity model``, however when comparing the scores on the word related tasks the raw and entity models perform equally, which means that entity annotation has no impact on the word related tasks. This is acknowledged when looking on the [results of the cocor package](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/ResultsCocor_BestModels.xlsx). The null hypothesis, namley that pearson value 1 is equal to pearson value 2, is always retained. Furthermore, when analyzing the computed p-values of the Pearson and Spearman correlation, one can see that they are always much smaller then the conventionally used significance level of 5%, 1% and 0.1% and therefore it can be assumed that the correlations are statistically significant.

The best score on the entity task is achieved instead with follwing parameter setting:
- size = 200
- windowSize = 5
- minCount = 5
- sg = 0
- hs = 0
- negative sampling = 16
- CBOW mean = 0

The models with this parameter setting are called **best entity relatedness models**.
Following results are achieved (Pearson and Spearman correlations of best entity relatedness models with parameter setting 200/5/5/0/0/16/0 and the corresponding p-values in brackets. The better one is highlighted in green):

![BestEntityModel](https://user-images.githubusercontent.com/48829194/62346711-666f4200-b4f7-11e9-9e2b-12673dd710a8.PNG)

It is quite interesting, that for this parameter setting the entity embeddings perform better than the word embeddings on the entity task and also the average score is higher. Another point to mention here is that when trying to get the scores for the entity task high, this leads to very low scores on the word related tasks. Besides, raw and entity models archieve not always equal scores on the word relatd tasks. More details can be seen in the [results of the cocor package](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/ResultsCocor_BestEntityRelatednessModels.xlsx).

In the following table the 5 best parameter settings for the average spearman score is shown (see [Results_5BestParameters_Average_and_Kore.xlsx](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/Results_5BestParameters_Average_and_Kore.xlsx) for more details):

![5Best](https://user-images.githubusercontent.com/48829194/62344549-bf86a800-b4ee-11e9-8919-080de3a9e68c.PNG)

And the following table shows the 5 best parameter settings for the Kore spearman score:

![5BestEntity](https://user-images.githubusercontent.com/48829194/62344551-c31a2f00-b4ee-11e9-9928-e6b9b2a552ac.PNG)

The 5 best avarage scores are achieved with the ``raw model``, while the ``entity model`` performs better on the 5 best parameters for the entity task (Kore spearman score). Furthermore, for the 5 best average scores always skipgram is used (sg = 1), while  for the 5 best Kore scores CBOW is used. 

Summing up, following conclusion can be made: 
1. When someone intends to obtain good embeddings for the entity task, this person should go with CBOW on a entity annotated corpus. However, this will only produce good embeddings for the entity task, not good embeddings for the word related tasks. 
2. A central role plays the word embeddings algorithm, in particular better embeddings are generally produced by skip gram. 

### FastText
When training the embeddings with other algorithms (FastText) no other trend can be derived (see [ResultFastText.xlsx](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/Results_FastText.xlsx)for more details).

### Other languages
Entity annotation has also no impact on word related tasks, when training the embeddings with other languages, expect from the Spain and French language. Better performance is also for other languages reached with entity embeddings on the entity task and in constrast to the English models also for the best models better performance on the entity task is reached by the entity embeddings. [Results_German.xlsx](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/Results_German.xlsx), [Results_Italian.xlsx](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/Results_Italian.xlsx), [Results_Spanish.xlsx](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/Results_Spanish.xlsx) and [Results_French.xlsx](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/Results_French.xlsx) give more details.


## Summary
First of all, **entity annotation has no significant impact on the word embedding training process**. This means if one should recommend someone whether he or she should use word or entity embeddings, the answer would mostly be using word embeddings. Nevertheless, one can see sometimes differences: While one can not see an impact on the word related tasks, better performance is reached with entity embeddings on the entity task, while for the best models better performance on the entity task is reached by the raw embeddings. Moreover, when trying to get the scores for the entity task high, only worse performance is reached on the world related tasks. However,  reaching only worse performance on the word related tasks do not imply completely noise embeedings as it is shown by the qualitative examination of these embeddings. 


