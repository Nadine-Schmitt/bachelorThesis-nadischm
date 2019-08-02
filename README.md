# Does entity annotations have an impact on the word embedding training process?

In my bachelor thesis I trained embeddings from raw text (word embeddings) and from entity annotated text (entity embeddings) with [gensim's word2vec libary](https://radimrehurek.com/gensim/models/word2vec.html) and evaluated them with word related tasks and entity tasks afterwords in order to answer following question:

                     Does entity annotations have an impact on the word embedding training process?
                     
An extensive parameter tuning is peformed and for the best parameters the results are checked with other algorithms of word embeddings using [gensim's fastText libary](https://radimrehurek.com/gensim/models/fasttext.html) and other languages (German, Italian, Spanish and French).

In the following figure an overview of the implementation is given:

![flow2](https://user-images.githubusercontent.com/48829194/62204597-ccd85100-b38d-11e9-97df-d09e76e18ba1.PNG)

For training, the whole [English wikipedia dump](https://dumps.wikimedia.org/enwiki/) (and [German wikipedia dump](https://dumps.wikimedia.org/dewiki/), [Italian wikipedia dump](https://dumps.wikimedia.org/itwiki/), [Spanish wikipedia dump](https://dumps.wikimedia.org/eswiki/), [French wikipedia dump](https://dumps.wikimedia.org/frwiki/) respectively) is used.
Firstly the wikipedia dump is downloaded and then the [WikiExtractor for Wikimentions](https://github.com/samuelbroscheit/wikiextractor-wikimentions) is used in order to convert the downloaded Bz2-files into several files of similiar size in a given directory. Each file contains several documents in a given document format. These are the input for the preproccesing, in which the text is prepared for the training of the word and entity embeddings.  An ``inputList_raw`` and ``inputList_entity`` is outputed, which are files, in which each row contains one sentence. They are used as input corpuses for the training of the word and entity embeddings with [word2vec](https://radimrehurek.com/gensim/models/word2vec.html). Afterwords, the models are evaluated with different evaluation tasks (by using Pearson correlation, Spearman correlation and Pairwise Accuracy as evaluation metrics).The results are compared with each other and to find out, if the Pearson and Spearman correlations are statistical significant, the p-value is calculated. To compare two pearson correlations the [cocor package in R](https://cran.r-project.org/web/packages/cocor/cocor.pdf) is used. The dataset for the entity evaluation task (Kore dataset) is only available in English, and therefore it is translated into the other languages.

## Download wikipedia dump
To download the wikipedia dump, a directory have to be created with
```markdown
mkdir wikidump
```
and then the dump is downloaded with following command (where the URL is the filename of the wikidump):
```markdown
wget "https://dumps.wikimedia.org/enwiki/20190201/enwiki-20190201-pages-articles-multistream.xml.bz2"
```

## Extract wikipedia dump
Downloading the whole wikipedia dump just gives a b2z file and the texts from the Wikipedia database dump has to be extracted and cleaned by the WikiExtractor.py, which is a [Python script](https://github.com/attardi/wikiextractor). The extraction is done with the [WikiExtractor for Wikimentions](https://github.com/samuelbroscheit/wikiextractor-wikimentions) from Samuel Broscheit, which is a modified version of the WikiExtractor with the additional option to extract the internal Wikipedia links from an article. To do so, the WikiExtractor.py is downloaded. Then following command
```markdown
python ~/bin/WikiExtractor.py --json --filter_disambig_pages --processes 4 --collect_links /data/wikidump/enwiki-20190201-pages-articles-multistream.xml.bz2 -o /data/wikiExtracted
```
is run in order that each articles dictionary contains an additional field ``internal_links``. Running the command for the [English wikipedia dump](https://dumps.wikimedia.org/enwiki/), 5669083 articles are extracted. As result the /data/wikiExtracted directory has a size of 21GB and 213 subfolders (from AA to IE), which each has a size of 98MB and contains 100 files (from wiki_00 to wiki_99).

## Preprocessing

The preprocessing is done for each downloaded and extracted wikipedia dump (i.e. for each language) twice in order to get a ``raw input corpus`` and an ``entity annotated corpus``. The preprocessing.py script can be run by calling the command
```markdown
python preproccesing.py WikiExtracted inputList_raw AcronymList -ger
```
for example for the ``german raw inputList`` and
```markdown
python preproccesing.py WikiExtracted inputList_entity AcronymList -e -ger
```
for the ``german entity inputList``. So, the ``extracted wikipedia dump`` is needed as input, as well as an [AcronomyList](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/AcronymLists). Moreover, it have to be also specified if one would like to get an raw or entity corpus (with ``-e``) and which language is used (``ger`` for German, ``it`` for Italian, ``es`` for Spanish and ``fr`` for French}. Besides, the output is stored in the ``inputList_raw`` or ``inputList_entity`` directory. Optionally it is also possible to lower casing the input text (setting ``-l`` flag), however better word and entity embeddings are always achived when not lower casing the input corpora and therefore it is not applied.

Note, that for splitting the text into a list of individual sentences (word2vec requires text, which is organized into sentences, as input) [nltk.sent_tokenize utility](https://www.nltk.org/api/nltk.tokenize.html) is used and have to be imported beforehand by running following commands:
```markdown
mkdir nltk_data
python
import nltk
nltk.download('punkt')
```
Besides, sometimes sentences are unwanted broken by [nltk.sent_tokenize utility](https://www.nltk.org/api/nltk.tokenize.html), since there can be a dot without ending of the sentence, when there is an acronym (Inc. for instance). If there is a dot and the next sentence starts with an upercase letter, then it is a new sentence. In the other case, the next sentence is appended to the current sentence. However there are examples like _i.e. Germany_, where after a dot there is an uppercase letter, but the sentence should not be splitted. Therefore an [AcronomyList](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/AcronymLists), which contains all acronyms with a dot of an given language, is used. If _i.e_  is for example in the [AcronomyList](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/AcronymLists) then the unwanted broken sentences are combined.


## Training and Evaluation
### Word related evaluation task
The word related task is based on the idea that the similarity between two words can be measured with the cosine similarity of their word embeddings. A list of word pairs along with their similarity rating, which is judged by human annotators, have to be provided and  following [goldstandards](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/data) are used. Note that not all datasets are available in all languages:

- Similarity353 (English, German, Italian)
- Relatedness353 (English, German, Italian)
- MEN (English)
- RG65 (English, German, Italian, Spanish, French)
- MTurk (English)
- SimLex999 (English, German, Italian, Spanish, French)
- RareWord (English)

The evaluation task is to measure how well the notion of word similarity according to human annotators is captured by the word embeddings. In other words, the distances between words in an embedding space can be evaluated through the human judgments on the actual semantic distances between these words. Once the cosine similarity between the words is computed, the two obtained distances are then compared with each other using Pearson or Spearman correlation. The more similar they are (i.e. Pearson or Spearman score is closed to 1), the better are the embeddings. 

### Entity evaluation task

The [Kore dataset](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/) is used as entity task. The dataset contain a total of 441 entities. There are 21 seed entities and for each seed there is a ranking of 20 candidate entities, which are linked to by the Wikipedia article of the seed. Example seed entities and Kore gold standard ranks of related entities are shown in following table:

| Seed | Related entity (rank) |
| ---- | --------------------- |
| Apple Inc. | Steve Jobs (1), Steve Wozniak (2) ... NeXT (10), Safari (web browser) (11) ...Ford Motor Company (20) |
| Johnny Depp | Pirates of the Carribbean (1), Jack Sparrow (2) ...  Into the Great Wide Open (10), ... Mad Love (20) |
| GTA IV | Niko Bellic (1), Liberty City (2) ...  New York (10), Bosnian War (11) ... Mothers Against Drunk Driving (20) |
| The Sopranos | Tony Soprano (1), David Chase (2) ...  Golden Globe Award (10), The Kings (11) ... Big Love (20) |
| Chuck Norris | Chuck Norris facts (1), Aaaron Norris (2) ... Northrop Corporation (10), ... Priscilla Presley (20) |

To measure how good the produced word and entity embeddings capture the semantic relatedness between entites following is done: For each seed entity of the Kore dataset a ranking of the 20 candidate entities is produced by using the ``word embeddings`` and the ``entity embeddings``. When using the ``word embeddings`` the similarity between the word embedding of the seed entity and the sum of the word embeddings of the single words is measured by using cosine similarity. When using entity embeddings only the entity embeddings of the given entites have to be considered. To illustarte this, a short example is given:
Let Google be the seed entity with 3 candidate entities ranked as follows:
```markdown
Google
	Larry Page (1)
	Sergey Brin (2)
	Google Maps (3)
```
For the word embeddings, the similarity of the word embedding ``Google`` with the word embedding of ``Larry`` plus the word embedding of ``Page`` is measured. When considering entity embeddings, the similarity of the entity embedding ``Google`` with the entity embedding of ``Larry_Page`` is measured. Afterwords, a ranking based on the similarity score is produced, for instance
\begin{lstlisting}[language=Python]
```markdown
Google	
	Google Maps (1)
	Larry Page (2)
	Sergey Brin (3)
```
Then the quality of the correlation between the gold ranking from Kore and the produced ranking is measured in terms of Spearman correlation and Pairwise accuracy. As result, one have for each entity seed a Spearman and Pairwise Accuracy score. Finally, it is averaged and for each method final value is reported.

Due to the reason that the Kore dataset is only avaiable in English, it is translated into German, Italian, Spanish and French. 

## Translation of Kore dataset
The Kore dataset is automatically translated into German, Italian, Spanish and French. It is done by the TranslateWikipageNames.py script and can be run by calling the command:
```markdown 
python translationWikipageNames.py KoreDataset.txt KoreDataSetTranslated.txt de
```
where the ``KoreDataset.txt`` is the source folder of the original English Kore dataset, ``KoreDataSetTranslated.txt`` is the target folder in which to store the translated data and ``de`` is the language in which one would like to translate to (``de``: German ``it``: Italian, ``es``: Spanish and ``fr``: French).

The resulting [new datasets](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/tree/master/data) are availble and can be downloaded. 

To translate a single entity from the English Kore dataset the [MediaWkiki Action API](https://www.mediawiki.org/wiki/API:Search) is used. For each English entity in the Kore dataset (which has an English wikipedia page), the corresponding wikipedia page in the target language is searched and then taken as translated entity. In the following code snippet the English entity _Google_ is translated into the German entity _Google+_:
![TranslationKore](https://user-images.githubusercontent.com/48829194/62262835-6e0ce900-b41a-11e9-8408-448e33bc640b.PNG)

Note that if there is no wikipedia page in the target language available, the English wikipage name is used. 
Furthermore, some special characters (/, +, ", !, ...) are removed from the original Kore dataset, e.g. _Game of Thrones/ Staffel 7_ is set to _Game of Thrones Staffel 7_. It is done beacuse these characters are also removed in the preprocessing process.

Remark, that there are entities in the datatset, where no corresponding embedding is existent (too less occurence of it in the input corpora for training). If it is a candidate entity then the embedding vector is set to zero. However the seed entity _Terremoti del XXI secolo_ is not avaiable in the Italian entity embedding model and therefore it is removed with its 20 cadidate entites from the original dataset. Moreover, the seed entity _Deus Ex: MankindDivided_ is not avaiable in the Spanish entity embedding model and therefore it is also removed with its 20 cadidate entites from the original dataset. Finally, as the seed entity _Sur écoute_ is not avaible in the French raw embedding model and the seed entities _Quake Champions_ and _Saison 7 de Futurama_ are not avaiable in the French entity embedding model, they are removed with their 20 candidate entities from the original dataset.


## Compare two pearson correlations
In order to compare two pearson correlations the [cocor package in R](https://cran.r-project.org/web/packages/cocor/cocor.pdf) is used. It can be downloaded from the [project's homepage](https://CRAN.R-project.org/package=cocor). 
In order to install the cocor package in R follwoing command is typed into the R console:
```markdown
install.packages("cocor", lib= "/my/own/R-packages/")
library("cocor")
``` 
After running the commands, the cocor package can be used to compare two pearson correlations. It is done for a _dependent overlapping group_ by using following function in R:
```markdown
cocor.dep.groups.overlap(r.jk, r.jh, r.kh, n, alternative = "two.sided", test = "all", alpha = 0.05, conf.level = 0.95, null.value = 0, data.name = NULL, var.labels = NULL, return.htest = FALSE)
```
where following arguments as input are required: 
- **r.jk** is a number of specifying the correlation between j and k (this correlation is used for comparison) 
- **r.jh** is a number of specifying the correlation between j and h (this correlation is used for comparison)
- **r.kh** is a number of specifying the correlation between k and h
- **n** is an integer defining the size of the group
- **alternative** is a character string specifying whether the alternative hypothesis is two-sided ("two.sided"; default) or one-sided ("greater" or "less", depending on the direction)
- **test** is a vector of character strings specifying the tests ((pearson1898, hotelling1940, hendrickson1970, williams1959, olkin1967, dunn1969, steiger1980, meng1992, hittner2003, or zou2007) to be used. With "all" all tests are applied (default)
- **alpha** is a number defining the alpha level for the hypothesis test. The default value is 0.05
- **conf.level** is a number defining the level of confidence for the confidence interval (if test meng1992 or zou2007 is used). The default value is 0.95
- **null.value** is a number defining the hypothesized difference between the two correlations used for testing the null hypothesis. The default value is 0. If the value is other than 0, only the test zou2007 that uses a confidence interval is available
- **data.name** is a character string giving the name of the data/group
- **var.labels** is a vector of three character strings specifying the labels for j, k, and h (in this order)
- **return.htest** is a logical indicating whether the result should be returned as a list containing a list of class htest for each test. The default value is FALSE

Illustrating this, an example of the comparison between the two pearson scores for Similarity353 for the best models with parameter setting (300,3,5,1,0,16) is shown in the following. As output from the training and evaluation on get an pearson score of 0.786 for the raw model and 0.793 for the entity embedding. As also the intercorrelation between the two correlations is needed as input parameter, the correlation between the cosine similarities of the raw model with the cosine similarities of the entity model is computed and given as 0.012. Besides, the Similaritym353 dataset has a size of 203 instances. Therefore following need to be typed in to the R command line in order to compare the two pearson correlations:
```markdown
cocor.dep.groups.overlap(r.jk= 0.786, r.jh= 0.793, r.kh= 0.012, n=203, alternative="two.sided", alpha=0.05, conf.level=0.95, null.value=0)
````
As output all results of the tests are shown and the null hypothesis is for this example always retained:

![OutputCocot](https://user-images.githubusercontent.com/48829194/62342257-86e2d080-b4e6-11e9-8685-94fb930be027.PNG)

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
for the ``entity model``. ``Kore.txt`` is the Kore dataset in the corresponding language and the second argument is the ``input corpus``. The fuctionality is quite simple: For each entity from the Kore datatset the number of occurence is calculated and printed as output. The results can be seen in the [excel sheets]().

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

![BestModelResult](https://user-images.githubusercontent.com/48829194/62343975-8816fc00-b4ec-11e9-8bef-2706b10ca331.PNG)

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

![BestEntityModel](https://user-images.githubusercontent.com/48829194/62344539-b564a980-b4ee-11e9-8a4d-5d4e593b0701.PNG)

It is quite interesting, that for this parameter setting the entity embeddings perform better than the word embeddings on the entity task and also the average score is higher. Another point to mention here is that when trying to get the scores for the entity task high, this leads to very low scores on the word related tasks. Besides, raw and entity models archieve not always equal scores on the word relatd tasks. More details can be seen in the [results of the cocor package](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/blob/master/Results/ResultsCocor_BestEntityRelatednessModels.xlsx).

In the following table the 5 best parameter settings for the average spearman score is shown:

![5Best](https://user-images.githubusercontent.com/48829194/62344549-bf86a800-b4ee-11e9-8919-080de3a9e68c.PNG)

And the following table shows the 5 best parameter settings for the Kore spearman score:

![5BestEntity](https://user-images.githubusercontent.com/48829194/62344551-c31a2f00-b4ee-11e9-9928-e6b9b2a552ac.PNG)

The 5 best avarage scores are achieved with the ``raw model``, while the ``entity model`` performs better on the 5 best parameters for the entity task (Kore spearman score). Furthermore, for the 5 best average scores always skipgram is used (sg = 1), while  for the 5 best Kore scores CBOW is used. 

Summing up, following conclusion can be made: 
1. When someone intends to obtain good embeddings for the entity task, this person should go with CBOW on a entity annotated corpus. However, this will only produce good embeddings for the entity task, not good embeddings for the word related tasks. 
2. A central role plays the word embeddings algorithm, in particular better embeddings are generally produced by skip gram. 

### FastText
When training the embeddings with other algorithms (FastText) no other trend can be derived (see ... for more details).

, no other trend can be derived


### Other languages

## Summary



Following answers to the given research question (RQ) and  its derivational subquestions (SQ) from section  \ref{sec:ResearchQuestion} can be given: (RQ) First of all, entity annotations has no significant impact on the word embedding training process. This means if one should recommend someone whether he or she should use word or entity embeddings, the answer would mostly be using word embeddings. Nevertheless, one can see sometimes differences: While one can not seen an impact on the word related tasks, better performance is reached with entity embeddings on the entity task, while for the best models better performance on the entity task is reached by the raw embeddings. Moreover, when trying to get the scores for the entity task high, only worse performance is reached on the world related tasks. Subquestion 1 (SQ1) is also answered by this statement, because  different results are achieved with different evaluation tasks. Besides, when training the embeddings with other algorithms, no other trend can be derived (SQ2) and it is postulated that entity annotation has also no impact on word related tasks, when training the embeddings with other languages, expect from the Spain and French language. Better performance is also for other languages reached with entity embeddings on the entity task and in constast to the English models also for the best models better performance on the entity task is reached by the entity embeddings (SQ3). Besides, other very interesting results are made in addition to the answers of the reseach question and its derivational subquestions. When analyzing the count of words from the Kore dataset a contradicting result to the literature can be seen and reaching only worse performance on the word related tasks do not imply completely noise embeedings as it is shown by the qualitative examination of these embeddings. Finally it is found that the Kore task is not suitable for the comparison of word and entity embeddings. \\ \\


### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

![flow2](https://user-images.githubusercontent.com/48829194/62204597-ccd85100-b38d-11e9-97df-d09e76e18ba1.PNG)


# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```



For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
