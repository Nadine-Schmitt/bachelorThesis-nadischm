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


![TranslationKore](https://user-images.githubusercontent.com/48829194/62262835-6e0ce900-b41a-11e9-8408-448e33bc640b.PNG)



## Compare two pearson correlations

## Further analysis

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


## Results

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
