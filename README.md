# Does entity annotations have an impact on the word embedding training process?

In my bachelor thesis I trained embeddings from raw text (word embeddings) and from entity annotated text (entity embeddings) with [gensim's word2vec libary](https://radimrehurek.com/gensim/models/word2vec.html) and evaluated them with word related tasks and entity tasks afterwords in order to answer following question:

                     Does entity annotations have an impact on the word embedding training process?
                     
An extensive parameter tuning were peformed and for the best parameters the results were checked with other algorithms of word embeddings using [gensim's fastText libary](https://radimrehurek.com/gensim/models/fasttext.html) and other languages (German, Italian, Spanish and French).

In the following figure an overview of the implementation is given:

![flow2](https://user-images.githubusercontent.com/48829194/62204597-ccd85100-b38d-11e9-97df-d09e76e18ba1.PNG)

For training, the whole [English wikipedia dump](https://dumps.wikimedia.org/enwiki/) (and [German wikipedia dump](https://dumps.wikimedia.org/dewiki/), [Italian wikipedia dump](https://dumps.wikimedia.org/itwiki/), [Spanish wikipedia dump](https://dumps.wikimedia.org/eswiki/), [French wikipedia dump](https://dumps.wikimedia.org/frwiki/) respectively).
Firstly the wikipedia dump is downloaded and then the [WikiExtractor for Wikimentions](https://github.com/samuelbroscheit/wikiextractor-wikimentions) is used in order to convert the downloaded Bz2-files into several files of similiar size in a given directory. Each file contains several documents in a given document format. These are the input for the preproccesing, in which the text is prepared for the training of the word and entity embeddings.  An inputList_raw and inputList_entity is outputed, which are files, in which each row contains one sentence. They are used as input corpuses for the training of the word and entity embeddings with [word2vec](https://radimrehurek.com/gensim/models/word2vec.html). Afterwords, the models are evaluated with different evaluation tasks (by using Pearson correlation, Spearman correlation and Pairwise Accuracy as evaluation metrics).The results are compared with each other and to find out, if the Pearson and Spearman correlations are statistical significant, the p-value is calculated. To compare two pearson correlations the [cocor package in R](https://cran.r-project.org/web/packages/cocor/cocor.pdf) is used. The dataset for the entity evaluation task (Kore dataset) is only available in English, and therefore it is translated into the other languages.

## Download wikipedia dump

## Extract wikipedia dump

## Preprocessing

## Training and Evaluation
#Word related evaluation task
- Similarity353 (English, German, Italian)
- Relatedness353 (English, German, Italian)
- MEN (English)
- RG65 (English, German, Italian, Spanish, French)
- MTurk (English)
- SimLex999 (English, German, Italian, Spanish, French)
- RareWord (English)


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

The countries and languages are grouped correctly. The connecting lines are approximately parallel and of the same length. So the concept of capitals and languages is understood by the word embedding model.

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
