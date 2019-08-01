## Does entity annotations have an impact on the word embedding training process?

In my bachelor thesis I trained embeddings from raw text (word embeddings) and from entity annotated text (entity embeddings) with [gensim's word2vec libary](https://radimrehurek.com/gensim/models/word2vec.html) and evaluated them with word related tasks and entity tasks afterwords in order to answer following question:

                      Does entity annotations have an impact on the word embedding training process?

An extensive parameter tuning were peformed and for the best parameters the results were checked with other algorithms of word embeddings using [gensim's fastText libary](https://radimrehurek.com/gensim/models/fasttext.html) and other languages (German, Italian, Spanish and French).

In the following figure an overview of the implementation is given:

![flow2](https://user-images.githubusercontent.com/48829194/62204597-ccd85100-b38d-11e9-97df-d09e76e18ba1.PNG)

For training, the whole [English wikipedia dump](https://dumps.wikimedia.org/enwiki/) (and [German wikipedia dump](https://dumps.wikimedia.org/dewiki/), [Italian wikipedia dump](https://dumps.wikimedia.org/itwiki/), [Spanish wikipedia dump](https://dumps.wikimedia.org/eswiki/), [French wikipedia dump](https://dumps.wikimedia.org/frwiki/) respectively).
Firstly the wikipedia dump is downloaded and then the WikiExtractor.py is run in order to convert the downloaded Bz2-files into several files of similiar size in a given directory. Each file contains several documents in a given document format. These are the input for the preproccesing, in which the text is prepared for the training of the word and entity embeddings. Dots at the end of the sentences are removed for example and therefore a list of Acronomys with dots is also needed as input, beacause of not wanting to removing dots from acronmys (\textit{e.g.} is one example). An inputlist raw and inputlist entity is outputed, which are files, in which each row contains one sentence. They are used as input corpuses for the training of the word and entity embeddings with word2vec. Afterwords, the models are evaluated with different evaluation tasks (see section \ref{sec:evaluationTask}) by using Pearson correlation, Spearman correlation and Pairwise Accuracy as evaluation metrics (see section \ref{sec:Pearson}).The results are compared with each other and to find out, if the pearson and spearman correlations are statistical significant, the p-value is calculated (see section \ref{sec:P-value} ) and to compare two pearson correlations the cocor package in R is used (see section \ref{sec:cocor} and \ref{sec:ImplementCocor}). The results are also compared with other algortihms (FastText) and other languages. For the other languages the wikipedia dump of the relevant language is downloaded and all steps mentionend above are done again. The dataset for the entity evaluation task, i.e. Kore dataset (see \ref{sec:EvaluationKore})is only available in English, and therefore it is translated into the relevant language.



You can use the [editor on GitHub](https://github.com/Nadine-Schmitt/bachelorThesis-nadischm/edit/master/README.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

![flow2](https://user-images.githubusercontent.com/48829194/62204597-ccd85100-b38d-11e9-97df-d09e76e18ba1.PNG)



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
