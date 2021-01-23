# NLP Contribution Graph
This repo contains data and code to solve SemEval 2021 Task 11: NLP Contribution Graph\
Owned by the BioNLP group, accessible only to group members.

## Data
* <code>training-data/</code> contains the training data merged with the trial data. There are 29 topics in all, each with one or more papers
* <code>interim/</code> contains csv files produced by the preprocessing step, reports for each subtask, and examples of cross-sentence triples

## Scripts
* **pre**.py - read and preprocess the data, report potential errors in them, and produce csv files
* <code>sent/</code> - this folder contains scripts for subtask 1: sentence classification\
  For this subtask, it is necessary to use the modified simpletransformers packages in this repo
  * **search**.py - run this script for hyperparameter search
  * **train**.py - train a model using a specific hyperparameter setting
  * **test**.py - test the trained model
* <code>ner/</code> - this folder contains scripts for subtask 2: key phrase extraction\
Contains **search**.py, **train**.py and **test**.py, with similar functions as those in the <code>sent/</code> folder\
  Command Line Parameter: <code>t</code> - BIO_type
  * By default t=0, use simple BIO tags to mark the phrases
  * t=1 to distinguish predicates from non-predicates in the tags
  * t=2 to specify if the phrase is a subject, predicate, object, or both subject and object\
  Note that your choice of <code>t</code> should be consistant across the 3 scripts, eg.:\
<code>cd ner</code>\
<code>python search.py --t 1</code>\
<code>python train.py --t 1</code>\
<code>python test.py --t 1</code>
* <code>rel/</code> - this folder contains scripts for subtask 3: triple extraction\
Contains **search**.py, **search1**.py and **train**.py
  * **search**.py conducts pairwise classification and then combine pairs as triples
  * **search1**.py conducts triple classification directly
  * classification for other types of triples will be implemented soon
* <code>simpletransformers/</code> - this folder contains the custimized simple transformers package
  * with custimized model for subtask 1, and custimized hyperparameters for better performance
  * extended from simple transformers version 0.51.10, does not conflict with common usage
  * please first install the common package by running this code:
    * <code>pip install simpletransformers==0.51.10</code>\
    find the installation directory, and replace the <code>simpletransformers</code> folder with this folder
  

## External Links
* See the [official website](https://ncg-task.github.io/) for details of this task
* [Competition page](https://competitions.codalab.org/competitions/25680) on CodaLab
* [Training data](https://github.com/ncg-task/training-data) and [trial data](https://github.com/ncg-task/trial-data) release. Keep an eye on latest updates, because bugs might be fixed on the go.
