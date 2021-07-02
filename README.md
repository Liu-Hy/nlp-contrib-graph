# NLP Contribution Graph
This repo contains data and code to solve SemEval-2021 Task 11: NLP Contribution Graph\
For detailed description of our method, please see the [paper](https://arxiv.org/abs/2105.05435/) "UIUC_BioNLP at SemEval-2021 Task 11: A Cascade of Neural Models for Structuring Scholarly NLP Contributions".

## Data
* <code>training-data/</code> contains the training data merged with the trial data. There are 29 topics in all, each with one or more papers
* <code>interim/</code> contains csv files produced from the training data by the preprocessing step
  * all_sent.csv - contains all the sentences, each with their section header information, position features, paper origin, etc.
  * pos_sent.csv - contains the positive (contribution-related) sentences, with the above information and their BIO tags.
  * triples.csv - contains each positive sentence with the predicates and terms in it, and the corresponding triples of different types.


## Scripts
* **pre**.py - preprocess training data, report potential errors, produce <code>all_sent.csv</code> and <code>pos_sent.csv</code>
* **ext**.py - preprocess training data, and produce <code>triples.csv</code>
* <code>simpletransformers/</code> - this folder contains the customized Simple Transformers package
  * With customized model for subtask 1, and additional hyperparameters for better performance
  * Extended from simple transformers version 0.51.10, compatible with common usage
  * Please first install the common package by running this code:
    * <code>pip install simpletransformers==0.51.10</code>\
    find the installation directory, and replace the <code>simpletransformers</code> folder with this folder

* <code>train_sent/</code> - Note that all scripts in this folder require the customized Simple Transformers package.
  * A binary classifier is trained for subtask 1: contribution sentence classification
  * A multi-class classifier is trained to classify sentences into information units
  * A filename ended with '_ens' indicates that submodels are trained for ensembling.

* <code>train_ner/</code> - The models are trained for subtask 2: key phrase extraction. 
  * In the 'specific_bio' scheme, we use specific BIO tags to indicate phrase types, and train an NER model directly.
  * In the 'simple_bio' scheme, we first identify the phrases, and then classify them into predicates and terms. The script for ensembling the models are also provided.

* <code>train_rel/</code> - For subtask 3: triple extraction\, four models are trained to extract triples of type A, B, C and D respectively. 
  * For type A triples, two schemes are implemented: pairwise classification and direct triple classification. Only the latter scheme is used in evaluation phases.

* <code>predict1/</code> scripts for Evaluation Phase 1 (end-to-end evaluation). Run the scripts in this order:
  * **pre**.py - test data preprocessing
  * **sent_binary**.py - contribution sentence classification
  * **sent_multi**.py - classify the predicted contribution sentences into information units
  * **ner**.py - extract the phrases. The 'specific-bio' scheme was used in this phase.
  * **predict_triples**.py - extract triples of type A, B, C and D, using different models.
  * **submit**.ipynb - output formatting for submission
* <code>predict2/</code> scripts for Evaluation Phase 2 Part 1: given the contribution sentence labels, do the rest.
  * The naming of scripts basically follows that in <code>predict1/</code>. 
  * A filename ended with '-ens' indicates that an ensemble of submodels is used for prediction.
  * In this phase and later, we used the 'simple-bio' scheme for phrase extraction.
* <code>predict3/</code> scripts for Evaluation Phase 2 Part 2: given the labels of contribution sentences and phrases, and do the rest.
  * We copied the result of information unit classification in <code>predict2/</code>. So after running <code>pre.py</code>, we directly start from phrase classification.

## External Links
* See the [official website](https://ncg-task.github.io/) for details of this task
* [Competition page](https://competitions.codalab.org/competitions/25680) on CodaLab
* [Training data](https://github.com/ncg-task/training-data) and [trial data](https://github.com/ncg-task/trial-data) release.
