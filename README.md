# NLP Contribution Graph
This repo contains data and code to solve SemEval-2021 Task 11: NLP Contribution Graph\
For detailed description of our method, please see the [paper](https://arxiv.org/abs/2105.05435/) "UIUC_BioNLP at SemEval-2021 Task 11: A Cascade of Neural Models for Structuring Scholarly NLP Contributions".

## Dependencies
* This repo requires <code>simpletransformers/</code> - the customized Simple Transformers package 
  * With customized model for subtask 1 to incorporate additional features
  * Extended from Simple Transformers version 0.51.10, compatible with common usage
  * Please first install the common package by running this code:
    * <code>pip install simpletransformers==0.51.10</code>\
    find the installation directory, and replace the <code>simpletransformers</code> folder with this folder

## Data
* <code>training_data/</code> - the training data merged with the trial data, with full annotation.
* <code>interim/</code> - intermediate data files converted from the training data
  * **all_sent**.csv - contains all the sentences, each with its section header, positional features, paper topic and index, BIO tags, etc.
  * **pos_sent**.csv - a subset of *all_sent*.csv consisting of all the positive sentences.
  * **triples**.csv - contains each positive sentence with the predicates and terms in it, and the corresponding triples of different types.
* <code>test_data/</code> - the test data, with sentence and phrase annotation released.

## Scripts
* **pre**.py - preprocess training data, report potential errors, produce *all_sent*.csv and *pos_sent*.csv
* **ext**.py - preprocess training data, and produce *triples*.csv

* <code>train_sent/</code> - Note that all scripts in this folder require the customized Simple Transformers package.
  * A binary classifier is trained for subtask 1: contribution sentence classification
  * A multi-class classifier is trained to classify sentences into information units
  * A filename ended with '_ens' indicates that submodels are trained for ensembling.

* <code>train_ner/</code> - The models are trained for subtask 2: key phrase extraction. 
  * In the 'specific_bio' scheme, we use specific BIO tags to indicate phrase types, and train an NER model directly.
  * In the 'simple_bio' scheme, we first identify the phrases, and then classify them into predicates and terms. The script for ensembling the models are also provided.

* <code>train_rel/</code> - For subtask 3: triple extraction\, four models are trained to extract triples of type A, B, C and D respectively. 
  * For type A triples, two schemes are implemented: pairwise classification and direct triple classification. Only the latter scheme is used in evaluation phases.

* <code>predict1/</code> - scripts for Evaluation Phase 1 (end-to-end evaluation). Run the scripts in this order:
  * **pre**.py - test data preprocessing
  * **sent_binary**.py - contribution sentence classification
  * **sent_multi**.py - information unit classification
  * **ner**.py - phrases extraction. The 'specific-bio' scheme was used in this phase.
  * **predict_triples**.py - extraction of type A, B, C and D triples, using different models.
  * **submit**.ipynb - output formatting for submission
* <code>predict2/</code> - scripts for Evaluation Phase 2 Part 1: given the contribution sentence labels, do the rest.
  * The naming of scripts basically follows that in *predict1/*. 
  * A filename ended with '-ens' indicates that an ensemble of submodels is used for prediction.
  * In this phase and later, we used the 'simple-bio' scheme for phrase extraction.
* <code>predict3/</code> - scripts for Evaluation Phase 2 Part 2: given the labels of contribution sentences and phrases, do the rest.
  * We copied the result of information unit classification in *predict2/*. Thus after running *pre*.py, we directly started from phrase classification.

## Useful Links
* [Task description paper](https://arxiv.org/abs/2106.07385)
* [Official website](https://ncg-task.github.io/) of the task
* [Training data](https://github.com/ncg-task/training-data) and [trial data](https://github.com/ncg-task/trial-data) release.
