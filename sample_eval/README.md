# Sample Test Data
Test data input and expected output examples for the SemEval 2020 Task 11 NLPContributionGraph evaluation per evaluation phase

### `Evaluation Phase 1: End-to-end evaluation phase`

#### Input
The test data release will be formatted as follows:

    [task-name-folder]/                                # machine-translation, named-entity-recognition
        ├── [article-counter-folder]/                  # ranges from 0 to N-1 if we annotated N articles per task
        │   ├── [articlename]-Grobid-out.txt           # plaintext output from the [Grobid parser](https://github.com/kermitt2/grobid)
        │   ├── [articlename]-Stanza-out.txt           # plaintext preprocessed output from [Stanza](https://github.com/stanfordnlp/stanza)
        │   └── ...                                    # if N articles were annotated, this repeats (N-1) more times
        └── ...                                        # if K tasks were selected overall, this repeats (K-1) more times

#### Submission to Codalab
The submission will have be organized per the following directory structure:

    [task-name-folder]/                                # machine-translation, named-entity-recognition
        ├── [article-counter-folder]/                  # ranges from 0 to N-1 if we annotated N articles per task
        │   ├── sentences.txt                          # annotated contribution sentences in the file identified by the sentence number in the preprocessed data with counter starting at 1
        │   └── entities.txt                           # annotated phrases from the contribution sentences where the phrase spans are specified by their first and last token numbers with the token counter starting at 1
        │   └── triples/                               # the folder containing information unit triples one per line
        │   │   └── research-problem.txt               # `research problem` triples (one research problem statement per line)
        │   │   └── model.txt                          # `model` triples (one statement per line)
        │   │   └── ...                                # there are 12 information units in all and each article may be annotated by 3 or 6
        │   └── ...                                    # if N articles were annotated, this repeats (N-1) more times
        └── ...                                        # if K tasks were selected overall, this repeats (K-1) more times

### `Evaluation Phase 2, Part 1: Phrase Extraction Testing`

Coming soon...

### `Evaluation Phase 2, Part 2: Triples Extraction Testing`

Coming soon...

### Notes for making a successful Codalab submission:

1. All `task-name-folder` files will have to zipped as one folder called `submission.zip`. The name has to be `submission.zip`, any other name will return an error on Codalab. Also note that on unzipping `submission.zip`, the individual [task-name-folder] should appear as the top-level directories. 

2. <b>`task-name-folder`.</b> The `trial-data` release considers five NLP tasks: `machine-translation`, `named-entity-recognition`, `question-answering`, `relation-classification`, and `text-classification.` In future data releases, there may be more tasks considered. The submission to Codalab must therefore ensure that all tasks in the particular evaluation release (dev set in the Practice phase or test set in the Evaluation phases) are included. The folder names for the tasks must accord with that in the pertinent release.

3. <b>Information unit names within the `triples` folder.</b> The list of candidate file names are the following twelve: research-problem.txt, approach.txt, model.txt, code.txt, dataset.txt, experimental-setup.txt, hyperparameters.txt, baselines.txt, results.txt, tasks.txt, experiments.txt, ablation-analysis.txt. In general, the names are all lowercased and compound words are separated by a hyphen. Note, if the triples submission made for files named differently from the given twelve, they will evaluated as false positives in terms of Information Unit and Triples evaluations. The script performs no name normalization operation.

4. <b>Triples data lines.</b> The first line of each file will be `(Contribution,has,[InformationUnitName])`. Note that unlike, the candidate file names for information units within which the triples data are listed, the `[InformationUnitName]` as the first triples data line takes the CamelCase format. So the candidates here are: ResearchProblem, Approach, Model, Code, Dataset, ExperimentalSetup, Hyperparameters, Baselines, Results, Tasks, Experiments, AblationAnalysis.

For a hands-on example, please refer to the data in this repository. 

Please post any remaining questions to this Shared Task Google Groups https://groups.google.com/forum/#!forum/ncg-task-semeval-2021
