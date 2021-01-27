# Evaluation Phase 1: End-to-end Pipeline Testing

#### Input Dataset for the SemEval 2020 Task 11 NLPContributionGraph Shared Task

The dataset is formatted as follows:

    [task-name-folder]/                                # constituency_parsing, coreference_resolution, data-to-text_generation, entity_linking, face_alignment, face_detection, hypernym_discovery, natural_language_inference
        ├── [article-counter-folder]/                  # ranges from 0 to N-1 if we annotated N articles per task
        │   ├── [articlename]-Grobid-out.txt           # plaintext output from the [Grobid parser](https://github.com/kermitt2/grobid)
        │   ├── [articlename]-Stanza-out.txt           # plaintext preprocessed output from [Stanza](https://github.com/stanfordnlp/stanza)
        │   └── ...                                    # if N articles were annotated, this repeats (N-1) more times
        └── ...   

#### Submission to Codalab
The submission will have be organized per the following directory structure:

    [task-name-folder]/                                # constituency_parsing, coreference_resolution, data-to-text_generation, entity_linking, face_alignment, face_detection, hypernym_discovery, natural_language_inference
        ├── [article-counter-folder]/                  # ranges from 0 to N-1 if we annotated N articles per task
        │   ├── sentences.txt                          # annotated contribution sentences in the file identified by the sentence number in the preprocessed data with counter starting at 1
        │   └── entities.txt                           # annotated phrases from the contribution sentences where the phrase spans are specified by their first and last token numbers with the token counter starting at 1
        │   └── triples/                               # the folder containing information unit triples one per line
        │   │   └── research-problem.txt               # `research problem` triples (one research problem statement per line)
        │   │   └── model.txt                          # `model` triples (one statement per line)
        │   │   └── ...                                # there are 12 information units in all and each article may be annotated by 3 or 6
        │   └── ...                                    # if N articles were annotated, this repeats (N-1) more times
        └── ...                                        # if K tasks were selected overall, this repeats (K-1) more times		
		
A valid sample submission for this phase can be downloaded [here](https://github.com/ncg-task/evaluation-phase1/blob/master/submission.zip)
