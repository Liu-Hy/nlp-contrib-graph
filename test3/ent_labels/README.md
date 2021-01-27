# Evaluation Phase 2, Part 2: Triples Extraction

#### Gold Entities Annotations for the NCG Shared Task Test Dataset

The dataset is formatted as follows:

    [task-name-folder]/                                # constituency_parsing, coreference_resolution, data-to-text_generation, entity_linking, face_alignment, face_detection, hypernym_discovery, natural_language_inference
        ├── [article-counter-folder]/                  # ranges from 0 to N-1 if we annotated N articles per task
        │   ├── entities.txt                           # annotated phrases in the contribution sentences in the file identified by the sentence number in the preprocessed data with counter starting at 1		
        │   └── ...                                    # if N articles were annotated, this repeats (N-1) more times
        └── ...   


		
