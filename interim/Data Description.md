# Meaning of each field in the data files

## all_sent.csv and pos_sent.csv

* **idx**: the index of this sentence in the paper, starting from 1
* **text**: the sentence text
* **main_heading**: The topmost header associated with the sentence. It should be a very short heading (<5 words) and contains critical words like ‘introduction’ , ‘experiment’, ‘ablation’, etc.
* **heading**: the innermost subheader that the sentence belongs to
* **topic**: the topic this paper is about
* **paper_idx**: the index of this paper in the topic folder, starting from 0
* **BIO**: the list of BIO tags that marks the boundaries of phrases in this sentence.
* **BIO_1**: more specific BIO tags that not only mark the phrases, but also whether each phrase is a predicate: ‘B-p’ and ‘I-p’ for predicates ; ‘B-n’ and ‘I-n’ for terms
* **BIO_2**: this column is no longer in use.
* **offset1**: the offset of this sentence w.r.t. the topmost header, starting from 0
* **offset2**: the offset of this sentence in the entire paper
* **offset3**: the offset of the sentence w.r.t. heuristically extracted headers (sentences that follow blank lines in plain text files)
* **pro1-3**: the proportional position feature corresponding to offsets 1-3
* **mask**: sentences in ‘background’, ‘related work’ and ‘conclusions’ (or equivalent) sections are skipped by the annotator, so they are masked out (by setting mask=0)
* **bi_labels**: labels for binary classification
* **labels**: labels for multiclass classification. If bi_labels==0, this field is None in the dateframe.

## triples.csv
We use a heuristic way to align each triple to exactly one positive sentence (when possible), even including the cross-sentence triples. <code>triples.csv</code> contains each positive sentence with some overlapping fields with <code>pos_sent.csv</code>, and also the aligned triples that are categorized into different types.

* **info_unit**: the information unit of the sentence
* **text**: the sentence text
* **predicates**: the list of predicates in the sentence, each marked with its start and end token index
* **subj/obj**: the list of terms in the sentence, each marked with its start and end token index
* **triple_A ~ triple_D**: the four types of triples that require neural relation extraction
* other types of triples: this old categorization is no longer used. Please refer to our paper.
* **topic**: the topic this paper is about
* **paper_idx**: the index of this paper in the topic folder, starting from 0
* **idx**: the index of this sentence in the paper, starting from 1