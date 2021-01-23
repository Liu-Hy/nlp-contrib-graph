import os
import pandas as pd
import pickle
from ast import literal_eval
import logging
import argparse
from simpletransformers.ner import (
    NERArgs,
    NERModel,
)

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, help='specify the type of BIO tags',
                    default=1)
args = vars(parser.parse_args())
BIO_type = args["t"]

if BIO_type == 0:
    labelset = ["B", "I", "O"]
    type_ls = ['']
elif BIO_type == 1:
    labelset = ["B-p", "I-p", "B-n", "I-n", "O"]
    type_ls = ['-p', '-n']
elif BIO_type == 2:
    labelset = ["B-s", "I-s", "B-p", "I-p", "B-ob", "I-ob", "B-b", "I-b", "O"]
    type_ls = ['-s', '-p', '-ob', 'b']

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

report = []

pos = pd.read_csv('../interim/pos_sent.csv')
col_name = pos.columns[6+BIO_type]
pos = pos.dropna(axis=0, subset=[col_name])
pos = pos.drop('bi_labels', axis=1)
pos = pos.sample(frac=1, random_state=1)
bound = int(0.9*len(pos))
p_train = pos[:bound]
#p_train.reset_index(drop = True)
p_eval = pos[bound:]
#p_eval.reset_index(drop = True)
train_data = []
for i in range(len(p_train)):
    words = p_train.iloc[i, 1].split(' ')
    tags = literal_eval(p_train.iloc[i, 6+BIO_type])
    for j in range(len(words)):
        train_data.append([i, words[j], tags[j]])
train_df = pd.DataFrame(train_data, columns=['sentence_id', 'words', 'labels'])

eval_data = []
for i in range(len(p_eval)):
    words = p_eval.iloc[i, 1].split(' ')
    tags = literal_eval(p_eval.iloc[i, 6+BIO_type])
    for j in range(len(words)):
        eval_data.append([i, words[j], tags[j]])
eval_df = pd.DataFrame(eval_data, columns=['sentence_id', 'words', 'labels'])

model_args = NERArgs()
model_args.train_batch_size = 8
model_args.learning_rate = 1e-4
model_args.num_train_epochs = 2
model_args.reprocess_input_data = True
model_args.save_model_every_epoch = False
model_args.overwrite_output_dir = True
model_args.fp16 = False
model_args.manual_seed = 1
model_args.use_multiprocessing = True
model_args.gradient_accumulation_steps = 4
model_args.do_lower_case = True  # when using uncased model

def get_entity_spans(ls):
    # given a sequence of BIO taggs, get the list of tuples representing spans of entities
    spans = []
    for type in type_ls:
        for i in range(len(ls)):
            st, ed = 0, 0
            if ls[i] == 'B' + type:
                st, ed = i, i + 1
                for j in range(i+1, len(ls)):
                    if ls[j] == 'I' + type:
                        ed += 1
                    else:
                        break
                spans.append((st, ed))
    spans = sorted(spans, key=lambda x: x[0])
    return spans

def phrase_F1(ref, pred):
    TP = FP = FN = 0
    for k in range(len(pred)):
        # get the lists of predicted and reference tuples from the tag sequence
        pred_ls = get_entity_spans(pred[k])
        ref_ls = get_entity_spans(ref[k])
        # get the list of true positives tuples, and so on
        TPs = [i for i in pred_ls if i in ref_ls]
        FPs = [i for i in pred_ls if i not in ref_ls]
        FNs = [i for i in ref_ls if i not in pred_ls]
        TP += len(TPs)
        FP += len(FPs)
        FN += len(FNs)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    print(f'precision: {precision}, recall: {recall}, F1: {F1}')
    return F1

# Create a TransformerModel
model = NERModel(
    "bert",
    "allenai/scibert_scivocab_uncased",
    labels=labelset,
    args=model_args,
)

# Train the model
model.train_model(train_df)

# Evaluate the model
#model.eval_model(eval_df, F1_score=phrase_F1)

