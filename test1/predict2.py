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

BIO_type = 1

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

pos = pd.read_csv('pos_sent.csv')
col_name = pos.columns[6+BIO_type]
pos = pos.dropna(axis=0, subset=[col_name])
pos = pos.drop('bi_labels', axis=1)

data = []
for i in range(len(pos)):
    words = pos.iloc[i, 1].split(' ')
    tags = literal_eval(pos.iloc[i, 6+BIO_type])
    for j in range(len(words)):
        data.append([i, words[j], tags[j]])
df = pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])

model_args = NERArgs()
model_args.reprocess_input_data = True
model_args.save_model_every_epoch = False
model_args.overwrite_output_dir = True
model_args.fp16 = False
model_args.manual_seed = 1
model_args.use_multiprocessing = True
model_args.do_lower_case = True  # when using uncased model


def phrase_F1(ref, pred):
    return 0

# Create a TransformerModel
model = NERModel(
    "bert",
    "trained_model_directory",
    labels=labelset,
    args=model_args,
)

# Evaluate the model
result, model_outputs, pred_label = model.eval_model(df, F1_score=phrase_F1)
# 'pred_label' is a list, where each element is the list of predicted labels for one sentence

