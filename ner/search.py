# search in the hyperparameter space with W&B sweep

import os
import pandas as pd
import pickle
from ast import literal_eval
import logging
import argparse
import wandb
from simpletransformers.ner import (
    NERArgs,
    NERModel,
)

parser = argparse.ArgumentParser()
parser.add_argument('--t', type=int, help='specify the type of BIO tags',
                    default=0)
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

sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "F1_score", "goal": "maximize"},
    "parameters": {
        "train_batch_size": {"values": [8, 16]},
        "learning_rate": {"values": [4e-5, 1e-4]},
    },
}
# "learning_rate": {"min": 5e-5, "max": 4e-4},
sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

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
# arguments for early stop
model_args.use_early_stopping = True
# model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 1
model_args.early_stopping_consider_epochs = True
# model_args.evaluate_during_training_steps = 500
model_args.evaluate_during_training_verbose = True

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.no_save = True
model_args.fp16 = False
model_args.evaluate_during_training = True
model_args.manual_seed = 1
model_args.use_multiprocessing = True
model_args.num_train_epochs = 8
# model_args.train_batch_size = 8 #16
model_args.gradient_accumulation_steps = 4
# model_args.learning_rate = 1e-4
model_args.do_lower_case = True  # when using uncased model
model_args.wandb_project = "Simple Sweep"

type_ls = ['-p', '-n']
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
    return F1
    # print(f'precision: {precision}, recall: {recall}, F1: {F1}')

def train():
    # Initialize a new wandb run
    wandb.init()

    # Create a TransformerModel
    model = NERModel(
        "bert",
        "allenai/scibert_scivocab_uncased",
        labels=labelset,
        args=model_args,
        sweep_config=wandb.config,
    )
    # "roberta", "roberta-base"

    # Train the model
    model.train_model(train_df, eval_df=eval_df,
                      F1_score=phrase_F1)

    # Evaluate the model
    # model.eval_model(valid_eval, F1_score=sklearn.metrics.f1_score)

    # Sync wandb
    wandb.join()

wandb.agent(sweep_id, train)


