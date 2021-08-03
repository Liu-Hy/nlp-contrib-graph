import logging

import pandas as pd
import numpy as np
from ast import literal_eval as load

from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('../interim/triples.csv')
df = df.rename(columns={'idx': 'indx'})
df['msk'] = 1
for i in range(len(df)):
    if len(load(df.iloc[i, 3])) < 2:
        df.iloc[i, 16] = 0
df = df[df['msk'] == 1]
df.insert(loc=0, column='idx', value=np.arange(len(df)))
df = df.sample(frac=1, random_state=1)
bound = int(0.9*len(df))
t_df = df[:bound]
e_df = df[bound:]
t_df = t_df.values
e_df = e_df.values

def convert(arr):
    ls = []
    for i in range(len(arr)):
        lth = len(load(arr[i, 4]))
        for p1 in range(lth-1):
            for p2 in range(p1+1, p1+2):  # p1+2 or lth
                phrase1 = load(arr[i, 4])[p1]
                phrase2 = load(arr[i, 4])[p2]
                a_ls = load(arr[i, 5])
                possible = 1
                for a in a_ls:
                    if (a[0] == phrase1[0] and a[2] == phrase2[0]) or (a[0] == phrase2[0] and a[2] == phrase1[0]):
                        possible = 0
                        break
                if possible == 1:
                    word_ls = arr[i, 2].split(' ')
                    word_ls.insert(phrase1[1][0], '<<')
                    word_ls.insert(phrase1[1][1]+1, '>>')
                    word_ls.insert(phrase2[1][0]+2, '[[')
                    word_ls.insert(phrase2[1][1]+3, ']]')
                    flg = 0
                    trip_ls = load(arr[i, 6])
                    for trip in trip_ls:
                        if trip[0] == phrase1[0] and trip[2] == phrase2[0]:
                            if trip[1] == 'has':
                                flg = 1
                                break
                            elif trip[1] == 'name':
                                flg = 2
                                break
                    ls.append([phrase1[0], phrase2[0],
                               trip_ls, ' '.join(word_ls), flg])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['phrase1', 'phrase2', 'triples', 'text', 'labels']
    return dataframe

train_df = convert(t_df)
eval_df = convert(e_df)
num_negative = len(train_df[train_df['labels'] == 0])
weight_list = [1]
for i in range(1, 3):
    weight_list.append(num_negative/len(train_df[train_df['labels'] == i]))

model_args = ClassificationArgs()
model_args.use_early_stopping = True
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 4
model_args.early_stopping_consider_epochs = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.output_dir = 'B/'
model_args.best_model_dir = 'B/best_model'
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = False
model_args.num_train_epochs = 20
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 3e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
#model_args.warmup_steps = 200
model_args.do_lower_case = True

def triple_F1(ref, pred):
    TP = FP = FN = 0
    for i in range(len(pred)):
        ref_ls = eval_df.iloc[i, 2]
        pred_ls = []
        if pred[i] != 0:
            if pred[i] == 1:
                trip = [eval_df.iloc[i, 0], 'has', eval_df.iloc[i, 1]]
            elif pred[i] == 2:
                trip = [eval_df.iloc[i, 0], 'name', eval_df.iloc[i, 1]]
            pred_ls.append(trip)
        TP += len([t for t in pred_ls if t in ref_ls])
        FP += len([t for t in pred_ls if t not in ref_ls])
        false_n = [t for t in ref_ls if t not in pred_ls]
        if len(false_n) > 0:
            FN += 1
    pc = TP/(TP+FP)
    rc = TP/(TP+FN)
    F1 = 2*pc*rc/(pc+rc)
    print(f'precision {pc}, recall {rc}, F1 {F1}')
    return F1

model = ClassificationModel(
    "bert",
    "allenai/scibert_scivocab_uncased",
    weight=weight_list,
    num_labels=3,
    args=model_args,
)
print(f'weight list: {weight_list}')
# Train the model
model.train_model(train_df, eval_df=eval_df, F1_score=triple_F1)
