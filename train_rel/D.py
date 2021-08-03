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
df = df[df['subj/obj'] != '[]']
df.insert(loc=0, column='idx', value=np.arange(len(df)))
df = df.sample(frac=1, random_state=1)
bound = int(0.9*len(df))
t_df = df[:bound]
e_df = df[bound:]

t_df = t_df.values
e_df = e_df.values

def convert(arr,is_eval):
    missed=0
    ls=[]
    for i in range(len(arr)):
        if arr[i,3]=='[]' or ((arr[i,3]!='[]') and (load(arr[i,3])[0][1][0]>load(arr[i,4])[0][1][0])):
            np = load(arr[i,4])[0]
            word_ls = arr[i,2].split(' ')
            word_ls.insert(np[1][0], '[[')
            word_ls.insert(np[1][1]+1, ']]')
            unit = arr[i,1]
            unit = (unit[0].upper()+unit[1:]).replace('-',' ')
            unit_ls = ['[[']+(unit.split(' '))+[']]']
            word_ls = unit_ls+[':']+word_ls
            flg=0
            if arr[i,8]=='[]':
                trip_ls = []
            else:
                trip_ls = load(arr[i,8])
                for trip in trip_ls:
                    if trip[1]=='has' and trip[2]==np[0]:
                        flg=1
                        break
            ls.append([unit, np[0], trip_ls, ' '.join(word_ls), flg])
        else:
            missed+=len(load(arr[i,8]))
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['info_unit', 'np', 'triples', 'text', 'labels']
    if is_eval:
        print(f'missed {missed} triples in the eval set')
        return dataframe, missed
    else:
        return dataframe

train_df = convert(t_df, 0)
eval_df, missed = convert(e_df, 1)
num_pos = len(train_df[train_df['labels'] == 1])
num_neg = len(train_df[train_df['labels'] == 0])
imbalance_ratio = num_neg/num_pos

model_args = ClassificationArgs()
model_args.use_early_stopping = True
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 18
model_args.early_stopping_consider_epochs = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.output_dir = 'D/'
model_args.best_model_dir = 'D/best_model'
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.manual_seed = 1
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.num_train_epochs = 12
model_args.train_batch_size = 8
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.warmup_steps = 200
model_args.do_lower_case = True

def triple_F1(ref, pred):
    TP = FP = FN = 0
    for i in range(len(pred)):
        ref_ls = eval_df.iloc[i, 2]
        pred_ls = []
        if pred[i] == 1:
            trip = [eval_df.iloc[i, 0], 'has', eval_df.iloc[i, 1]]
            pred_ls.append(trip)
        TP += len([t for t in pred_ls if t in ref_ls])
        FP += len([t for t in pred_ls if t not in ref_ls])
        FN += len([t for t in ref_ls if t not in pred_ls])
    FN += missed
    pc = TP/(TP+FP)
    rc = TP/(TP+FN)
    print('computing F1')
    F1 = 2*pc*rc/(pc+rc)
    print(f'precision {pc}, recall {rc}, F1 {F1}')
    return F1

model = ClassificationModel(
    "bert",
    "allenai/scibert_scivocab_uncased",
    weight=[1, imbalance_ratio],
    args=model_args,
)
print(f'imbalance ratio: {imbalance_ratio}')
# Train the model
model.train_model(train_df, eval_df=eval_df, F1_score=triple_F1)
