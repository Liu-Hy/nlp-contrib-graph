import logging
import pandas as pd
import numpy as np
from ast import literal_eval

from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('../interim/triples.csv')
df = df.rename(columns={'idx': 'indx'})
df.insert(loc=0, column='idx', value=np.arange(len(df)))
df = df.sample(frac=1, random_state=1)
bound = int(0.9*len(df))
t_df = df[:bound]
e_df = df[bound:]

def convert(arr):
    ls = []
    for i in range(len(arr)):
        pre_ls = literal_eval(arr[i, 3])
        np_ls = literal_eval(arr[i, 4])
        p_idx = [(1, *p[1]) for p in pre_ls]
        n_idx = [(0, *n[1]) for n in np_ls]
        for p in range(len(p_idx)):
            for j in range(len(n_idx)-1):
                for k in range(j+1, len(n_idx)):
                    word_ls = arr[i, 2].split(' ')
                    indx = sorted([p_idx[p], n_idx[j], n_idx[k]], key=lambda x:x[1])
                    for w in range(len(indx)):
                        if indx[w][0] == 1:
                            word_ls.insert(indx[w][1]+2*w, '<<')
                            word_ls.insert(indx[w][2]+2*w+1, '>>')
                        else:
                            word_ls.insert(indx[w][1]+2*w, '[[')
                            word_ls.insert(indx[w][2]+2*w+1, ']]')
                    flg = 0
                    for tp in literal_eval(arr[i, 5]):
                        if tp == [np_ls[j][0], pre_ls[p][0], np_ls[k][0]]:
                            flg = 1
                            break
                    ls.append([int(arr[i, 0]), ' '.join(word_ls), flg])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['idx', 'text', 'labels']
    return dataframe

t_df = t_df.values
e_df = e_df.values

train_df = convert(t_df)
eval_df = convert(e_df)

# dowmsample the negative samples
train_pos = train_df[train_df['labels'] == 1]
train_neg = train_df[train_df['labels'] == 0]
# train_neg = train_neg.sample(n=len(train_pos), random_state=1)
train_df = pd.concat([train_pos, train_neg])
train_df = train_df.sample(frac=1, random_state=1)

model_args = ClassificationArgs()
model_args.use_early_stopping = True
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 2
model_args.early_stopping_consider_epochs = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.output_dir = 'A/'
model_args.best_model_dir = 'A/best_model'
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.manual_seed = 1
model_args.fp16 = False
model_args.num_train_epochs = 10
model_args.use_multiprocessing = False
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.warmup_steps = 200
model_args.do_lower_case = True

def triple_F1(ref, pred):
    # pred is an ndarray with shape (Number of test samples, )
    tp = len(np.where((pred == 1) & (ref == 1))[0])
    fp = len(np.where((pred == 1) & (ref == 0))[0])
    fn = len(np.where((pred == 0) & (ref == 1))[0])
    pc = tp / (tp + fp)
    rc = tp / (tp + fn)
    f1 = 2 * pc * rc / (pc + rc)
    print('-' * 30)
    print(f'precision {pc}, recall {rc}, F1 {f1}')
    print('-' * 30)
    return f1

ratio = len(train_neg)/len(train_pos)
model = ClassificationModel(
    "bert",
    "allenai/scibert_scivocab_uncased",
    args=model_args,
    weight=[1, ratio],
)

model.train_model(train_df, eval_df=eval_df,
                  F1_score=triple_F1)


