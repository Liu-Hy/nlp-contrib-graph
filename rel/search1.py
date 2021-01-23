# search in the hyperparameter space with W&B sweep
import logging

import pandas as pd
import numpy as np
from ast import literal_eval
import sklearn

import wandb

from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "F1_score", "goal": "maximize"},
    "parameters": {
        "train_batch_size": {"values": [16]},
        "learning_rate": {"values": [1e-5]},
        #"threshold": {"min": 0.8, "max": 0.99}
    },
}
# "learning_rate": {"min": 5e-5, "max": 4e-4},
sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('../interim/try1.csv')
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

train_df.to_csv('train.csv')
eval_df.to_csv('eval.csv')

# dowmsample the negative samples
train_pos = train_df[train_df['labels'] == 1]
train_neg = train_df[train_df['labels'] == 0]
# train_neg = train_neg.sample(n=len(train_pos), random_state=1)
train_df = pd.concat([train_pos, train_neg])
train_df = train_df.sample(frac=1, random_state=1)

# Create a ClassificationModel

model_args = ClassificationArgs()
# arguments for early stop
model_args.use_early_stopping = True
# model_args.early_stopping_delta = 0.01
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 8
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
model_args.num_train_epochs = 2
# model_args.train_batch_size = 8 #16
model_args.gradient_accumulation_steps = 4
# model_args.learning_rate = 1e-4
# model_args.num_labels = 2 # SHOULD it be here?
# model_args.weight = [1, 3] # SHOULD it be here?
model_args.do_lower_case = True  # when using uncased model
model_args.wandb_project = "Simple Sweep"

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
    
def train():
    # Initialize a new wandb run
    wandb.init()
    ratio = len(train_neg)/len(train_pos)
    print(ratio)
    # Create a TransformerModel
    model = ClassificationModel(
        "bert",
        "allenai/scibert_scivocab_uncased",
        args=model_args,
        weight=[1,ratio],
        sweep_config=wandb.config,
    )

    # Train the model
    model.train_model(train_df, eval_df=eval_df,
                      F1_score=triple_F1)

    # Evaluate the model
    # model.eval_model(valid_eval, F1_score=sklearn.metrics.f1_score)

    # Sync wandb
    wandb.join()

wandb.agent(sweep_id, train)
