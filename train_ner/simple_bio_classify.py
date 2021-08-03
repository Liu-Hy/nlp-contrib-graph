import logging
import pandas as pd
import numpy as np
from ast import literal_eval as load
import sklearn

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

def convert(arr):
    ls = []
    for i in range(len(arr)):
        pre_ls = load(arr[i, 3])
        np_ls = load(arr[i, 4])
        for pre in pre_ls:
            word_ls = arr[i, 2].split(' ')
            word_ls.insert(pre[1][0], '<<')
            word_ls.insert(pre[1][1]+1, '>>')
            ls.append([int(arr[i, 0]), ' '.join(word_ls), 1])
        for np in np_ls:
            word_ls = arr[i, 2].split(' ')
            word_ls.insert(np[1][0], '<<')
            word_ls.insert(np[1][1]+1, '>>')
            ls.append([int(arr[i, 0]), ' '.join(word_ls), 0])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['idx', 'text', 'labels']
    return dataframe

df = df.values
df = convert(df)
df = df.sample(frac=1, random_state=1)
bound = int(0.9*len(df))
train_df = df[:bound]
eval_df = df[bound:]

train_pos = train_df[train_df['labels'] == 1]
train_neg = train_df[train_df['labels'] == 0]
ratio = len(train_neg)/len(train_pos)
print(f'imbalance ratio: {ratio}')

model_args = ClassificationArgs()
model_args.use_early_stopping = True
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 2
model_args.early_stopping_consider_epochs = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.fp16 = False
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.output_dir = 'classify/'
model_args.use_multiprocessing = False
model_args.best_model_dir = 'classify/best_model'

model_args.num_train_epochs = 10
model_args.train_batch_size = 8
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.do_lower_case = False

model = ClassificationModel(
    "xlmroberta",
    "xlm-roberta-base",
    args=model_args,
    weight=[1, ratio],
)

model.train_model(train_df, eval_df=eval_df, F1_score=sklearn.metrics.f1_score)
