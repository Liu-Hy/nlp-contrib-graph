import logging

import pandas as pd
import sklearn
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from simpletransformers.classification import (
    ClassificationArgs1,
    ClassificationModel1,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('../interim/pos_sent.csv')
df = df.drop(columns=['BIO', 'BIO_1', 'BIO_2']).dropna(subset=['labels'])
df['title'] = df['main_heading'] + ': ' + df['heading']
df.loc[((df['main_heading'] == df['heading']) | (
    pd.isnull(df['heading']))), 'title'] = df['main_heading']
df['title'] = df['title'].fillna('')
df['paper'] = df['topic'] + df['paper_idx'].astype(str)
df.loc[(df['labels'] == 'hyperparameters') | (df['labels']
                                              == 'experimental-setup'), 'labels'] = 'hyper-setup'
df.loc[(df['labels'] == 'model') | (
    df['labels'] == 'approach'), 'labels'] = 'method'

df = df.sample(frac=1, random_state=1)
bound = int(0.85*len(df))

train_df = df[:bound]
eval_df = df[bound:]
print(f'train_df has length: {len(train_df)}')
print(f'eval_df has length: {len(eval_df)}')
train_df = train_df[(train_df['mask'] == 1) & (train_df['labels'] != 'code')]
eval_df = eval_df[(eval_df['mask'] == 1) & (eval_df['labels'] != 'code')]
label_list = ['results',
              'ablation-analysis',
              'method',
              'baselines',
              'dataset',
              'hyper-setup',
              'experiments',
              'research-problem',
              'tasks']
num_max = len(df[df['labels'] == 'results'])
weight_list = []
for label in label_list:
    weight_list.append(num_max/len(df[df['labels'] == label]))

model_args = ClassificationArgs1()
model_args.labels_list = label_list
model_args.use_early_stopping = True
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 3
model_args.early_stopping_consider_epochs = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_args.downsample = 1.0
model_args.normalize_ofs = True
model_args.out_learning_rate = 5e-5
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.output_dir = 'multi/'
model_args.best_model_dir = 'multi/best_model'
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.manual_seed = 1
model_args.use_multiprocessing = False  # set to True if cpu memory is enough
model_args.fp16 = False
model_args.num_train_epochs = 20
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.warmup_steps = 500
model_args.do_lower_case = True

def F1_score(ref, pred):
    print(classification_report(ref, pred, target_names=label_list))
    cm = confusion_matrix(ref, pred, labels=np.arange(9), normalize=None)
    print(cm)
    cm = confusion_matrix(ref, pred, labels=np.arange(9), normalize='true')
    cm = cm.tolist()
    df_cm = pd.DataFrame(cm, index=label_list, columns=label_list)
    plt.figure(figsize=(9, 9))
    sn.heatmap(df_cm, annot=True)

    return sklearn.metrics.f1_score(ref, pred, labels=np.arange(9), average='macro')

model = ClassificationModel1(
    "bert",
    "allenai/scibert_scivocab_uncased",
    weight=weight_list,
    num_labels=9,
    args=model_args,
)
print(f'weight list: {weight_list}')
# Train the model
model.train_model(train_df, eval_df=eval_df, F1_score=F1_score)
