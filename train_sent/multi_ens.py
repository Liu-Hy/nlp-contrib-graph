import logging
import pandas as pd
import numpy as np
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
df.loc[((df['main_heading']==df['heading'])|(pd.isnull(df['heading']))),'title']=df['main_heading']
df['title'] = df['title'].fillna('')
df['paper'] = df['topic'] + df['paper_idx'].astype(str)
df.loc[(df['labels']=='hyperparameters')|(df['labels']=='experimental-setup'), 'labels']='hyper-setup'
df.loc[(df['labels']=='model')|(df['labels']=='approach'), 'labels']='method'

from collections import Counter
Counter(df['labels'].values)

df = df[(df['mask'] == 1)&(df['labels']!='code')]
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
print(f'df has length: {len(df)}')

model_args = ClassificationArgs1()
model_args.labels_list = label_list
model_args.downsample=1.0
model_args.normalize_ofs=True
model_args.out_learning_rate=5e-5
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.save_model_every_epoch = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1 
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = False  # set to True if cpu memory is enough
model_args.num_train_epochs = 20
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.warmup_steps = 500
model_args.do_lower_case = True

for i in range(8):
    idx = np.random.randint(len(df), size=len(df))
    print(idx[:30])
    model_args.output_dir='multi_sub'+str(i)
    model_args.best_model_dir = 'multi_sub'+str(i)+'/best_model'
    model = ClassificationModel1(
        "bert",
        "allenai/scibert_scivocab_uncased",
        weight=weight_list,
        num_labels=9,
        args=model_args,
    )
    t_d=df
    t_d=t_d.iloc[idx]
    print(f'weight list: {weight_list}')
    model.train_model(t_d)



