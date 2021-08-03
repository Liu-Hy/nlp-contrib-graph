import logging

import pandas as pd
import sklearn
import random
from simpletransformers.classification import (
    ClassificationArgs1,
    ClassificationModel1,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('../interim/all_sent.csv')
df = df.drop(columns=['BIO', 'BIO_1', 'BIO_2', 'labels']).rename(
    columns={'bi_labels': 'labels'})
df['title'] = df['main_heading'] + ': ' + df['heading']
df.loc[((df['main_heading'] == df['heading']) | (
    pd.isnull(df['heading']))), 'title'] = df['main_heading']
df['title'] = df['title'].fillna('')
df['paper'] = df['topic'] + df['paper_idx'].astype(str)
ids = df["paper"].unique()
random.seed(1)
random.shuffle(ids)
bound = int(0.9*len(ids))

train_df = df.set_index("paper").loc[ids[:bound]].reset_index()
eval_df = df.set_index("paper").loc[ids[bound:]].reset_index()
train_df = train_df.sample(frac=1, random_state=1)

# Some sentences are in the 'related work' or 'conclusion' section, and should be masked out.
train_df = train_df[train_df['mask'] == 1]
eval_df = eval_df[eval_df['mask'] == 1]

train_pos = train_df[train_df['labels'] == 1]
train_neg = train_df[train_df['labels'] == 0]
imbalance_ratio = len(train_neg) / len(train_pos)

# Create a ClassificationModel
model_args = ClassificationArgs1()
model_args.use_early_stopping = True
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 2
model_args.early_stopping_consider_epochs = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_args.downsample = 1.0
model_args.normalize_ofs = True
model_args.out_learning_rate = 1e-4

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.output_dir = 'binary/'
model_args.best_model_dir = 'binary/best_model'
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.manual_seed = 1
model_args.fp16 = False
model_args.num_train_epochs = 7
model_args.train_batch_size = 16
model_args.use_multiprocessing = False  # set to True if cpu memory is enough
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.warmup_steps = 200
model_args.do_lower_case = True

# Create a TransformerModel
model = ClassificationModel1(
    "bert",
    "allenai/scibert_scivocab_uncased",
    weight=[1, imbalance_ratio/model_args.downsample],
    args=model_args,
)

model.train_model(train_df, eval_df=eval_df,
                    F1_score=sklearn.metrics.f1_score)

