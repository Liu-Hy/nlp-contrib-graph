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

df = pd.read_csv('train_all_sent.csv')
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
print(f'train_df has length: {len(train_df)}')
print(f'eval_df has length: {len(eval_df)}')

# Some sentences are in the 'related work' or 'conclusion' section, and should be masked out.
train_df = train_df[train_df['mask'] == 1]
eval_df = eval_df[eval_df['mask'] == 1]

# downsample the imbalanced training data
train_pos = train_df[train_df['labels'] == 1]
train_neg = train_df[train_df['labels'] == 0]
imbalance_ratio = len(train_neg) / len(train_pos)

# Create a ClassificationModel
model_args = ClassificationArgs1()

model_args.downsample = 1.0
model_args.normalize_ofs = True
model_args.out_learning_rate = 1e-4
model_args.scheduler = "constant_schedule_with_warmup"
model_args.warmup_steps = 200
model_args.reprocess_input_data = True
model_args.save_model_every_epoch = False
model_args.overwrite_output_dir = True
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = True
model_args.num_train_epochs = 8
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.do_lower_case = True

# Create a TransformerModel
model = ClassificationModel1(
    "bert",
    "allenai/scibert_scivocab_uncased",
    weight=[1, imbalance_ratio/model_args.downsample],
    args=model_args,
)
print(f'imbalance_ratio: {imbalance_ratio}')
print(f'class weight: {imbalance_ratio/model_args.downsample}')

model.train_model(train_df, eval_df=eval_df,
                    F1_score=sklearn.metrics.f1_score)

