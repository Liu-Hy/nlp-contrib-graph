# search in the hyperparameter space with W&B sweep
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

# remove the skipped sentences from the training set
train_df = train_df[train_df['mask'] == 1]
# split the eval set into two parts. Only predict on the unmasked part.
#skpd_eval = eval_df[eval_df['mask'] == 0]
#print('The number of skipped samples in eval set is: ', len(skpd_eval))
eval_df = eval_df[eval_df['mask'] == 1]


# Create a ClassificationModel

model_args = ClassificationArgs1()

model_args.downsample = 1.0
model_args.normalize_ofs = True
model_args.out_learning_rate = 1e-4
model_args.overwrite_output_dir = True
model_args.scheduler = "constant_schedule_with_warmup"
model_args.warmup_steps = 200
model_args.reprocess_input_data = True
model_args.save_model_every_epoch = False
model_args.overwrite_output_dir = True
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = True
model_args.num_train_epochs = 2
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.do_lower_case = True  # when using uncased model

# Create a TransformerModel
model = ClassificationModel1(
    "bert",
    "outputs",
    weight=[1, imbalance_ratio/model_args.downsample],
    args=model_args,
)

# Train the model
model.train_model(train_df, eval_df=eval_df,
                  F1_score=sklearn.metrics.f1_score)

# Evaluate the model
model.eval_model(eval_df, F1_score=sklearn.metrics.f1_score)
