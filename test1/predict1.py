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

df = pd.read_csv('all_sent.csv')
df = df.drop(columns=['BIO', 'BIO_1', 'BIO_2']).rename(
    columns={'labels': 'predictions'}).rename(columns={'bi_labels': 'labels'})
df['title'] = df['main_heading'] + ': ' + df['heading']
df.loc[((df['main_heading'] == df['heading']) | (
    pd.isnull(df['heading']))), 'title'] = df['main_heading']
df['title'] = df['title'].fillna('')
#df["text"] = df["main_heading"]+': '+df["text"]
df['paper'] = df['topic'] + df['paper_idx'].astype(str)

#eval_df = eval_df[eval_df['mask'] == 1]

# downsample the imbalanced training data

# Create a ClassificationModel

model_args = ClassificationArgs1()

model_args.normalize_ofs = True
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = True
model_args.do_lower_case = True  # when using uncased model

# Create a TransformerModel
model = ClassificationModel1(
    "bert",
    "trained_model_directory",
    args=model_args,
)

# make predictions on the unlabeled sentences
# here the result makes no sense

result, model_outputs, wrong_predictions = model.eval_model(df, F1_score=sklearn.metrics.f1_score)
# model_outputs is a numpy array of shape (number of samples, 2)
predictions = model_outputs.argmax(axis=1)
# select the sentences that are predicted positive, to be the input for subtask 2
mask = df['mask'].values
# sentences that are masked out are forced to be negative
predictions = predictions * mask
pos = df[predictions == 1]
pos.to_csv('pos_sent.csv', index=False)

