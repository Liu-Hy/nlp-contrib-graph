import logging
import pandas as pd
import sklearn
from simpletransformers.classification import (
    ClassificationArgs1,
    ClassificationModel1,
)

logging.basicConfig(level=logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('all_sent.csv')
df = df.drop(columns=['BIO_2', 'labels']).rename(
    columns={'bi_labels': 'labels'})
df['title'] = df['main_heading'] + ': ' + df['heading']
df.loc[((df['main_heading'] == df['heading']) | (
    pd.isnull(df['heading']))), 'title'] = df['main_heading']
df['title'] = df['title'].fillna('')

model_args = ClassificationArgs1()

model_args.normalize_ofs = True
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.use_multiprocessing = False
model_args.manual_seed = 1
model_args.fp16 = False
model_args.do_lower_case = True

# Create a TransformerModel
model = ClassificationModel1(
    "bert",
    "../train_sent/binary/best_model",
    args=model_args,
)

result, model_outputs, wrong_predictions = model.eval_model(
    df, F1_score=sklearn.metrics.f1_score)

predictions = model_outputs.argmax(axis=1)
# select the sentences that are predicted positive, to be the input for subtask 2
mask = df['mask'].values
# sentences in the 'related work' or 'conclusion' sections are forced to be negative
predictions = predictions * mask
pos = df[predictions == 1]
pos.to_csv('pos_sent.csv', index=False)
