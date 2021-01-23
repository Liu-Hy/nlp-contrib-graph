# search in the hyperparameter space with W&B sweep
import logging

import pandas as pd
import sklearn
import random
import wandb
from simpletransformers.classification import (
    ClassificationArgs1,
    ClassificationModel1,
)

'''
Some hyperparameters are customized for this sentence classification task:
downsample: 1/downsample of the negative(majority class)examples will be remained after downsampling
            class weight will be set to a corresponding proper value
normalize_ofs: whether or not to normalize the offset(position) features of sentences to have a mean of 0 and std of 0.5
out_learning_rate: the learning rate of the final 2-layer classifer, which is set larger than the lr to fine-tune the BERT model

other hyperparameters that are tried out manually:
freeze: whether to freeze the first 9 layers of BERT during training
share_weight: whether to use one BERT or two seperate BERTs to encode the sentence and the title
'''
sweep_config = {
    "method": "bayes",  # grid, random
    "metric": {"name": "F1_score", "goal": "maximize"},
    "parameters": {
        "downsample": {"values": [1.0]},
        "normalize_ofs": {"values": [True, False]},
        "out_learning_rate": {"min": 2e-5, "max": 5e-4},
    },
}
#4.91239e-4 4 epochs F1:62
sweep_id = wandb.sweep(sweep_config, project="Simple Sweep")

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('../interim/all_sent.csv')
df = df.drop(columns=['BIO', 'BIO_1', 'BIO_2', 'labels']).rename(
    columns={'bi_labels': 'labels'})
df['title'] = df['main_heading'] + ': ' + df['heading']
df.loc[((df['main_heading']==df['heading'])|(pd.isnull(df['heading']))),'title']=df['main_heading']
df['title'] = df['title'].fillna('')
#df["text"] = df["main_heading"]+': '+df["text"]
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

# remove the skipped sentences from the training set
train_df = train_df[train_df['mask'] == 1]
# split the eval set into two parts. Only predict on the unmasked part.
#skpd_eval = eval_df[eval_df['mask'] == 0]
#print('The number of skipped samples in eval set is: ', len(skpd_eval))
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
model_args.evaluate_during_training_verbose = True

model_args.overwrite_output_dir = True
model_args.scheduler = "linear_schedule_with_warmup"
model_args.warmup_steps = 200
model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.no_save = True
model_args.evaluate_during_training = True #
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = True
model_args.num_train_epochs = 7
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.do_lower_case = True #when using uncased model
model_args.wandb_project = "Simple Sweep"

def train():
    # Initialize a new wandb run
    wandb.init()

    # Create a TransformerModel
    model = ClassificationModel1(
        "bert",
        "allenai/scibert_scivocab_uncased",
        weight=[1, imbalance_ratio/model_args.downsample],
        args=model_args,
        sweep_config=wandb.config,
    )
    print(f'imbalance_ratio: {imbalance_ratio}')
    print(f'class weight: {imbalance_ratio/model_args.downsample}')
    # Train the model
    model.train_model(train_df, eval_df=eval_df, F1_score=sklearn.metrics.f1_score)

    # Evaluate the model
    # model.eval_model(eval_df, F1_score=sklearn.metrics.f1_score)

    # Sync wandb
    wandb.join()

wandb.agent(sweep_id, train)
