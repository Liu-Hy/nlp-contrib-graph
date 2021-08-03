import pandas as pd
import numpy as np
from ast import literal_eval as load
import logging
from simpletransformers.ner import (
    NERArgs,
    NERModel,
)

labelset = ["B", "I", "O"]
type_ls = ['']

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

pos = pd.read_csv('../interim/pos_sent.csv')
col_name = pos.columns[6]
pos = pos.dropna(axis=0, subset=[col_name])
pos = pos.drop('bi_labels', axis=1)

def convert(df):
    data = []
    for i in range(len(df)):
        words = df.iloc[i,1].split(' ')
        tags = load(df.iloc[i, 6])
        for j in range(len(words)):
            data.append([i, words[j], tags[j]])
    frame = pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])
    return frame

model_args = NERArgs()

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.fp16 = False
model_args.save_model_every_epoch = True
model_args.save_eval_checkpoints = False
model_args.save_steps = -1
model_args.manual_seed = 1
model_args.use_multiprocessing = False
model_args.num_train_epochs = 4
model_args.train_batch_size = 8
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 5e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.do_lower_case = False

def get_entity_spans(ls):
    # given a sequence of BIO taggs, get the list of tuples representing spans of entities
    spans = []
    for type in type_ls:
        for i in range(len(ls)):
            st, ed = 0, 0
            if ls[i] == 'B' + type:
                st, ed = i, i + 1
                for j in range(i+1, len(ls)):
                    if ls[j] == 'I' + type:
                        ed += 1
                    else:
                        break
                spans.append((st, ed))
    spans = sorted(spans, key=lambda x: x[0])
    return spans

for i in range(32):
    idx = np.random.randint(len(pos), size=len(pos))
    print(idx[:30])
    model_args.output_dir = 'identify_sub'+str(i)
    model_args.best_model_dir = 'identify_sub'+str(i)+'/best_model'
    model = NERModel(
        'bert',
        'allenai/scibert_scivocab_cased',
        labels=labelset,
        args=model_args,
    )
    p_train = pos
    p_train = p_train.iloc[idx]
    train_df=convert(p_train)
    model.train_model(train_df)

