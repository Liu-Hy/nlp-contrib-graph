import pandas as pd
from ast import literal_eval
import logging
from simpletransformers.ner import (
    NERArgs,
    NERModel,
)

labelset = ["B-p", "I-p", "B-n", "I-n", "O"]
type_ls = ['-p', '-n']

logging.basicConfig(level=logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

pos = pd.read_csv('pos_sent.csv')

data = []
for i in range(len(pos)):
    words = pos.iloc[i, 1].split(' ')
    tags = literal_eval(pos.iloc[i, 7])
    for j in range(len(words)):
        data.append([i, words[j], tags[j]])
df = pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])

model_args = NERArgs()
model_args.reprocess_input_data = True
model_args.save_model_every_epoch = False
model_args.use_multiprocessing = False
model_args.overwrite_output_dir = True
model_args.fp16 = False
model_args.manual_seed = 1
model_args.do_lower_case = False

def phrase_F1(ref, pred):
    return 0

model = NERModel(
    "bert",
    "../train_ner/specific/best_model",
    labels=labelset,
    args=model_args,
)

result, model_outputs, pred_label = model.eval_model(df, F1_score=phrase_F1)
# 'pred_label' is a list, where each element is the list of predicted labels for one sentence

p = [str(l) for l in pred_label]
pos = pd.read_csv('pos_sent.csv')
pos['BIO_1'] = p
pos.to_csv('pos_sent.csv')

def get_entity_spans(k, ls):
    spans = [[], []]
    for i in range(len(ls)):
        if ls[i] == 'B-p':
            for j in range(i+1, len(ls)):
                if ls[j] != 'I-p':
                    phrase = ' '.join(pos.loc[k, 'text'].split(' ')[i:j])
                    tup = (phrase, (i, j))
                    spans[0].append(tup)
                    break
                elif j == len(ls)-1:
                    phrase = ' '.join(pos.loc[k, 'text'].split(' ')[i:(j+1)])
                    tup = (phrase, (i, (j+1)))
                    spans[0].append(tup)
        elif ls[i] == 'B-n':
            for j in range(i+1, len(ls)):
                if ls[j] != 'I-n':
                    phrase = ' '.join(pos.loc[k, 'text'].split(' ')[i:j])
                    tup = (phrase, (i, j))
                    spans[1].append(tup)
                    break
                elif j == len(ls)-1:
                    phrase = ' '.join(pos.loc[k, 'text'].split(' ')[i:(j+1)])
                    tup = (phrase, (i, (j+1)))
                    spans[1].append(tup)
    return spans

phrase_ls = []
for k in range(len(pred_label)):
    phrase_ls.append(get_entity_spans(k, pred_label[k]))

a = pd.DataFrame(phrase_ls)
a.columns = ['predicates', 'subj/obj']
a = pd.concat([pos[['labels', 'text']], a], axis=1)
a = a.reset_index(drop=True)

a.to_csv('phrases.csv', index=False)
