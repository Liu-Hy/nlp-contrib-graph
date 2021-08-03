import os
import pandas as pd
import numpy as np
import sklearn
from scipy.special import softmax
from ast import literal_eval as load
import logging
from collections import Counter
from simpletransformers.ner import (
    NERArgs,
    NERModel,)
from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,)

labelset = ["B", "I", "O"]
type_ls = ['']

logging.basicConfig(level=logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

pos = pd.read_csv('pos_sent.csv')

data = []
for i in range(len(pos)):
    words = pos.loc[i, 'text'].split(' ')
    tags = load(pos.loc[i, 'BIO'])
    for j in range(len(words)):
        data.append([i, words[j], tags[j]])
df = pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])

def get_entity_spans(ls):
    # given a sequence of BIO tags, get the list of tuples representing spans of entities
    spans = []
    for type in type_ls:
        for i in range(len(ls)):
            st, ed = 0, 0
            if ls[i] == 'B'+type:
                st, ed = i, i + 1
                for j in range(i+1, len(ls)):
                    if ls[j] == 'I'+type:
                        ed += 1
                    else:
                        break
                spans.append((st, ed))
    spans = sorted(spans, key=lambda x: x[0])
    return spans

def phrase_F1(ref, pred):
    return 0

# get the paths of the ~96 submodels
base_dir = '../train_ner/'
model_ls = []
for i in range(32):
    folder = os.path.join(base_dir, 'identify_sub'+str(i))
    models = os.listdir(folder)
    models = [os.path.join(folder, model) for model in models if model[:11]
              == 'checkpoint-' and model[-1] != '1']
    model_ls += models

model_args = NERArgs()
model_args.reprocess_input_data = True
model_args.save_model_every_epoch = False
model_args.overwrite_output_dir = True
model_args.fp16 = False
model_args.manual_seed = 1
model_args.use_multiprocessing = False
model_args.do_lower_case = False

# get submodel predictions
each_pred = [[] for k in range(len(model_ls))]
for i in range(len(model_ls)):
    model = NERModel(
        'bert',
        model_ls[i],
        labels=labelset,
        args=model_args,
    )
    result, model_outputs, pred_label = model.eval_model(
        df, F1_score=phrase_F1)
    each_pred[i] = [get_entity_spans(label) for label in pred_label]

# ensemble the predictions
pred_ls = [[] for i in range(len(pos))]
for i in range(len(pos)):
    preds = []
    for j in range(len(model_ls)):
        preds += each_pred[j][i]
    cnt = Counter(preds)
    pred_ls[i] = [k for k, v in cnt.items() if v > 62]
    pred_ls[i] = sorted(pred_ls[i], key=lambda x: x[0])

p = [str(l) for l in pred_ls]
pos = pd.read_csv('pos_sent.csv')
pos['BIO_1'] = p
pos.to_csv('pos_sent.csv')

'''
Classify the phrases into predicates and terms
Ensembling is also used
'''
pred_ls2 = list(pos['BIO_1'].values)
pred_ls2 = [load(p) for p in pred_ls2]
ls1 = []
for i in range(len(pred_ls2)):
    for j in range(len(pred_ls2[i])):
        tup = pred_ls2[i][j]
        word_ls = pos.loc[i, 'text'].split(' ')
        word_ls.insert(tup[0], '<<')
        word_ls.insert(tup[1]+1, '>>')
        ls1.append([i, j, ' '.join(word_ls), 0])
dataframe = pd.DataFrame(ls1)
dataframe.columns = ['sent_idx', 'phrase_idx', 'text', 'labels']

model_args1 = ClassificationArgs()
model_args1.overwrite_output_dir = True
model_args1.reprocess_input_data = True
model_args1.manual_seed = 1
model_args1.fp16 = False
model_args1.use_multiprocessing = False  # True if cpu memory is enough
model_args1.do_lower_case = False

model_ls = []
for i in range(8):
    folder = os.path.join(base_dir, 'classify_sub'+str(i))
    models = os.listdir(folder)
    models = [os.path.join(folder, model)
              for model in models if model[:11] == 'checkpoint-']
    model_ls += models

each_pred = []
for i in range(len(model_ls)):
    model1 = ClassificationModel(
        "xlmroberta",
        model_ls[i],
        args=model_args1,
    )
    result, model_outputs, wrong_predictions = model1.eval_model(
        dataframe, F1_score=sklearn.metrics.f1_score)
    each_pred.append(model_outputs)

np.set_printoptions(precision=5)
m = np.zeros_like(each_pred[0])
for i in range(len(each_pred)):
    m = m + softmax(each_pred[i], axis=1)
m = m/len(each_pred)
pred = m.argmax(axis=1)

dataframe['pred'] = list(pred)

phrase_ls = [[[], []] for i in range(len(pos))]
for i in range(len(dataframe)):
    s_id = dataframe.loc[i, 'sent_idx']
    p_id = dataframe.loc[i, 'phrase_idx']
    st, ed = pred_ls2[s_id][p_id]
    word_ls = pos.loc[s_id, 'text'].split(' ')
    phrase = ' '.join(word_ls[st:ed])
    tup = (phrase, (st, ed))
    if dataframe.loc[i, 'pred'] == 0:
        phrase_ls[s_id][1].append(tup)
    else:
        phrase_ls[s_id][0].append(tup)

a = pd.DataFrame(phrase_ls)
a.columns = ['predicates', 'subj/obj']
a = pd.concat([pos[['labels', 'text']], a], axis=1)
a = a.reset_index(drop=True)

a.to_csv('phrases.csv', index=False)
