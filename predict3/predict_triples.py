import logging
from ast import literal_eval as load
import pandas as pd
import numpy as np
from ast import literal_eval

from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

logging.basicConfig(level=logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df0 = pd.read_csv('phrases.csv')
df0.insert(loc=0, column='idx', value=np.arange(len(df0)))
sent_num = len(df0)

# generate a dataframe of possible type A triples to be classified, and so on
def convert_A(arr):
    trip_ls = []
    ls = []
    for i in range(len(arr)):
        pre_ls = literal_eval(arr[i, 3])
        np_ls = literal_eval(arr[i, 4])
        p_idx = [(1, *p[1]) for p in pre_ls]
        n_idx = [(0, *n[1]) for n in np_ls]
        for p in range(len(p_idx)):
            for j in range(len(n_idx)-1):
                for k in range(j+1, len(n_idx)):
                    trip = [np_ls[j][0], pre_ls[p][0], np_ls[k][0]]
                    trip_ls.append(trip)
                    word_ls = arr[i, 2].split(' ')
                    indx = sorted([p_idx[p], n_idx[j], n_idx[k]],
                                  key=lambda x: x[1])
                    for w in range(len(indx)):
                        if indx[w][0] == 1:
                            word_ls.insert(indx[w][1]+2*w, '<<')
                            word_ls.insert(indx[w][2]+2*w+1, '>>')
                        else:
                            word_ls.insert(indx[w][1]+2*w, '[[')
                            word_ls.insert(indx[w][2]+2*w+1, ']]')
                    flg = 0
                    ls.append([int(arr[i, 0]), ' '.join(word_ls), flg])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['idx', 'text', 'labels']
    return dataframe, trip_ls

def convert_B(arr):
    ls = []
    trip_list = []
    A = pd.read_csv('A.csv').values
    for i in range(len(arr)):
        a_ls = load(A[i,0])
        lth = len(load(arr[i, 4]))
        for p1 in range(lth-1):
            for p2 in range(p1+1, p1+2):
                phrase1 = load(arr[i, 4])[p1]
                phrase2 = load(arr[i, 4])[p2]
                possible = 1
                # terms that co-occcur in type A cannot be in type B triples
                for a in a_ls:
                    if (a[0] == phrase1[0] and a[2] == phrase2[0]) or (a[0] == phrase2[0] and a[2] == phrase1[0]):
                        possible = 0
                        break
                if possible == 1:
                    word_ls = arr[i, 2].split(' ')
                    triple = [phrase1[0], 'has', phrase2[0]]
                    trip_list.append(triple)
                    word_ls.insert(phrase1[1][0], '<<')
                    word_ls.insert(phrase1[1][1]+1, '>>')
                    word_ls.insert(phrase2[1][0]+2, '[[')
                    word_ls.insert(phrase2[1][1]+3, ']]')
                    flg = 0
                    trip_ls = []
                    ls.append([int(arr[i, 0]), phrase1[0], phrase2[0],
                               trip_ls, ' '.join(word_ls), flg])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['idx', 'phrase1',
                         'phrase2', 'triples', 'text', 'labels']
    return dataframe, trip_list

def convert_C(arr):
    trip_list = []
    ls = []
    for i in range(len(arr)):
        uni_name = arr[i, 1]
        uni_name = (uni_name[0].upper()+uni_name[1:]).replace('-', ' ')
        if arr[i, 3] != '[]' and arr[i, 4] != '[]':
            pre = load(arr[i, 3])[0]
            np = load(arr[i, 4])[0]
            if pre[1][0] < np[1][0]:
                triple = [uni_name, pre[0], np[0]]
                trip_list.append(triple)
                word_ls = arr[i, 2].split(' ')
                word_ls.insert(pre[1][0], '<<')
                word_ls.insert(pre[1][1]+1, '>>')
                word_ls.insert(np[1][0]+2, '[[')
                word_ls.insert(np[1][1]+3, ']]')
                unit = arr[i, 1]
                unit = (unit[0].upper()+unit[1:]).replace('-', ' ')
                unit_ls = ['[[']+(unit.split(' '))+[']]']
                word_ls = unit_ls+[':']+word_ls
                flg = 0
                trip_ls = []
                ls.append([int(arr[i, 0]), unit, pre[0], np[0],
                           trip_ls, ' '.join(word_ls), flg])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['idx', 'info_unit',
                         'pre', 'np', 'triples', 'text', 'labels']
    return dataframe, trip_list

def convert_D(arr):
    trip_list = []
    ls = []
    for i in range(len(arr)):
        if arr[i, 4] != '[]':
            if arr[i, 3] == '[]' or ((arr[i, 3] != '[]') and (load(arr[i, 3])[0][1][0] > load(arr[i, 4])[0][1][0])):
                np = load(arr[i, 4])[0]
                uni_name = arr[i, 1]
                uni_name = (uni_name[0].upper()+uni_name[1:]).replace('-', ' ')
                triple = [uni_name, 'has', np[0]]
                trip_list.append(triple)
                word_ls = arr[i, 2].split(' ')
                word_ls.insert(np[1][0], '[[')
                word_ls.insert(np[1][1]+1, ']]')
                unit = arr[i, 1]
                unit = (unit[0].upper()+unit[1:]).replace('-', ' ')
                unit_ls = ['[[']+(unit.split(' '))+[']]']
                word_ls = unit_ls+[':']+word_ls
                flg = 0
                trip_ls = []
                ls.append([int(arr[i, 0]), unit, np[0],
                           trip_ls, ' '.join(word_ls), flg])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['idx', 'info_unit', 'np', 'triples', 'text', 'labels']
    return dataframe, trip_list

convert_ls = [convert_A, convert_B, convert_C, convert_D]

model_args = ClassificationArgs()

model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = False
model_args.do_lower_case = True

def triple_F1(ref, pred):
    return 0

df0 = df0.values
lt = ['A','B','C','D']
# predict using the model for each type, and store the predicted triples
for i in range(4):
    df, trip_list = convert_ls[i](df0)
    model = ClassificationModel(
        "bert",
        "../train_rel/"+lt[i]+"/best_model",
        args=model_args,
    )
    result, model_outputs, wrong_predictions = model.eval_model(
        df, F1_score=triple_F1)
    preds = list(model_outputs.argmax(axis=1))
    df['preds']=preds
    df['cand']=trip_list
    df.loc[df['preds']==0,'cand']=None
    if i==1:
        for j in range(len(df)):
            if df.loc[j, 'preds'] == 2:
                df.loc[j, 'cand'][1] = 'name'
    data=[]
    for k in range(sent_num):
        temp = list(df[df['idx']==k]['cand'])
        temp = [t for t in temp if t]
        data.append(str(temp))
    data=pd.DataFrame(data,columns=['triple_'+lt[i]])
    data.to_csv(lt[i]+'.csv', index=False)

A = pd.read_csv('A.csv')
B = pd.read_csv('B.csv')
C = pd.read_csv('C.csv')
D = pd.read_csv('D.csv')

t = pd.read_csv('phrases.csv')
t = t.reset_index(drop=True)
t = pd.concat([t, A, B, C, D], axis=1)
pos = pd.read_csv('pos_sent.csv')
bu = pos[['topic', 'paper_idx', 'idx']]
t = pd.concat([t, bu], axis=1)
t.to_csv('triples.csv')
