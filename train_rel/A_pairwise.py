import logging
import pandas as pd
import numpy as np
from ast import literal_eval

from simpletransformers.classification import (
    ClassificationArgs,
    ClassificationModel,
)

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('../interim/triples.csv')
df = df.rename(columns={'idx': 'indx'})
df.insert(loc=0, column='idx', value=np.arange(len(df)))
df = df.sample(frac=1, random_state=1)
bound = int(0.9*len(df))
t_df = df[:bound]
e_df = df[bound:]

def convert(arr):
    ls = []
    for i in range(len(arr)):
        pre_ls = literal_eval(arr[i, 3])
        np_ls = literal_eval(arr[i, 4])
        for pre in pre_ls:
            for np in np_ls:
                word_ls = arr[i, 2].split(' ')
                if pre[1][0] < np[1][0]:
                    word_ls.insert(pre[1][0], '<<')
                    word_ls.insert(pre[1][1]+1, '>>')
                    word_ls.insert(np[1][0]+2, '[[')
                    word_ls.insert(np[1][1]+3, ']]')
                else:
                    word_ls.insert(pre[1][0], '<<')
                    word_ls.insert(pre[1][1]+1, '>>')
                    word_ls.insert(np[1][0], '[[')
                    word_ls.insert(np[1][1]+1, ']]')
                flg = 0
                for tp in literal_eval(arr[i, 5]):
                    if pre[0] == tp[1] and np[0] in tp[::2]:
                        flg = 1
                        break
                ls.append([int(arr[i, 0]), arr[i, 1], ' '.join(word_ls), flg])
    dataframe = pd.DataFrame(ls)
    dataframe.columns = ['idx', 'info_unit', 'text', 'labels']
    return dataframe

t_df = t_df.values
e_df = e_df.values

train_df = convert(t_df)
eval_df = convert(e_df)

train_df.to_csv('train.csv')
eval_df.to_csv('eval.csv')

# dowmsample the negative samples
train_pos = train_df[train_df['labels'] == 1]
train_neg = train_df[train_df['labels'] == 0]
train_neg = train_neg.sample(n=len(train_pos), random_state=1)
train_df = pd.concat([train_pos, train_neg])
train_df = train_df.sample(frac=1, random_state=1)

shp = []
for i in range(len(e_df)):
    shp.append((len(literal_eval(
        e_df[i, 3])) * len(literal_eval(e_df[i, 4])), len(literal_eval(e_df[i, 3]))))

model_args = ClassificationArgs()
model_args.use_early_stopping = True
model_args.early_stopping_metric = "F1_score"
model_args.early_stopping_metric_minimize = False
model_args.early_stopping_patience = 2
model_args.early_stopping_consider_epochs = True
model_args.evaluate_during_training = True
model_args.evaluate_during_training_verbose = True

model_args.reprocess_input_data = True
model_args.overwrite_output_dir = True
model_args.output_dir = 'Ap/'
model_args.best_model_dir = 'Ap/best_model'
model_args.save_model_every_epoch = True
model_args.save_steps = -1
model_args.manual_seed = 1
model_args.use_multiprocessing = False
model_args.fp16 = False
model_args.num_train_epochs = 8
model_args.train_batch_size = 16
model_args.gradient_accumulation_steps = 4
model_args.learning_rate = 1e-5
model_args.scheduler = "polynomial_decay_schedule_with_warmup"
model_args.polynomial_decay_schedule_power = 0.5
model_args.warmup_steps = 200
model_args.do_lower_case = True

def triple_F1(ref, pred):
    # pred is an ndarray with shape (Number of test samples, )
    TP = FP = FN = 0
    st = 0
    # calculate and report the metrics for candidate pairs
    tp = len(np.where((pred == 1) & (ref == 1))[0])
    fp = len(np.where((pred == 1) & (ref == 0))[0])
    fn = len(np.where((pred == 0) & (ref == 1))[0])
    pc = tp / (tp + fp)
    rc = tp / (tp + fn)
    f1 = 2 * pc * rc / (pc + rc)
    switch = 0
    for i in range(len(e_df)):
        if shp[i][0] != 0:
            table = pred[st: st+shp[i][0]].reshape(shp[i][1], -1)
            st += shp[i][0]
            pre_ls, np_ls = literal_eval(e_df[i, 3]), literal_eval(e_df[i, 4])
            ref_ls = literal_eval(e_df[i, 5])
            pred_ls = []
            for j in range(len(table)):
                if sum(table[j]) >= 2:
                    if sum(table[j]) == 2:
                        a, b = sorted(np.where(table[j] == 1)[0])
                        pred_ls.append(
                            [np_ls[a][0], pre_ls[j][0], np_ls[b][0]])
                else:# SHOULD IT BE THIS WAY??
                    left = [k for k in range(
                        len(np_ls)) if np_ls[k][1][0] < pre_ls[j][1][0]]
                    right = [k for k in range(
                        len(np_ls)) if np_ls[k][1][0] > pre_ls[j][1][0]]
                    if left and right:
                        for x in left:
                            for y in right:
                                pred_ls.append(
                                    [np_ls[x][0], pre_ls[j][0], np_ls[y][0]])
            TPs = [i for i in pred_ls if i in ref_ls]
            FPs = [i for i in pred_ls if i not in ref_ls]
            FNs = [i for i in ref_ls if i not in pred_ls]
            if FPs or FNs:
                word_l = e_df[i, 2].split(' ')
                p_idx = [(1, *p[1]) for p in literal_eval(e_df[i, 3])]
                n_idx = [(0, *n[1]) for n in literal_eval(e_df[i, 4])]
                indx = sorted(p_idx + n_idx, key=lambda x: x[1])
                for k in range(len(indx)):
                    if indx[k][0] == 1:
                        word_l.insert(indx[k][1]+2*k, '<<')
                        word_l.insert(indx[k][2]+2*k+1, '>>')
                    else:
                        word_l.insert(indx[k][1]+2*k, '[[')
                        word_l.insert(indx[k][2]+2*k+1, ']]')
                print('text: ' + ' '.join(word_l))
            if FPs:
                print('False positive:')
                FP_str = '\n'.join(['||'.join(FP) for FP in FPs])
                print(FP_str)
            if FNs:
                print('False negative:')
                FN_str = '\n'.join(['||'.join(FN) for FN in FNs])
                print(FN_str)
            if FPs or FNs:
                print('\n')
            if FPs and FNs:
                for trip1 in FPs:
                    for trip2 in FNs:
                        if trip1[0] == trip2[2] and trip1[1] == trip2[1] and trip1[2] == trip2[0]:
                            switch += 1
            TP += len(TPs)
            FP += len(FPs)
            FN += len(FNs)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    print('-' * 30)
    print(f'number of switched tuple is: {switch}')
    print(
        f'Candidate pair classification: precision {pc}, recall {rc}, F1 {f1}')
    print(
        f'Triple extraction: precision {precision}, recall {recall}, F1 {F1}')
    print('-' * 30)
    return F1


# Create a TransformerModel
model = ClassificationModel(
    "bert",
    "allenai/scibert_scivocab_uncased",
    args=model_args,
)

model.train_model(train_df, eval_df=eval_df,
                    F1_score=triple_F1)

#result, model_outputs, wrong_predictions = model.eval_model(
    #eval_df, F1_score=triple_F1)

