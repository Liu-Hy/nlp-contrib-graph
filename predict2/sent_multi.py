import logging
import pandas as pd
from simpletransformers.classification import (
    ClassificationArgs1,
    ClassificationModel1,
)

logging.basicConfig(level=logging.WARNING)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

df = pd.read_csv('pos_sent.csv')
df['title'] = df['main_heading'] + ': ' + df['heading']
df.loc[((df['main_heading'] == df['heading']) | (pd.isnull(df['heading']))), 'title'] = df['main_heading']
df['title'] = df['title'].fillna('')
df['labels']='results'

label_list = ['results',
               'ablation-analysis',
               'method',
               'baselines',
               'dataset',
               'hyper-setup',
               'experiments',
               'research-problem',
               'tasks']

model_args = ClassificationArgs1()

model_args.labels_list = label_list
model_args.normalize_ofs = True
model_args.overwrite_output_dir = True
model_args.reprocess_input_data = True
model_args.manual_seed = 1
model_args.fp16 = False
model_args.use_multiprocessing = False
model_args.do_lower_case = True 

model = ClassificationModel1(
    "bert",
    "../train_sent/multi/best_model",
    num_labels=9,
    args=model_args,
)

def F1_score(ref, pred):
    print('This is prediction mode. Please neglect the evaluation score.')
    return 0.5

result, model_outputs, wrong_predictions = model.eval_model(df, F1_score=F1_score)

preds = list(model_outputs.argmax(axis=1))
preds = [label_list[p] for p in preds]
df['labels']=preds
df['paper']=df['topic']+df['paper_idx'].astype(str)

# decide whether an article has the unit 'model' or 'approach', if it has a sentence classified as 'method'
def classify_method(df):
    headings = ''.join(str(heading).lower() for heading in df['main_heading'].unique())
    if 'model' in headings:
        return 'model'
    elif 'approach' in headings:
        return 'approach'
    text = ''.join(df['text']).lower()
    if 'system' in text or 'architecture' in text:
        return 'model'
    if 'approach' in text and \
        not ('existing approach' in text or 'previous approach' in text or 'former approach' in text):
        return 'approach'
    return 'model'

terms = set(['keras', 'pytorch', 'torch', 'tensorflow', 'theano', 'cuda', 'cpu', 'cpus',
              'gpu', 'gpus', 'nvidia', 'intel', 'gtx', 'geforce', 'tesla','titan',
             'nltk', 'spacy', 'corenlp', 'stanford', 'tokenizer',
              'run'])

# decide whether an article has the unit 'hyperparameters' or 'experimental-setup', if it has a sentence classified as 'hyper-setup'
def classify_hyper_setup(df): # df should contain sentences from the same paper
    df=df[df['labels']=='hyper-setup']
    text=' '.join(list(df['text'].values))
    if len(terms.intersection(set(text.lower().split()))) > 0:
        return 'experimental-setup'
    else:
        return 'hyperparameters'

# judge if a sentence is in 'Code' unit
def judge_code(s):
    s=s.lower()
    if 'github.' in s or 'github .' in s:
        return 1
    elif ('https:' in s or 'http:' in s) and ('available' in s or 'code' in s):
        return 1
    else:
        return 0

# apply the above rules to subdivide similar units
df1=df[df['labels']=='method']
paper_ls=list(df1['paper'].unique())
for paper in paper_ls:
    df.loc[(df['paper']==paper)&(df['labels']=='method'),'labels']=classify_method(df[df['paper']==paper])

df2=df[df['labels']=='hyper-setup']
paper_ls=list(df2['paper'].unique())
for paper in paper_ls:
    df.loc[(df['paper']==paper)&(df['labels']=='hyper-setup'),'labels']=classify_hyper_setup(df[df['paper']==paper])
# identify code sentences and override labels
for i in range(len(df)):
    if judge_code(df.iloc[i,1])==1:
        df.at[i,'labels']='code'

df.to_csv('pos_sent.csv', index=False)
