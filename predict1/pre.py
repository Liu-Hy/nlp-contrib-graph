'''
Data preprocessing and cleaning
get a dataframe of all sentences, together with relevant information to the tasks
'''
import os
import re
import math
import pandas as pd

base_dir = os.path.join('..', 'test_data', 'input')
sep = os.path.sep

def get_dir(topic_ls=None, paper_ls=None):
    # Get the list of paper directories
    dir_ls = []
    if topic_ls is None:
        topic_ls = os.listdir(base_dir)
        topic_ls.remove('README.md')
    if paper_ls is None:
        for topic in topic_ls:
            paper_ls = os.listdir(os.path.join(base_dir, topic))
            for i in paper_ls:
                dir_ls.append(os.path.join(base_dir, topic, i))
    else:
        for topic in topic_ls:
            for i in paper_ls:
                dir_ls.append(os.path.join(base_dir, topic, str(i)))
    return dir_ls

def get_file_path(dirs):
    # Get the relevant files from each directory of paper.
    rx = '(.*Stanza-out.txt$)'
    file_path = []
    for dir in dirs:
        for file in os.listdir(dir):
            res = re.match(rx, file)
            if res:
                new = os.path.join(dir, file)
                break
        file_path.append(new)
    return file_path

def is_heading(line):
    # Determine if a line is a heading
    ls = line.split(' ')
    # Titles rarely end with these words
    False_end = ['by', 'as', 'in', 'and', 'that']
    if len(ls) < 10 and ls[-1] not in False_end:
        rx = '^[A-Z][^?]*[^?:]$|^title$|^abstract$'  # regex heuristic rules
        res = re.match(rx, line)
        return True if res else False
    return False

def is_main_heading(line, judge_mask=False):
    '''
    Assume that the line is a heading, determine if it is a main heading
    A main heading is either a typical main section heading, or it contains lexical cues that are considered important for judgement.
    '''
    if len(line.split(' ')) <= 4:
        if judge_mask:    # if the aim is to judge whether the sentence should be skipped
            lex_cue = 'background|related|conclusion'  # |related work
        else:
            lex_cue = 'title|abstract|introduction|background|related|conclusion|model|models|method|methods|approach|architecture|system|application|experiment|experiments|experimental setup|implementation|hyperparameters|training|result|results|ablation|baseline|evaluation'  # |related work
        exp = re.compile(lex_cue)
        # Decide if it is a main heading
        return True if exp.search(line.lower()) else False
    else:
        return False

# Determin if a sentence conforms to a specific case method.
# Their are three case methods in all, eg: Attention Is All You Need; ATTENTION IS ALL YOU NEED; Attention is all you need.

def check_case(line, flag):
    if flag == 1:
        match = re.search(r'[a-z]', line)
        if match:
            return False
        return True
    else:
        wd_num = 0
        words = line.split(' ')
        if flag == 0:
            stp_wd = ['a', 'an', 'and', 'the', 'or', 'if', 'by', 'as', 'to', 
            'of', 'for', 'in', 'on', 'but', 'via', 'nor', 'with']
            if not words[0].istitle():
                wd_num += 1
            if len(words) > 1:
                if not words[-1].istitle():
                    wd_num += 1
                for word in words[1:-1]:
                    if not word.istitle() and word not in stp_wd:
                        wd_num += 1
            return wd_num <= math.ceil(len(words)/5)
        if flag == 2:
            if not words[0].istitle():
                wd_num += 1
            for word in words[1:]:
                if re.match(r'[A-Z]', word):
                    wd_num += 1
            return wd_num <= math.ceil(len(words)/3)

# read the relevant files from the folder of one paper, and produce a data table for that paper.
def load_paper_sentence(sent_path):  # (sent_path, label_path)
    sent = []
    count = [0, 0, 0]
    task, index = sent_path.split(sep)[-3:-1]
    # Decide the case type of the titles in this paper, by counting over the main headings and find the maximum
    with open(sent_path, 'r') as f:
        while(True):
            line = f.readline().rstrip("\n")
            if line:
                if is_heading(line) and is_main_heading(line):
                    for m in range(3):
                        if check_case(line, m):
                            count[m] += 1
            else:
                break
    ocr_path = sent_path[:-14]+'Grobid-out.txt'
    with open(ocr_path, 'r') as f:
        fl = f.readlines()
    title_ls = []
    for i in range(len(fl)):
        if fl[i] == '\n':
            if i < (len(fl)-1):
                title_ls.append(fl[i+1].rstrip())
        if fl[i].rstrip().lower() in ['title', 'abstract', 'introduction']:
            title_ls.append(fl[i].rstrip())
    with open(sent_path, 'r') as f:
        i = 0
        flg = count.index(max(count))
        # two string buffers, storing the heading and the main heading respectively
        heading, main_h = '', ''
        ofs1 = ofs3 = 0
        while(True):
            i += 1
            line = f.readline().rstrip("\n")
            if line:
                if line in title_ls:
                    ofs3 = 0
                else:
                    ofs3 += 1
                if is_heading(line) and check_case(line, flg):
                    heading = line    # update the heading buffer

                    if is_main_heading(line):
                        ofs1 = 0
                        main_h = line    # update the main heading buffer too
                        # The line itself is a main heading, no heading needs to be stored.
                        sent.append(
                            [i, line, '', '', task, index, None, None, None, ofs1, 0, i-1, 0, ofs3, 0, 1, 0, None])
                    else:
                        ofs1 += 1
                        # for plain headings, store the main heading it belongs to.
                        # judge if it should be masked
                        if is_main_heading(main_h, judge_mask=True):
                            sent.append([i, line, main_h, '', task,
                                         index, None, None, None, ofs1, 0, i-1, 0, ofs3, 0, 0, 0, None])
                        else:
                            sent.append([i, line, main_h, '', task,
                                         index, None, None, None, ofs1, 0, i-1, 0, ofs3, 0, 1, 0, None])
                else:
                    # For plain text line, store both the heading and the main heading.
                    ofs1 += 1
                    if is_main_heading(main_h, judge_mask=True):
                        sent.append([i, line, main_h, heading, 
                                     task, index, None, None, None, ofs1, 0, i-1, 0, ofs3, 0, 0, 0, None])
                    else:
                        sent.append([i, line, main_h, heading, 
                                     task, index, None, None, None, ofs1, 0, i-1, 0, ofs3, 0, 1, 0, None])
            else:
                break
    for i in range(1, len(sent)):
        if sent[i][9] == 0:
            sof = sent[i-1][9]
            if sof > 1:
                for j in range(i-sof, i):
                    sent[j][10] = sent[j][9]/sof
        if sent[i][13] == 0:
            sof = sent[i-1][13]
            if sof > 1:
                for j in range(i-sof, i):
                    sent[j][14] = sent[j][13]/sof
        if i == len(sent)-1:
            sof = sent[i][9]
            if sof > 1:
                for j in range(i-sof+1, i+1):
                    sent[j][10] = sent[j][9]/sof
            sof = sent[i][13]
            if sof > 1:
                for j in range(i-sof+1, i+1):
                    sent[j][14] = sent[j][13]/sof
        sent[i][12] = sent[i][11]/len(sent)
    for i in range(len(sent)):
        sent[i][6] = ['O'] * len(sent[i][1].split(' '))
        sent[i][8] = sent[i][7] = sent[i][6]
    return sent

def load_data_sentence(file_path):
    # Get the data table of all the papers in file_path
    data = []
    for sentence_path in file_path:
        paper_data = load_paper_sentence(sentence_path)
        data += paper_data
    return data

dirs = get_dir()
file_path = get_file_path(dirs)
data = load_data_sentence(file_path)

df = pd.DataFrame(data)
df.columns = ['idx', 'text', 'main_heading', 'heading',
              'topic', 'paper_idx', 'BIO', 'BIO_1', 'BIO_2', 'offset1', 'pro1', 'offset2', 'pro2', 'offset3', 'pro3', 'mask', 'bi_labels', 'labels']

df.to_csv('all_sent.csv', index=False)
print('\n\"all_sent.csv\" has been saved in this folder')
