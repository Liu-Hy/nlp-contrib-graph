'''
Generate a csv file of sentences with their info units, predicates, terms, and different types of triples
'''
import os
import re
import math
import pandas as pd
import json
from parse import *

base_dir = 'training_data'
sep = os.path.sep
reports = []
'''
If none of the phrases in a triple can be found in any sentence in the corresponding info unit,
and the triple does not take the form of "Contribution||has||(unit name)", it will be listed here.
'''
special_trip = []

def get_dir(topic_ls=None, paper_ls=None):
    # Get the list of paper directories
    dir_ls = []
    if topic_ls is None:
        topic_ls = os.listdir(base_dir)
        topic_ls.remove('train-README.md')
        topic_ls.remove('trial-README.md')
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
    rx = '(.*Stanza-out.txt$)|(^sentences.txt$)'
    file_path = []
    for dir in dirs:
        new = ['', '']  # stores the paths of the sentence file and the label file
        for file in os.listdir(dir):
            res = re.match(rx, file)
            if res:
                if res.group(1):
                    new[0] = os.path.join(dir, file)
                if res.group(2):
                    new[1] = os.path.join(dir, file)
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
            lex_cue = 'background|related work|conclusion'
        else:
            lex_cue = 'title|abstract|introduction|background|related work|conclusion|model|models|method|methods|approach|architecture|system|application|experiment|experiments|experimental setup|implementation|hyperparameters|training|result|results|ablation|baseline|evaluation'
        exp = re.compile(lex_cue)
        # Decide if it is a main heading
        return True if exp.search(line.lower()) else False
    else:
        return False

# Determin if a sentence conforms to a specific case method.
# Their are three case methods in all, eg: Attention Is All You Need; ATTENTION IS ALL YOU NEED; Attention is all you need.

def case_check(line, flag):
    if flag == 1:
        match = re.search(r'[a-z]', line)
        if match:
            return False
        return True
    else:
        countW = 0
        words = line.split(' ')
        if flag == 0:
            prep = ['a', 'an', 'and', 'the', 'or', 'but', 'nor', 'if',
                    'for', 'in', 'on', 'with', 'by', 'as', 'to', 'of', 'via']
            if not words[0].istitle():
                countW += 1
            if len(words) > 1:
                if not words[-1].istitle():
                    countW += 1
                for word in words[1:-1]:
                    if not word.istitle() and word not in prep:
                        countW += 1
            return countW <= math.ceil(len(words)/5)
        if flag == 2:
            if not words[0].istitle():
                countW += 1
            for word in words[1:]:
                if re.match(r'[A-Z]', word):
                    countW += 1
            return countW <= math.ceil(len(words)/3)

# read the relevant files from the folder of one paper, and produce a data table for that paper.
def load_paper_sentence(sent_path, label_path):
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
                        if case_check(line, m):
                            count[m] += 1
            else:
                break

    with open(sent_path, 'r') as f:
        i = 0
        flg = count.index(max(count))
        # two string buffers, storing the heading and the main heading respectively
        heading, main_h = '', ''
        while(True):
            i += 1
            line = f.readline().rstrip("\n")
            if line:
                if is_heading(line) and case_check(line, flg):
                    heading = line    # update the heading buffer

                    if is_main_heading(line):
                        main_h = line    # update the main heading buffer too
                        # The line itself is a main heading, no heading needs to be stored.
                        sent.append(
                            [i, line, '', '', task, index, None, None, None, [], [], [], [], [], [], [], [], [], [], [], 1, 0, None])
                    else:
                        # for plain headings, store the main heading it belongs to.
                        # judge if it should be masked
                        if is_main_heading(main_h, judge_mask=True):
                            sent.append([i, line, main_h, '', task,
                                         index, None, None, None, [], [], [], [], [], [], [], [], [], [], [], 0, 0, None])
                        else:
                            sent.append([i, line, main_h, '', task,
                                         index, None, None, None, [], [], [], [], [], [], [], [], [], [], [], 1, 0, None])
                else:
                    # For plain text line, store both the heading and the main heading.
                    if is_main_heading(main_h, judge_mask=True):
                        sent.append([i, line, main_h, heading,
                                     task, index, None, None, None, [], [], [], [], [], [], [], [], [], [], [], 0, 0, None])
                    else:
                        sent.append([i, line, main_h, heading,
                                     task, index, None, None, None, [], [], [], [], [], [], [], [], [], [], [], 1, 0, None])
            else:
                break

    # slice the sentence with the span of characters, returns the span of words
    def get_word_idx(sent, start, end):
        ls = sent.split(' ')
        if isinstance(start, str):
            start = int(start)
        if isinstance(end, str):
            end = int(end)
        # if the span of characters doesn't conform to word boundaries, 'st' and 'en' will remain 0.
        st, en = 0, 0
        length = [len(word) for word in ls]
        count = 0
        for i in range(len(ls)):
            if start == count:
                st = i
                break
            count += (length[i]+1)
        for j in range(st, len(ls)):
            count += (length[j]+1)
            if end == (count-1):
                en = j + 1
                break
        return st, en

    # Mark the label of contribution-ralated sentences, and initialize their BIO tag sequences.
    with open(label_path, 'r') as f:
        while(True):
            line = f.readline().rstrip("\n")
            if line:
                sent[int(line)-1][-2] = 1
                sent[int(line)-1][6] = ['O'] * \
                    len(sent[int(line)-1][1].split(' '))
            else:
                break

    # go over the entities and change the corresponding part of BIO sequences
    ent_path = sep.join(label_path.split(sep)[:-1]+['entities.txt'])
    with open(ent_path, 'r') as f:
        while(True):
            line = f.readline().rstrip("\n")
            if line:
                info = line.split('\t')
                sentence = sent[int(info[0])-1][1]
                # if sentence.split(' ')[0].lower()[1:] == sentence.split(' ')[0][1:]:
                # sentence = sentence[0].lower() + sentence[1:]
                st, en = get_word_idx(sentence, info[1], info[2])
                phrase = info[3].strip()
                # If the span of characters does not match the given phrase, use the given phrase instead
                if ' '.join(sentence.split(' ')[st: en]).strip() != phrase:
                    st_char = (' ' + sentence).find(' ' + phrase + ' ')
                    st, en = get_word_idx(
                        sentence, st_char, st_char + len(phrase))
                    if st == 0 and en == 0:
                        #print(f'Could not find the phrase \'{info[3]}\' in the {int(info[0])}th sentence of \'{task}\' paper {index}')
                        continue
                    # else:
                        #print(f'In the {int(info[0])}th sentence of \'{task}\' paper {index}, the entity \'{info[3]}\' is not in the span ({info[1]}, {info[2]})')
                for j in range(st, en):
                    if sent[int(info[0])-1][6] is None:
                        #print(f'A phrase exists in the {int(info[0])}th sentence of \'{task}\' paper {index}, which is not labeled as a contribution sentence.')
                        sent[int(info[0])-1][6] = ['O'] * \
                            len(sent[int(info[0])-1][1].split(' '))
                    if j == st:
                        sent[int(info[0])-1][6][j] = 'B'
                    else:
                        sent[int(info[0])-1][6][j] = 'I'
            else:
                break

    # decide which information unit each positive sentence belongs to.
    j_dir = sep.join(sent_path.split(sep)[:-1]) + sep + 'info-units'
    # For each json file representing an information unit
    for unit in os.listdir(j_dir):
        js_file = os.path.join(j_dir, unit)
        try:
            with open(js_file, 'r') as f:
                data = json.load(f, strict=False)
            lst = find_source(data, [])
            if "TITLE" in lst:  # When the title is a source sentence, sometimes it is abbreviated as 'TITLE'
                sent[1][-1] = unit[:-5]
            for j in range(len(sent)):
                if sent[j][1] in lst:
                    sent[j][-1] = unit[:-5]
        except json.JSONDecodeError as e:
            js_position = sep.join(js_file.split(sep)[-4:])
            #print(f'JSONDecodeError in {js_position}\n', e)
            continue

    # given a sequence of BIO taggs, get the list of tuples representing spans of entities
    def get_entity_spans(ls):
        spans = []
        for i in range(len(ls)):
            st, ed = 0, 0
            if ls[i] == 'B':
                st, ed = i, i + 1
                for j in range(i+1, len(ls)):
                    if ls[j] == 'I':
                        ed += 1
                    else:
                        break
                spans.append((st, ed))
        return spans

    for i in range(len(sent)):
        if sent[i][6] is not None:
            sent[i][8] = sent[i][7] = sent[i][6]

    # try to find the SPO(Subject, Predicate, Object) type of each phrase
    aux = []
    for i in range(len(sent)):
        if sent[i][6] is not None:
            tup_ls = get_entity_spans(sent[i][6])
            # use three booleans to indicate if the phrase has ever been a subject, predicate or object
            tuple_ls = [[0, 0, 0, tup] for tup in tup_ls]
            word_ls = sent[i][1].split(' ')
            phrase_ls = [' '.join(word_ls[st:en]) for st, en in tup_ls]
            # store the sentence idx, tuple_ls, and phrase_ls.
            aux.append([i, tuple_ls, phrase_ls, []])
    t_dir = sep.join(sent_path.split(sep)[:-1]) + sep + 'triples'
    paper_triple_stat = [0] * 5
    paper_normal_root = 0
    for unit in os.listdir(t_dir):
        t_file = os.path.join(t_dir, unit)
        js_file = os.path.join(j_dir, unit.replace('.txt', '.json'))
        try:
            with open(js_file, 'r') as g:
                js = json.load(g, strict=False)
                js = {'Contribution': js}
        except json.JSONDecodeError as e:
            js_position = sep.join(js_file.split(sep)[-4:])
            #print(f'JSONDecodeError in {js_position}\n', e)
            continue
        except FileNotFoundError as fe:
            #print(fe)
            continue
        with open(t_file, 'r') as f:
            while(True):
                line = f.readline().rstrip("\n")
                if line:
                    # empty the temporary buffer
                    for a in range(len(aux)):
                        aux[a][3] = []
                    if line[0] == '(':
                        line = line[1:]
                    if line[-1] == ')':
                        line = line[:-1]
                    triple = line.split('||')
                    evidence = find_tri_sent(
                        js, triple, [], [], [])  # unit[:-4]
                    if not evidence:
                        js_position = sep.join(js_file.split(sep)[-4:])
                        paper_triple_stat[0] += 1
                        #print(f'the triple \'{triple}\' not found in {js_position}')
                    else:
                        cands = evidence[0].split('\n')
                        for i in range(len(cands)):
                            for j in range(len(aux)):
                                if cands[i].strip() == sent[aux[j][0]][1]:
                                    for w in range(3):
                                        for k in range(len(aux[j][2])):
                                            if aux[j][2][k] == triple[w]:
                                                aux[j][3].append((w, k))
                                                break
                                    break
                        lens = [len(aux[j][3]) for j in range(len(aux))]
                        new_item = []
                        #try:
                        paper_triple_stat[max(lens)] += 1
                        #except IndexError:
                        #print(f'List index out of range. The actual number of max is {max(lens)} for triple \'{triple}\' in\n', t_file)
                        if max(lens) == 0:
                            if triple[0] == 'Contribution' and triple[1] == 'has':
                                paper_normal_root += 1
                            else:
                                special_trip.append(triple)
                        else:
                            idx = lens.index(max(lens))
                            found = [0, 0, 0]
                            if max(lens) != 3 and len(cands) > 1:
                                new_item.append(triple)
                                new_item.append(
                                    [aux[idx][0], idx, sent[aux[idx][0]][1]])
                            for t in range(len(aux[idx][3])):
                                w, k = aux[idx][3][t]
                                aux[idx][1][k][w] = 1
                                found[w] = 1
                            if max(lens) == 3:
                                sent[aux[idx][0]][9].append(triple)
                            if max(lens) == 2:
                                if found == [1, 0, 1]:
                                    sent[aux[idx][0]][10].append(triple)
                                elif found == [0, 1, 1]:
                                    uni_name = unit[:-4]
                                    if triple[0] == (uni_name[0].upper()+uni_name[1:]).replace('-', ' '):
                                        sent[aux[idx][0]][11].append(triple)
                                    else:  # SPEC A
                                        sent[aux[idx][0]][14].append(triple)
                                else:  # SPEC B
                                    sent[aux[idx][0]][15].append(triple)
                            if max(lens) == 1:
                                if found == [0, 0, 1]:
                                    uni_name = unit[:-4]
                                    if triple[0] == (uni_name[0].upper()+uni_name[1:]).replace('-', ' '):
                                        sent[aux[idx][0]][12].append(triple)
                                    elif triple[0] == 'Contribution':
                                        sent[aux[idx][0]][13].append(triple)
                                    else:  # SPEC C
                                        sent[aux[idx][0]][16].append(triple)
                                else:  # SPEC D
                                    sent[aux[idx][0]][17].append(triple)

                            for i in range(3):
                                if found[i] == 0:
                                    for j in range(len(aux)):
                                        for w, k in aux[j][3]:
                                            if w == i and triple[w] == aux[j][2][k]:
                                                aux[j][1][k][w] = 1
                                                if max(lens) != 3 and len(cands) > 1:
                                                    new_item.append(
                                                        [aux[j][0], j, sent[aux[j][0]][1]])
                                                break
                                        else:
                                            continue
                                        break
                            if new_item and len(new_item) > 2:
                                cross_sents = new_item[1:]
                                cross_sents = sorted(
                                    cross_sents, key=lambda x: x[0])
                                new_item = new_item[0:1]
                                new_item += cross_sents
                                new_item.append('- '*80)
                                paper_triple_stat[4] += 1
                                global reports
                                reports += new_item
                else:
                    break

    # An S-P-O type corresponds to a combination of boolean indicators
    # The 4 keys stand for 'predicate', 'subject', 'object', 'both subject and object' respectively.
    good_state = {'p': [0, 1, 0], 's': [1, 0, 0],
                  'ob': [0, 0, 1], 'b': [1, 0, 1]}
    for i in range(len(aux)):
        for item in aux[i][1]:
            if item[:3] not in good_state.values():
                # if the label of any phrase in the sentence cannot be decided,
                # delete the tag sequence to filter out this sentence
                sent[aux[i][0]][7] = sent[aux[i][0]][8] = None
                break

    for i in range(len(aux)):
        '''
        interprete the boolean states to phrase types according to the BIO_type setting,
        and change the corresponding parts in BIO sequences
        BIO_type=1: decide whether it is a predicate
        BIO_type=2: decide which of the four keys in 'good_state' it belongs to
        '''
        if sent[aux[i][0]][7] is not None:
            sent[aux[i][0]][7] = ['O']*len(sent[aux[i][0]][7])
            sent[aux[i][0]][8] = ['O']*len(sent[aux[i][0]][8])
            for k in range(len(aux[i][1])):  # for item in aux[i][1]
                st, en = aux[i][1][k][3]
                if aux[i][1][k][:3] == good_state['p']:
                    sent[aux[i][0]][18].append((aux[i][2][k], aux[i][1][k][3]))
                    sent[aux[i][0]][7][st] = 'B-p'
                    for j in range(st+1, en):
                        sent[aux[i][0]][7][j] = 'I-p'
                else:
                    sent[aux[i][0]][19].append((aux[i][2][k], aux[i][1][k][3]))
                    sent[aux[i][0]][7][st] = 'B-n'
                    for j in range(st+1, en):
                        sent[aux[i][0]][7][j] = 'I-n'
                for key, value in good_state.items():
                    if aux[i][1][k][:3] == value:
                        sent[aux[i][0]][8][st] = 'B-'+key
                        for j in range(st+1, en):
                            sent[aux[i][0]][8][j] = 'I-'+key
    # print(f'paper triple stat: {paper_triple_stat}')
    return sent, paper_triple_stat, paper_normal_root

def load_data_sentence(file_path):
    # Get the data table of all the papers in file_path
    triple_stat = [0] * 5
    data = []
    normal_root = 0
    for tuple in file_path:
        sentence_path, label_path = tuple
        paper_data, paper_triple_stat, paper_normal_root = load_paper_sentence(
            sentence_path, label_path)
        for i in range(5):
            triple_stat[i] += paper_triple_stat[i]
        data += paper_data
        normal_root += paper_normal_root
    count_1 = sum(triple_stat[:-1])
    count_2 = sum(triple_stat[:-2])
    count_3 = triple_stat[-1]
    print('\nTotal triple statistics: ', '- '*30)
    print(f'There are {count_1} triples in all. Among them,')
    print(
        f'{count_2} ({round(count_2 * 100 / count_1, 2)}%) cannot get all phrases from a single sentence')
    print(
        f'{count_3} ({round(count_3 * 100 / count_1, 2)}%) are cross-sentence triples')
    print(f'{normal_root} ({round(normal_root * 100 / count_1, 2)}%) takes the form \'Contribution||has||...\'')
    #print(f'0: {triple_stat[0]} ({round(triple_stat[0] * 100 / count_1, 2)}%)')
    #print(f'1: {triple_stat[1]} ({round(triple_stat[1] * 100 / count_1, 2)}%)')
    #print(f'2: {triple_stat[2]} ({round(triple_stat[2] * 100 / count_1, 2)}%)')
    #print(f'3: {triple_stat[3]} ({round(triple_stat[3] * 100 / count_1, 2)}%)')
    return data

dirs = get_dir()
file_path = get_file_path(dirs)
data = load_data_sentence(file_path)

df = pd.DataFrame(data)
df.columns = ['idx', 'text', 'main_heading', 'heading',
              'topic', 'paper_idx', 'BIO', 'BIO_1', 'BIO_2', 'triple_A', 'triple_B', 'triple_C', 'triple_D', 'SPEC_0', 'SPEC_1', 'SPEC_2', 'SPEC_3', 'SPEC_4', 'predicates', 'subj/obj', 'mask', 'bi_labels', 'labels']
pos = df[df['bi_labels'] == 1]
pos = pos[['labels', 'text', 'predicates', 'subj/obj', 'triple_A',
           'triple_B', 'triple_C', 'triple_D', 'SPEC_0', 'SPEC_1', 'SPEC_2', 'SPEC_3', 'SPEC_4', 'topic', 'paper_idx', 'idx']].rename(columns={'labels': 'info_unit'})
pos.to_csv('./interim/triples.csv', index=False)
print('\n\"triples.csv\" has been saved to ./interim')
