import json
import glob
import logging
import os
import re

import random
import matplotlib.pyplot as plt

import numpy as np
import torch

from transformers import BertTokenizer


def get_movie_genre(genre_dir = '/home/xiongyi/dataxyz/data/imdb/title.basics.tsv'):
    tit_2_gen = {}
    with open(genre_dir) as f:
        lines = f.readlines()
    for line in lines[1:]:
        title = line.split('\t')[0]
        genre = line.split('\t') [-1]
        tit_2_gen[title] = genre.strip().split(',')
    return tit_2_gen


def combine_movie_genre(tit_2_gen, movie_dir = '/home/xiongyi/dataxyz/data/imdb/train', visualize = False):
    all_data ={}
    all_genre_counts = {}
    for _, label in enumerate(['neg', 'pos']):
        texts = {}
        text_dir = os.path.join(movie_dir, label)
        text_fns = os.listdir(text_dir)
        for i,fn in enumerate(text_fns):
            if i % 100 == 0:
                print (i, 'in ', len(text_fns))
            # print (fn)
            text_id = int(fn.split('_')[0])
            raw_text= open(os.path.join(text_dir, fn)).readline().strip()
            texts[text_id] = remove_tags(raw_text)

        with open(os.path.join(movie_dir, 'urls_'+label+'.txt')) as f:
            urls = f.readlines()
        urls = [u.strip() for u in urls]
        titles = [ u.split('title/')[1].split('/user')[0] for u in urls]
        genres = [tit_2_gen[t]  if t in tit_2_gen else None  for t in titles]
        all_data[label] = [ (texts[i], genres[i])  for i in range(len(genres))]
        all_genre_counts[label] = {}
        all_genre_counts[label]['None'] = 0
        for g in genres:
            if g is None:
                all_genre_counts[label]['None'] += 1
            else:
                for gg in g:
                    if gg in all_genre_counts[label]:
                        all_genre_counts[label][gg] += 1
                    else:
                        all_genre_counts[label][gg] = 0
    '''
    for visualizing genre distribution
    '''
    if visualize:
        all_genre_names = [g for g in list(all_genre_counts['pos'].keys()) if g in list(all_genre_counts['neg'].keys())]
        plt.figure(figsize =(30,15))
        neg_scores = [all_genre_counts['neg'][g] for g in all_genre_names]
        pos_scores = [all_genre_counts['pos'][g] for g in all_genre_names]
        ind = np.arange(len(all_genre_names))    # the x locations for the groups

        p1 = plt.bar(ind, neg_scores)
        p2 = plt.bar(ind, pos_scores,bottom=neg_scores, tick_label= [pos_scores[i]/neg_scores[i] for i in range(len(neg_scores)) ])
        for i in range(len(all_genre_names)):
            pos_rate = "{0:.2f}".format(pos_scores[i]/(pos_scores[i]+neg_scores[i]))
            plt.text(i - 0.25, neg_scores[i] + pos_scores[i], pos_rate, color='k', fontweight='bold')
        plt.ylabel('Num')
        plt.title('Sentiment ratio by genre')
        plt.xticks(ind, all_genre_names)
        plt.legend((p1[0], p2[0]), ('neg', 'pos'))

        plt.savefig('Sentiment_ratio_by_genre')
        plt.show()
    return all_data, all_genre_counts



def prepare_imdb_subset(all_data, genre_0 = 'Drama', genre_1 = 'Horror', genre_0_pos_ratio = 0.8, sample_size = 2000 ):
    samples = {}
   
    for label_id, label in enumerate(['neg', 'pos']):
        dat = all_data[label]
        dat_0,dat_1 =[], []
        for d in dat:
            if d[-1] is None:
                continue
            if genre_0 in d[-1] and genre_1 not in d[-1]:
                new_d = (d[0], 0)
                dat_0.append(new_d)
            if genre_1 in d[-1] and genre_0 not in d[-1]:
                new_d = (d[0], 1)
                dat_1.append(new_d)
        samples[label] = []
        print (len(dat_0), ' ', len(dat_1))
        if genre_0_pos_ratio > 0: 
            if label == 'pos':
                dat_0_sample_index = np.random.choice(range(len(dat_0)), size = int(genre_0_pos_ratio * sample_size / 2), replace=True)
                dat_1_sample_index = np.random.choice(range(len(dat_1)), size = int((1-genre_0_pos_ratio) * sample_size / 2), replace=True)
            else:
                dat_0_sample_index = np.random.choice(range(len(dat_0)), size = int((1-genre_0_pos_ratio) * sample_size / 2), replace=True)
                dat_1_sample_index = np.random.choice(range(len(dat_1)), size = int(genre_0_pos_ratio * sample_size / 2), replace=True)
            dat_0_samples = [dat_0[i] for i in dat_0_sample_index]
            dat_1_samples = [dat_1[i] for i in dat_1_sample_index]
            samples[label] = dat_0_samples + dat_1_samples
            samples[label] = [s + (label_id,) for s in samples[label]]
        else:
            #if genre pos ratio == -1, don't do any re-sampling
            dat = dat_0 + dat_1
            sample_index = np.random.choice(range(len(dat)) , int(sample_size/2), replace = True)
            samples[label] = [dat[i] for i in sample_index]
            samples[label] = [s + (label_id,) for s in samples[label]]

    all_samples = samples['pos'] + samples['neg']
    random.shuffle(all_samples)
    return all_samples

def save_tsv(filename, samples):
    #save a tsv containing text, movie_genre, sentiment
    with open(filename, 'w') as f:
        for sample in samples:
            f.write(sample[0] + '\t' + str(sample[1]) + '\t' +str(sample[2]) +'\n')
    return len(samples)


def remove_tags(text):
    TAG_RE = re.compile(r'<[^>]+>')
    TAG_S = re.compile(r'\s+')
    text = TAG_RE.sub(' ', text)
    text = TAG_S.sub(' ', text)

    return text

if __name__ == "__main__":
    print ('Start')
    GENRE_0 = 'Drama'
    GENRE_1 = 'Horror'
    tit_2_gen = get_movie_genre(genre_dir = '/home/xiongyi/dataxyz/data/imdb/title.basics.tsv')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
    for prefix in ['train','test']:
        movie_dir = os.path.join('/home/xiongyi/dataxyz/data/imdb', prefix)
        all_data, all_genre_counts = combine_movie_genre(tit_2_gen, movie_dir=movie_dir)
        
        for genre_0_pos_ratio in [-1]:
            print ('*#$' * 10)
            print (genre_0_pos_ratio )
            if prefix == 'train':
                sample_size = len(all_data['pos']) +len(all_data['neg'])
            elif prefix == 'test':
                sample_size = len(all_data['pos']) +len(all_data['neg'])
            samples = prepare_imdb_subset(all_data, genre_0=GENRE_0, genre_1 = GENRE_1, genre_0_pos_ratio = genre_0_pos_ratio, sample_size = sample_size)
            save_tsv(filename='full_{}_{}_pos_ratio_{}.tsv'.format(prefix,GENRE_0, genre_0_pos_ratio), samples=samples)

