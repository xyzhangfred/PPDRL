import os
import numpy as np
import random
import string

def load_yelp(file_dir = '/home/xiongyi/dataxyz/data/yelp_style_transfer', file_name = 'yelp.{}'.format('train'), line_num = 50000):

    with open(os.path.join(file_dir, file_name), 'r') as f:
        lines = f.readlines()
        #randomly pick 
        inds = np.random.choice(range(len(lines)), line_num)
        lines = [lines[ind] for ind in inds]

        # lines = lines[:10]
        labels = [int(line.split('\t')[0]) for line in lines]
        sents = [line.split('\t')[1].strip() for line in lines]
    return labels, sents



def load_imdb(file_dir, file_name, line_num = 50000, shuffle = True):

    with open(os.path.join(file_dir, file_name), 'r') as f:
        lines = f.readlines()
        #randomly pick 
        if line_num <= len(lines):
            inds = np.random.choice(range(len(lines)), line_num)
            lines = [lines[ind] for ind in inds]
        else:
            print ('not enough lines, using all the data!')
        # lines = lines[:10]
    sents = [line.split('\t')[0].strip() for line in lines]
    sentiments = [int(line.split('\t')[1]) for line in lines]
    genres = [int(line.split('\t')[2].strip()) for line in lines]
    if shuffle:
            c = list(zip(sents, sentiments,genres))
            random.shuffle(c)
            sents, sentiments,genres = zip(*c)
    return  sents, sentiments, genres

def load_MDs(file_dir, file_name, line_num = 50000, labels = ['gender']):
    ##load the physician dataset with multiple labels

    with open(os.path.join(file_dir, file_name), 'r') as f:
        lines = f.readlines()
        #randomly pick 
        inds = np.random.choice(range(len(lines)), line_num)
        lines = [lines[ind] for ind in inds]

        # lines = lines[:10]
        labels = [line.split('\t')[0] for line in lines]
        sents = [line.split('\t')[1].strip() for line in lines]
    return labels, sents


def load_moji(file_dir='/home/xiongyi/dataxyz/repos/NotMine/demog-text-removal/data/processed/sentiment_race',shuffle = True, save_tsv = False):
    ##load the physician dataset with multiple labels
    vocabs = None
    with open(os.path.join(file_dir, 'vocab')) as f:
        vocabs = f.readlines()
    vocabs = [v.strip() for v in vocabs]
    sents = []
    races = []
    sentiments = []
    for sentiment_id, sentiment in enumerate(['neg', 'pos']):
        for race_id, race in enumerate(['neg', 'pos']):
            file_name = '{}_{}'.format(sentiment, race)
            with open(os.path.join(file_dir, file_name)) as f:
                lines = f.readlines()
                for line in lines:
                    sent = [vocabs[int(i)] for i in line.strip().split()]
                    sents.append(sent)
                    races.append(race_id)
                    sentiments.append(sentiment_id)
    #weird decoding bug
    new_sents = []
    for sent in sents:
        if sent[0][:2] =="b'":
            sent[0] = sent[0][2:]
        #filter out @ and links, etc
        new_sent =[]
        for vocab in sent:
            if vocab == '':
                continue
            if vocab[0] in ['@', '_']:
                continue
            elif vocab[-3:] in ['com', 'org']:
                continue
            elif vocab.startswith('pic') or vocab.startswith('http') or vocab.startswith('www'):
                continue
            else:
                new_sent.append(vocab)
        new_sents.append(' '.join(new_sent))
    if shuffle:
        c = list(zip(new_sents, races,sentiments))
        random.shuffle(c)
        new_sents, races,sentiments = zip(*c)
    if save_tsv:
        #save a tsv with only the sents and the prot_attr
        with open( os.path.join(file_dir, 'deepmoji.tsv'), 'w') as f:
            for sent, race in zip(new_sents, races):
                f.write(str(race)+'\t'+sent+'\n')


    return races,sentiments, new_sents

def split_moji(file_dir='/home/xiongyi/dataxyz/repos/NotMine/demog-text-removal/data/processed/sentiment_race',\
     out_dir = '/home/xiongyi/dataxyz/data/demog-text-removal/sentiment_race'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ##load the physician dataset with multiple labels
    vocabs = None
    with open(os.path.join(file_dir, 'vocab')) as f:
        vocabs = f.readlines()
    vocabs = [v.strip() for v in vocabs]
    sents = []
    races = []
    sentiments = []
    for sentiment_id, sentiment in enumerate(['neg', 'pos']):
        for race_id, race in enumerate(['neg', 'pos']):
            file_name = '{}_{}'.format(sentiment, race)
            with open(os.path.join(file_dir, file_name)) as f:
                lines = f.readlines()
                for line in lines:
                    sent = [vocabs[int(i)] for i in line.strip().split()]
                    sents.append(sent)
                    races.append(race_id)
                    sentiments.append(sentiment_id)
    #weird decoding bug
    new_sents = []
    for sent in sents:
        if sent[0][:2] =="b'":
            sent[0] = sent[0][2:]
        #filter out @ and links, etc
        new_sent =[]
        for vocab in sent:
            if vocab == '':
                continue
            if vocab[0] in ['@', '_']:
                continue
            elif vocab[-3:] in ['com', 'org']:
                continue
            elif vocab.startswith('pic') or vocab.startswith('http') or vocab.startswith('www'):
                continue
            else:
                new_sent.append(vocab)
        new_sents.append(' '.join(new_sent))
    #shuffle the data:
    c = list(zip(new_sents, races,sentiments))
    random.shuffle(c)
    new_sents, races,sentiments = zip(*c)
    #split into train, eval ,test
    train_idx = np.random.choice(range(len(new_sents)), int(0.8 * len(new_sents)), replace=False)
    not_train = [i for i in range(len(new_sents)) if i not in train_idx]
    eval_idx = np.random.choice( not_train, int(0.1 * len(new_sents)), replace=False)
    test_idx = [i for i in not_train if i not in eval_idx]

    #save into 3 separate tsv files.
    
    
    train_sents = [new_sents[i] for i in train_idx]
    train_races = [races[i] for i in train_idx]
    train_sentiments = [sentiments[i] for i in train_idx]
    with open(os.path.join(out_dir, 'train.tsv'), 'w') as f:
        for sent, race, sentiment in zip(train_sents, train_races, train_sentiments):
            f.write(sent +' \t' + str(race) + '\t' + str(sentiment) + '\n')


    eval_sents = [new_sents[i] for i in eval_idx]
    eval_races = [races[i] for i in eval_idx]
    eval_sentiments = [sentiments[i] for i in eval_idx]
    with open(os.path.join(out_dir, 'eval.tsv'), 'w') as f:
        for sent, race, sentiment in zip(eval_sents, eval_races, eval_sentiments):
            f.write(sent +' \t' + str(race) + '\t' + str(sentiment) + '\n')

    test_sents = [new_sents[i] for i in test_idx]
    test_races = [races[i] for i in test_idx]
    test_sentiments = [sentiments[i] for i in test_idx]
    with open(os.path.join(out_dir, 'test.tsv'), 'w') as f:
        for sent, race, sentiment in zip(test_sents, test_races, test_sentiments):
            f.write(sent +' \t' + str(race) + '\t' + str(sentiment) + '\n')
    print ("train: {}, eval: {}, test: {}.".format(len(train_sents), len(eval_sents), len(test_sents)) )
    # return races,sentiments, new_sents


def load_moji_split(file_dir = '/home/xiongyi/dataxyz/data/demog-text-removal/sentiment_race', prefix = 'train', shuffle = True,\
    line_num = None):

    full_dir = os.path.join(file_dir, '{}.tsv'.format(prefix))

    with open(full_dir, 'r') as f:
        lines = f.readlines()
    if shuffle:
        random.shuffle(lines)
    
    if line_num is None or line_num < len(lines):
        print ('Using all data')
    else:
        lines = lines[:line_num]
    races,sentiments, sents =[],[],[]
    err_num = 0
    for line in lines:
        try:
            sent,race,sentiment = line.strip().split('\t')
        except:
            err_num += 1
            continue
        races.append(int(race))
        sentiments.append(int(sentiment))
        sents.append(sent)
    print ('skipped {} lines due to error'.format(err_num))
    return races,sentiments, sents



def split_MD(file_dir='/home/xiongyi/dataxyz/repos/NotMine/RateMDs/all_GitHub.csv',\
     out_dir = '/home/xiongyi/dataxyz/data/RateMDs'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    ##load the physician dataset with multiple labels
    raw_scores = []
    genders = []
    specialties = []
    reviews = []
    with open(file_dir) as f:
        lines = f.readlines()

    for line in lines[1:]:
        texts = line.strip().split(',')
        idx, dor_id, name, specialty, gender, S, P, H, K = texts[:9]
        review = ','.join(texts[9:])
        review = review.replace("b'", '')
        review = review.replace("'", '')
        try:
            avg_score = 1/4  * (float(S)+float(P)+float(H)+float(K))
        except:
            print ('formatting error')
            continue
        genders.append(gender)
        raw_scores.append(avg_score)
        specialties.append(specialty)
        reviews.append(review)

    ####Rescale Scores: >= 4.0 ---> 1, < 4.0 --->0
    scores = [1 if s >= 4.0 else 0  for s in raw_scores]
    ####gender: Male:0, Female:1
    genders = [0 if g == 'm' else 1  for g in genders]


    ###for EDA    
    #     reviews.append(review)
    # from collections import Counter
    
    # gc = Counter(genders)
    # spc = Counter(specialties)
    # scc = Counter(scores)
    
    # new_scores = [s > 3.0 for s in scores] 
    # gsc_c = Counter(list(zip(genders,new_scores)))
    #shuffle the data:
    c = list(zip(reviews, genders,specialties,scores))
    random.shuffle(c)
    reviews, genders,specialties,scores = zip(*c)
    #split into train, eval ,test
    train_idx = np.random.choice(range(len(reviews)), int(0.8 * len(reviews)), replace=False)
    not_train = [i for i in range(len(reviews)) if i not in train_idx]
    eval_idx = np.random.choice( not_train, int(0.1 * len(reviews)), replace=False)
    test_idx = [i for i in not_train if i not in eval_idx]

    #save into 3 separate tsv files.
    train_reviews = [reviews[i] for i in train_idx]
    train_genders = [genders[i] for i in train_idx]
    train_specialties = [specialties[i] for i in train_idx]
    train_scores = [scores[i] for i in train_idx]
    with open(os.path.join(out_dir, 'train.tsv'), 'w') as f:
        for review, gender, specialty,score in zip(train_reviews, train_genders, train_specialties,train_scores):
            f.write(review +' \t' + str(gender) + '\t' + str(specialty) + '\t'+ str(score)+ '\n')
    
    eval_reviews = [reviews[i] for i in eval_idx]
    eval_genders = [genders[i] for i in eval_idx]
    eval_specialties = [specialties[i] for i in eval_idx]
    eval_scores = [scores[i] for i in eval_idx]
    with open(os.path.join(out_dir, 'eval.tsv'), 'w') as f:
        for review, gender, specialty,score in zip(eval_reviews, eval_genders, eval_specialties,eval_scores):
            f.write(review +' \t' + str(gender) + '\t' + str(specialty) + '\t'+ str(score)+ '\n')


    test_reviews = [reviews[i] for i in test_idx]
    test_genders = [genders[i] for i in test_idx]
    test_specialties = [specialties[i] for i in test_idx]
    test_scores = [scores[i] for i in test_idx]
    with open(os.path.join(out_dir, 'test.tsv'), 'w') as f:
        for review, gender, specialty,score in zip(test_reviews, test_genders, test_specialties,test_scores):
            f.write(review +' \t' + str(gender) + '\t' + str(specialty) + '\t'+ str(score)+ '\n')
    print ("train: {}, eval: {}, test: {}.".format(len(train_reviews), len(eval_reviews), len(test_reviews)) )
    # return races,sentiments, new_sents


def load_MD(file_dir = '/home/xiongyi/dataxyz/data/demog-text-removal/sentiment_race', prefix = 'train', shuffle = True,\
    line_num = None):

    full_dir = os.path.join(file_dir, '{}.tsv'.format(prefix))

    with open(full_dir, 'r') as f:
        lines = f.readlines()
    if shuffle:
        random.shuffle(lines)
    
    if line_num is None or line_num < len(lines):
        print ('Using all data')
    else:
        lines = lines[:line_num]
    genders,scores, reviews =[],[],[]
    err_num = 0
    for line in lines:
        try:
            review,gender,specialty,score = line.strip().split('\t')
        except:
            err_num += 1
            continue
        genders.append(int(gender))
        scores.append(int(score))
        reviews.append(review)
    print ('skipped {} lines due to error'.format(err_num))
    return genders,scores, reviews

if __name__ == "__main__":
    split_MD()