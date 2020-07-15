import os
import numpy as np
from data_util import load_moji
from sklearn.model_selection import train_test_split
def split_moji_by_race(file_dir = '/home/xiongyi/dataxyz/data/demog-text-removal/sentiment_race',\
    r0s0_rates = [0.5,0.4,0.3,0.2,0.1], total_num = 10000):
    races,sentiments, sents = load_moji()
    race_sentiment_groups = {}
    for race in [0, 1]:
        for sentiment in [0, 1]:
            race_sentiment_id = [i for i in range(len(races)) if races[i] == race and sentiments[i] == sentiment]
            race_sentiment_groups['{}_{}'.format(race, sentiment)] = race_sentiment_id
    ###‘1_0’ is the smallest group, use that to calculate sizes of other groups.
    for r0s0_rate in r0s0_rates:
        out_dir = os.path.join(file_dir, 'r0s0_{}'.format(r0s0_rate))
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        subgroup_sizes = {}
        sub_groups = {}
        subgroup_sizes['0_0'] = int (total_num * r0s0_rate * 0.5)
        subgroup_sizes['0_1'] = int (total_num * (1-r0s0_rate) * 0.5)
        subgroup_sizes['1_0'] = subgroup_sizes['0_1']
        subgroup_sizes['1_1'] = subgroup_sizes['0_0']
        all_train_idx = []
        all_eval_idx = []
        for race in [0, 1]:
            for sentiment in [0, 1]:
                group_name = '{}_{}'.format(race, sentiment)
                if subgroup_sizes[group_name] > len(race_sentiment_groups[group_name]):
                    replace = True
                else:
                    replace = False
                sub_groups[group_name] = np.random.choice(race_sentiment_groups[group_name],size=subgroup_sizes[group_name], replace=replace)
                train_idx, eval_idx = train_test_split(sub_groups[group_name], test_size = 0.2)
                all_train_idx += list(train_idx)
                all_eval_idx += list(eval_idx)
            
        ###write to train.tsv and test.tsv
        with open(os.path.join(out_dir, 'train.tsv') , 'w') as f:
            for train_id in all_train_idx:
                f.write('{}\t{}\t{}\n'.format(sents[train_id], races[train_id], sentiments[train_id]))
        with open(os.path.join(out_dir, 'eval.tsv') , 'w') as f:
            for eval_id in all_eval_idx:
                f.write('{}\t{}\t{}\n'.format(sents[eval_id], races[eval_id], sentiments[eval_id]))

# def split_MD_by_race(file_dir = '/home/xiongyi/dataxyz/data/demog-text-removal/sentiment_race',\
#     r0s0_rates = [0.5,0.4,0.3,0.2,0.1], total_num = 10000):
#     races,sentiments, sents = load_moji()
#     race_sentiment_groups = {}
#     for race in [0, 1]:
#         for sentiment in [0, 1]:
#             race_sentiment_id = [i for i in range(len(races)) if races[i] == race and sentiments[i] == sentiment]
#             race_sentiment_groups['{}_{}'.format(race, sentiment)] = race_sentiment_id
#     ###‘1_0’ is the smallest group, use that to calculate sizes of other groups.
#     for r0s0_rate in r0s0_rates:
#         out_dir = os.path.join(file_dir, 'r0s0_{}'.format(r0s0_rate))
#         if not os.path.exists(out_dir):
#             os.makedirs(out_dir)
#         subgroup_sizes = {}
#         sub_groups = {}
#         subgroup_sizes['0_0'] = int (total_num * r0s0_rate * 0.5)
#         subgroup_sizes['0_1'] = int (total_num * (1-r0s0_rate) * 0.5)
#         subgroup_sizes['1_0'] = subgroup_sizes['0_1']
#         subgroup_sizes['1_1'] = subgroup_sizes['0_0']
#         all_train_idx = []
#         all_eval_idx = []
#         for race in [0, 1]:
#             for sentiment in [0, 1]:
#                 group_name = '{}_{}'.format(race, sentiment)
#                 if subgroup_sizes[group_name] > len(race_sentiment_groups[group_name]):
#                     replace = True
#                 else:
#                     replace = False
#                 sub_groups[group_name] = np.random.choice(race_sentiment_groups[group_name],size=subgroup_sizes[group_name], replace=replace)
#                 train_idx, eval_idx = train_test_split(sub_groups[group_name], test_size = 0.2)
#                 all_train_idx += list(train_idx)
#                 all_eval_idx += list(eval_idx)
            
#         ###write to train.tsv and test.tsv
#         with open(os.path.join(out_dir, 'train.tsv') , 'w') as f:
#             for train_id in all_train_idx:
#                 f.write('{}\t{}\t{}\n'.format(sents[train_id], races[train_id], sentiments[train_id]))
#         with open(os.path.join(out_dir, 'eval.tsv') , 'w') as f:
#             for eval_id in all_eval_idx:
#                 f.write('{}\t{}\t{}\n'.format(sents[eval_id], races[eval_id], sentiments[eval_id]))



if __name__ == "__main__":
    split_moji_by_race()