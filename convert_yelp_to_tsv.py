import os
import random
FILE_DIR = '/home/xiongyi/dataxyz/data/yelp_style_transfer'

def convert_yelp():
    for prefix in ['train', 'dev', 'test']:
        all_lines = []
        for class_name in [0,1]:
            fn = 'sentiment.{}.{}'.format(prefix, class_name)
            with open(os.path.join(FILE_DIR, fn), 'r') as f:
                lines = f.readlines()
                lines = [str(class_name)+ '\t' + line.strip() + '\n' for line in lines]
            all_lines = all_lines + lines
        random.shuffle(all_lines)
        save_fn = 'yelp.{}'.format(prefix)
        with open(os.path.join(FILE_DIR, save_fn), 'w') as f:
            f.writelines(all_lines)

def convert_imdb(filepath = '/home/xiongyi/dataxyz/repos/Mine/PPDRL/full_train_Drama_pos_ratio_-1.tsv', output_path = 'full_train_Drama_pos_ratio_-1_genre_only.tsv'):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    with open(output_path, 'w') as f:
        for line in lines:
            text, sent, genre = line.strip().split('\t')
            new_line = genre +'\t'+text +'\n'
            f.write(new_line)

convert_imdb()