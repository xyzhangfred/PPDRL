#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import os, json
import time
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.autograd import Variable
from graphviz import Digraph
from torch.autograd import Variable
# make_dot was moved to https://github.com/szagoruyko/pytorchviz
from torchviz import make_dot


from tqdm import trange
from transformers import BertTokenizer
from transformers.file_utils import cached_path
from transformers.modeling_bert import BertModel,BertForMaskedLM

from pplm_classification_head import ClassificationHead
from run_pplm import to_var,top_k_filter,get_bag_of_words_indices_bert,build_bows_one_hot_vectors
from run_pplm_bert import perturb_hidden_bert, full_text_generation_bert, generate_text_pplm_bert, run_pplm_example_bert

from data_util import load_yelp

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10
LINE_NUM = 500


QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}



def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS['generic'] = meta

def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id



def pplm_multi_sents(sents, labels):
    all_outputs = []
    for sent, label in zip(sents, labels):
        #assuming labels are either 0 or 1, for each input we generate 1 sentence with the same label and 1 with the opposite label
        
        """
        discrim_weights yelp_bert/generic_classifier_head_epoch_3.pt --discrim_meta yelp_bert/generic_classifier_head_meta.json 
        --cond_text "Potato is the most popular food in the world ?" 
        --class_label 1 --length 100 --gamma 1.5 --num_iterations 3 
        --num_samples 5 --stepsize 0.01  --kl_scale 0.1 --gm_scale 0.95 --sample

        python examples/run_pplm.py -D sentiment --class_label 3 --cond_text 
        "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 
        --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95

        """
        KL = 0.1
        NUM_ITER = 10
        STEP = 0.2
        MASK_PROB = 0.5
        #pick_best or iterative
        strategy = 'pick_best' 
        # print ('SAME LABEL\n')
        same_label_output = run_pplm_example_bert(
            mask_prob = MASK_PROB,
            cond_text=sent,
            num_samples=1,
            bag_of_words=None,
            discrim='generic',
            discrim_weights='yelp_bert/generic_classifier_head_epoch_3.pt',
            discrim_meta='yelp_bert/generic_classifier_head_meta.json' ,
            class_label= label,
            stepsize=STEP,
            temperature=1.0,
            top_k=10,
            sample=True,
            num_iterations=NUM_ITER,
            decay=False,
            gamma=1.0,
            gm_scale=0.9,
            kl_scale=KL,
            seed=0,
            colorama=False,
            verbosity='regular',
            strategy=strategy,
            return_sent= True
        )

        # print (('DIFF LABEL\n'))
        diff_label_output = run_pplm_example_bert(
            mask_prob = MASK_PROB,
            cond_text=sent,
            num_samples=1,
            bag_of_words=None,
            discrim='generic',
            discrim_weights='yelp_bert/generic_classifier_head_epoch_3.pt',
            discrim_meta='yelp_bert/generic_classifier_head_meta.json' ,
            class_label= 1-int(label),
            stepsize=STEP,
            temperature=1.0,
            top_k=10,
            sample=True,
            num_iterations=NUM_ITER,
            decay=False,
            gamma=1.5,
            gm_scale=0.9,
            kl_scale=KL,
            seed=0,
            colorama=False,
            verbosity='regular',
            strategy=strategy,
            return_sent= True
        )

        all_outputs.append( [sent, same_label_output, diff_label_output] )
    return all_outputs

def pplm_to_file(output_file, file_dir = '/home/xiongyi/dataxyz/data/yelp_style_transfer', file_name = 'yelp.train', \
    line_num = 10000,
    kl = 0.1, num_iter = 20, step = 0.1, mask_prob = 0.5, gm_scale = 0.9, gamma = 1.0, strategy = 'pick_best'):
    with open(output_file+'_args', 'w') as f:
        try:
            f.write(str(locals()))
        except:
            pass
    
    labels,sents = load_yelp(file_dir= file_dir,file_name=file_name, line_num = line_num)
    all_outputs = []
    line_count = 0
    start = time.time()
    all_new_score, all_new_sents = [],[]
    corr = 0
    corr_same = 0
    with open(output_file, 'w') as f:
        for sent, label in zip(sents, labels):
            line_count += 1

            if line_count == 10:
                print ('finished 10 lines in {} s'.format(time.time() - start))

            if line_count % 100 == 0:
                print ('finished {} lines in {} s'.format(line_count, time.time() - start))
            #assuming labels are either 0 or 1, for each input we generate 1 sentence with the same label and 1 with the opposite label
            
            """
            python examples/run_pplm.py -D sentiment --class_label 3 --cond_text 
            "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 
            --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
            """
            #pick_best or iterative
            diff_label_output = run_pplm_example_bert(
                mask_prob = mask_prob,
                cond_text=sent,
                num_samples=1,
                bag_of_words=None,
                discrim='generic',
                discrim_weights='yelp_bert/generic_classifier_head_epoch_3.pt',
                discrim_meta='yelp_bert/generic_classifier_head_meta.json' ,
                class_label= 1-int(label),
                stepsize=step,
                temperature=1.0,
                top_k=10,
                sample=True,
                num_iterations=num_iter,
                decay=False,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl,
                colorama=False,
                verbosity='regular',
                strategy=strategy,
                return_sent= True
            )
            new_sent = diff_label_output[2][0][0]
            new_score = diff_label_output[2][0][2]
            new_label = new_score[1] > 0.5
            if new_label == 1-int(label):
                corr += 1
            
            all_new_sents.append(new_sent)
            all_new_score.append(new_score)
            f.write(sent + '\t' + label +'\t' +new_sent +'\t' +str(new_score[0]) +'\n')
            f.flush()
            # print (('DIFF LABEL\n'))
            same_label_output = run_pplm_example_bert(
                mask_prob = mask_prob,
                cond_text=sent,
                num_samples=1,
                bag_of_words=None,
                discrim='generic',
                discrim_weights='yelp_bert/generic_classifier_head_epoch_3.pt',
                discrim_meta='yelp_bert/generic_classifier_head_meta.json' ,
                class_label= int(label),
                stepsize=step,
                temperature=1.0,
                top_k=10,
                sample=True,
                num_iterations=num_iter,
                decay=False,
                gamma=gamma,
                gm_scale=gm_scale,
                kl_scale=kl,
                colorama=False,
                verbosity='regular',
                strategy=strategy,
                return_sent= True
            )
            new_sent = same_label_output[2][0][0]
            new_score = same_label_output[2][0][2]
            new_label = new_score[1] > 0.5
            if new_label == 1-int(label):
                corr_same += 1
            print (sent, label)
            print ('*****DIFF LABEL *****')

            print (diff_label_output[0])
            print (diff_label_output[1][0])
            print (diff_label_output[2][0][1])
            print (diff_label_output[2][0][2])

            print ('*****SAME LABEL *****')
            print (same_label_output[0])
            print (same_label_output[1][0])
            print (same_label_output[2][0][1])
            print (same_label_output[2][0][2])

        acc = corr/len(sents)
        acc_same = corr_same/len(sents)
        with open(output_file+'_args', 'w') as f:
            f.write('\n' + 'accuracy: {}'.format(acc))
        print ('accuracy ', corr/len(sents))
        print ('accuracy same ', acc_same)

    reorganize_results(all_outputs = all_outputs, out_file_name = 'out_sents' ,sents = sents, labels = labels)
    
    return all_outputs,acc




def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--num_samples",
    #     type=int,
    #     default=1,
    #     help="Number of samples to generate from the modified latents",
    # )
    # parser.add_argument(
    #     "--discrim",
    #     "-D",
    #     type=str,
    #     default=None,
    #     choices=("clickbait", "sentiment", "toxicity", "generic"),
    #     help="Discriminator to use",
    # )
    # parser.add_argument('--discrim_weights', type=str, default=None,
    #                     help='Weights for the generic discriminator')
    # parser.add_argument('--discrim_meta', type=str, default=None,
    #                     help='Meta information for the generic discriminator')
    # parser.add_argument(
    #     "--class_label",
    #     type=int,
    #     default=-1,
    #     help="Class label used for the discriminator",
    # )
    # parser.add_argument("--length", type=int, default=100)
    # parser.add_argument("--stepsize", type=float, default=0.02)
    # parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument("--top_k", type=int, default=10)
    # parser.add_argument(
    #     "--sample", action="store_true",
    #     help="Generate from end-of-text as prefix"
    #     #what does this mean??
    # )
    # parser.add_argument("--num_iterations", type=int, default=3)
    # parser.add_argument("--gamma", type=float, default=1.5, help = "")
    # parser.add_argument("--gm_scale", type=float, default=0.9)
    # parser.add_argument("--kl_scale", type=float, default=0.01, help = "weight of kl loss.")
    # parser.add_argument("--seed", type=int, default=0)
    # parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    # parser.add_argument("--colorama", action="store_true",
    #                     help="colors keywords")
    # parser.add_argument("--verbosity", type=str, default="very_verbose",
    #                     choices=(
    #                         "quiet", "regular", "verbose", "very_verbose"),
    #                     help="verbosiry level")

    parser.add_argument("--output_file","-o", type=str, required=True,
                        help="where to put the generated sents")
    parser.add_argument("--line_num","-l", type=int, default=100,
                        help="where to put the generated sents")
    parser.add_argument("--mask_prob","-m", type=float, default=0.3,
                        help="where to put the generated sents")
    parser.add_argument("--strategy","-s", type=str, default='pick_best',
                        help="pick_best or iterative")
    parser.add_argument("--stepsize", type=float, default=0.1)
    parser.add_argument("--num_iter", type=int, default=10)

    args = parser.parse_args()

    all_outputs,acc = pplm_to_file(args.output_file, line_num=args.line_num, mask_prob=args.mask_prob, strategy=args.strategy, step=args.stepsize,\
        num_iter = args.num_iter)
    #would it take too long? Probably...
    

def reorganize_results(all_outputs, out_file_name,sents, labels):
    #reorganize the results
    all_diff_score_compare = []
    all_diff_score = []
    with open(out_file_name, 'w') as f:
        for i,output in enumerate(all_outputs):
            label = labels[i]
            print ('*' * 15 + '\n')
            print ('{} {} '.format(sents[i], labels[i]))
            print (str(output[1][1][0]))
            #output =  [sent, same_label_output, diff_label_output] 
            print ('SAME LABEL\n')
            print (output[1][0])
            print (output[1][1][1])
            print (str(output[1][2][0][0]) +' '+ str(output[1][2][0][2]))
            print ('DIFF LABEL\n')

            print (output[2][0])
            print (output[2][1][1])
            # print ('After Filling in, Score')
            print (output[2][2][0][0], output[2][2][0][2])
            diff_score_compare = output[1][2][0][2][1-int(label)] - output[1][1][0][1-int(label)]
            all_diff_score_compare.append(diff_score_compare)

            diff_score = output[2][2][0][2][1-int(label)] - output[1][1][0][1-int(label)]
            all_diff_score.append(diff_score)
    print (np.mean(all_diff_score_compare))
    print (np.mean(all_diff_score))
    


if __name__ == "__main__":
    main()

