# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
""" Finetuning the library models for sequence classification on GLUE (Bert, XLM, XLNet, RoBERTa, Albert, XLM-RoBERTa)."""


import logging
import os
import sys
import argparse
import time
from datetime import datetime as datetime

from tqdm import tqdm, trange
import numpy as np
import torch
import torch.optim as optim

from bert_util import tokenize, InputExample, convert_examples_to_features,get_dataloader
from data_util import load_yelp, load_imdb, load_moji_split
from transformers import BertTokenizer, BertForSequenceClassification

from pp_rep_learning import calc_tpr

def train(
        train_dataloader,
        eval_dataloader,
        model,
        epochs=10,
        learning_rate=0.0001,
        batch_size=64,
        log_interval=10,
        save_model=False,
        cached=False,
        no_cuda=False,
        output_fp='.'
):
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    eval_losses = []
    eval_accuracies = []

    for epoch in range(epochs):
        start = time.time()
        print("\nEpoch", epoch + 1)

        train_epoch(data_loader=train_dataloader, model=model, optimizer=optimizer
        , epoch=epoch, log_interval=log_interval, device=device)
        
        eval_loss, eval_accuracy = evaluate_performance(
            data_loader=eval_dataloader,
            model=model,
            device=device
        )

        end = time.time()
        print("Epoch took: {:.3f}s".format(end - start))

        eval_losses.append(eval_loss)
        eval_accuracies.append(eval_accuracy)

        # print("\nExample prediction")
        # predict(example_sentence, discriminator, idx2class,
        #         cached=cached, device=device)

    min_loss = float("inf")
    min_loss_epoch = 0
    max_acc = 0.0
    max_acc_epoch = 0
    print("Test performance per epoch")
    print("epoch\tloss\tacc")
    for e, (loss, acc) in enumerate(zip(eval_losses, eval_accuracies)):
        print("{}\t{}\t{}".format(e + 1, loss, acc))
        if loss < min_loss:
            min_loss = loss
            min_loss_epoch = e + 1
        if acc > max_acc:
            max_acc = acc
            max_acc_epoch = e + 1
    print("Min loss: {} - Epoch: {}".format(min_loss, min_loss_epoch))
    print("Max acc: {} - Epoch: {}".format(max_acc, max_acc_epoch))
    # If bert then do not add? Why?
    if save_model:
        save_dir = os.path.join(output_fp, 'finetuned_bert')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        model.save_pretrained(save_dir)
    return model




def train_epoch(data_loader, model, optimizer,
                epoch=0, log_interval=10, device='cpu'):
    print ('device :', device)
    samples_so_far = 0
    model.train()
    for batch_idx, batch in enumerate(data_loader):
        batch = [t.to(device) for t in batch]
        inputs = {'input_ids':      batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'labels': batch[3] }

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs[0]
        loss.backward()
        optimizer.step()

        samples_so_far += len(batch[0])

        if batch_idx % log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch + 1,
                    samples_so_far, len(data_loader.dataset),
                    100 * samples_so_far / len(data_loader.dataset), loss.item()
                )
            )


def evaluate_performance(data_loader, model, device='cpu'):
    eval_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = [t.to(device) for t in batch]
            inputs = {'input_ids':      batch[0],
                            'attention_mask': batch[1],
                            'token_type_ids': batch[2],
                            'labels': batch[3] }

            outputs = model(**inputs)
            reps = outputs[1]
            # sum up batch loss
            eval_loss += outputs[0].item()
            # get the index of the max log-probability
            preds = outputs[1].argmax(dim=1, keepdim=True)
                

            targets =batch[3]
            correct += preds.eq(targets.view_as(preds)).sum().item()
    eval_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)
   

    print(
        "Performance on test set: "
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            eval_loss, correct, len(data_loader.dataset),
            100. * accuracy
        )
    )

    return eval_loss, accuracy




def main():
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", default="bert-base-uncased", type=str,    
                        help="The pretrained transformer model")

    parser.add_argument("--dataset", default='moji', type=str,    
                        help="pick from yelp, imdb, moji")
    parser.add_argument("--dataset_fp", default='/home/xiongyi/dataxyz/data/demog-text-removal/sentiment_race/r0s0_0.1', type=str,    
                        help="dataset file path")
    parser.add_argument("--epochs", type=int, default=3, metavar="N",
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", default=5e-5, type=float,    
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int,    
                        help="epochs to train")
    parser.add_argument("--batch_size", type=int, default=32, metavar="N",
                        help="input batch size for training (default: 64)")
    parser.add_argument("--log_interval", type=int, default=10, metavar="N",
                        help="how many batches to wait before logging training status")
    parser.add_argument("--output_fp", default='finetuned_models', type=str,    
                        help="output dir")

    parser.add_argument("--protected_group", default=None, type=int,    
                        help="if not None, train only on selected group and make it balanced")
    args = parser.parse_args()
    args.device = torch.device("cuda")
    if not os.path.exists(args.output_fp):
        os.makedirs(args.output_fp)
    #add a step for training classifiers, two types --- finetuned and average hidden rep

    model = BertForSequenceClassification.from_pretrained(args.pretrained_model,  output_hidden_states=True)
    model.to(args.device)
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model)

    if args.dataset == 'imdb':
        file_name = 'train_Drama_pos_ratio_{}.tsv'.format(args.g0_pos_ratio)
        sents, sentiments, genres = load_imdb(file_dir = '/home/xiongyi/dataxyz/data/imdb/tsv', file_name = file_name, \
        line_num = args.num_data)
        main_labels = sentiments
        prot_labels = genres
        train_num = int(len(sents) * 0.9)
        print ('train num:', train_num)
        train_main_labels = main_labels[:train_num]
        train_sents = sents[:train_num]
        eval_main_labels = main_labels[train_num:]
        eval_sents = sents[train_num:]
    elif args.dataset == 'moji':
        train_races,train_sentiments, train_sents = load_moji_split(file_dir = args.dataset_fp, prefix = 'train')
        eval_races,eval_sentiments, eval_sents = load_moji_split(file_dir = args.dataset_fp, prefix = 'eval')
        train_main_labels = train_sentiments
        train_prot_labels = train_races
        eval_main_labels = eval_sentiments
        eval_prot_labels = eval_races
        
    #if train on only one group
    if args.protected_group is not None:
        train_idx = [i for i in range(len(train_sents)) if train_prot_labels[i] == args.protected_group]
        eval_idx = [i for i in range(len(eval_sents)) if eval_prot_labels[i] == args.protected_group]
        train_sents = [train_sents[i] for i in train_idx]
        train_main_labels = [train_main_labels[i] for i in train_idx]
        eval_sents = [eval_sents[i] for i in eval_idx]
        eval_main_labels = [eval_main_labels[i] for i in eval_idx]


    if args.dataset == 'imdb':
        max_seq_length = 512
        label_list = [0, 1]
        
    elif args.dataset == 'yelp':
        max_seq_length = 128
        label_list = ['0', '1']

    elif args.dataset == 'moji':
        max_seq_length = 128
        label_list = [0, 1]

    print ('label list: ', label_list)
    train_input_examples = [InputExample(guid=i, text_a=train_sents[i], label=train_main_labels[i])   for i in range(len(train_sents))]
    train_input_features = convert_examples_to_features(examples = train_input_examples, label_list=label_list, max_seq_length = max_seq_length, tokenizer=tokenizer)
    train_dataloader = get_dataloader(train_input_features, batch_size=args.batch_size)

    eval_input_examples = [InputExample(guid=i, text_a=eval_sents[i], label=eval_main_labels[i])   for i in range(len(eval_sents))]
    eval_input_features = convert_examples_to_features(examples = eval_input_examples, label_list=label_list, max_seq_length = max_seq_length, tokenizer=tokenizer)
    eval_dataloader = get_dataloader(eval_input_features, batch_size=args.batch_size)
    

    train(
        train_dataloader,
        eval_dataloader,
        model,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_model=True,
        output_fp=args.output_fp
    )
    
    

if __name__ == "__main__":
    main()