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
import json
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
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
PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

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
def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def perturb_hidden_bert(
        all_hidden,
        model,
        masked,
        context = None,
        masked_lm_labels = None,
        unpert_all_hidden=None,
        unpert_logits=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR
):
    #This function perturb the hidden representation and output the new hid-rep (for all the masked tokens)

    # Generate inital perturbed past
    # Here initialize grad, should only be w.r.t. the parts we are actually updating tho
    # We don't perturb the last layer since it's not meaningful!
    grad_accumulator = [
        (np.zeros(p_.shape).astype("float32"))
        for p_ in all_hidden[:-1]
    ]

    decay_mask = 1.0
    #TODO  Change this mask so that only unmasked hiddens are updated, can experiment with different setting
    if True:
        window_mask = torch.ones_like(all_hidden[0]).to(device)
    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    sentence_length = all_hidden[0].shape[1]
    masked_indices = np.where(masked_lm_labels[0].cpu().numpy() > 0)[0]
    is_masked = torch.ones_like(all_hidden[0]).to(device)
    #masked token = 1, unmasked = 0
    is_masked[0][masked_indices] = 0
    unpert_hidden = all_hidden

    #we never modify the accumulated_hidden of unmasked tokens!
    accumulated_unmasked_hidden = (unpert_hidden[-1] * (1-is_masked)).sum(1)[0]
    for i in range(num_iterations):
        #in each iteration, update something

        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed hidden
        perturbed_hidden = list(map(add, all_hidden[:-1], curr_perturbation))
        # curr_perturbation[2].register_hook(print('current grad layer [1]', curr_perturbation[0].grad))
        # _, _, _, curr_length, _ = curr_perturbation[0].shape
        ##manually calculate bert outputs
        new_hidden = [h for h in all_hidden]
        for layer_no,layer in enumerate(model.bert.encoder.layer):
            #calculate the output of this layer by applying the module of this layer to hidden of previous layers
            input_hidden = perturbed_hidden[layer_no] * is_masked + new_hidden[layer_no] * (1-is_masked)
            new_hidden[layer_no+1] = layer(hidden_states = input_hidden)[0] 
            
        # output_hidden = model.bert.encoder.layer[-1](hidden_states = perturbed_hidden[-1])[0] 
        #### _, all_pred_logits, all_hidden = model(input_ids = context, masked_lm_labels = masked_lm_labels)
        #use last layer hidden output for prediction
        all_pred_logits = model.cls(new_hidden[-1]) 
        # getBack(loss.grad_fn)


        # make_dot(loss).view()
        hidden = all_hidden[-1]
        accumulated_masked_hidden = (new_hidden[-1] * is_masked).sum(1)[0]
        new_accumulated_hidden = accumulated_unmasked_hidden + accumulated_masked_hidden
        # new_accumulated_hidden = None
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        #TODO Fix this, GPT2 might be different with Bert here
        #only take the masked token logits for loss calculation
        logits = all_pred_logits[0, masked_indices]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())
        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()

            # curr_score = torch.unsqueeze(probs, dim=1)
            ##For bert horizon_length should always be 0! so the below part is not necessary
            horizon_length = 0
            # wte = model.resize_token_embeddings()
            # for _ in range(horizon_length):
            #     inputs_embeds = torch.matmul(curr_score, wte.weight.data)
            #     _, curr_unpert_past, curr_all_hidden = model(
            #         past=curr_unpert_past,
            #         inputs_embeds=inputs_embeds
            #     )
            #     curr_hidden = curr_all_hidden[-1]
            #     new_accumulated_hidden = new_accumulated_hidden + torch.sum(
            #         curr_hidden, dim=1)
            #TODO do we include the bos and eos tokens?
            prediction = classifier(new_accumulated_hidden /
                                    sentence_length)

            label = torch.tensor(class_label,
                                 device=device,
                                 dtype=torch.long)
            #had to have a dimension for batchsize
            discrim_loss = ce_loss(prediction.unsqueeze(0), label.unsqueeze(0))
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            #calculate kl loss
            unpert_probs = F.softmax(unpert_logits[0, masked_indices, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        # TODO Double check if we indid only calculated grad on (K,V) pairs of the masked tokens
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)]
        else:
            grad_norms = [ (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        # new_past = []
        # for p_ in past:
        #     new_past.append(p_.detach())
        # past = new_past

    # apply the accumulated perturbations to the past
    ##
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    perturbed_hidden = list(map(add, all_hidden, grad_accumulator))

    return perturbed_hidden, new_accumulated_hidden, grad_norms, loss_per_iter


def full_text_generation_bert(
        model,
        tokenizer,
        context=None,
        masked_indices = None,
        masked_lm_labels = None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        **kwargs
):
    classifier, class_id = get_classifier( discrim, class_label, device )

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices_bert(bag_of_words.split(";"),
                                               tokenizer)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm_bert(
        #this is just letting bert does its job without changing anything
        model=model,
        tokenizer=tokenizer,
        masked_indices = masked_indices,
        masked_lm_labels=masked_lm_labels,
        context=context,
        device=device,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []
    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm_bert(
            model=model,
            tokenizer=tokenizer,
            context=context,
            masked_indices = masked_indices,
            masked_lm_labels=masked_lm_labels,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm_bert(
        model,
        tokenizer,
        context=None,
        masked_indices = None,
        masked_lm_labels = None,
        all_hidden=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR
):
    context_t = torch.tensor(context, device=device, dtype=torch.long)
    while len(context_t.shape) < 2:
        context_t = context_t.unsqueeze(0)
    output_so_far = context_t
    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,device)
    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    # if verbosity_level >= VERBOSE:
    #     range_func = trange(len(masked_indices), ascii=True)
    # else:
    #     range_func = range(len(masked_indices))

    # for now, predict all the masked words at the same time
    # Get past/probs for current output, except for last word
    # Note that GPT takes 2 inputs: past + current_token
    # run model forward to obtain unperturbed
    
    if all_hidden is None and output_so_far is not None:
        #if there is some context
        if output_so_far.shape[1] > 1:
            _, _,all_hidden = model(input_ids = output_so_far, masked_lm_labels = masked_lm_labels)
    # sequence_output, pooled_output, (hidden_states), (attentions)
    unpert_mlm_loss, unpert_pred_logits, unpert_all_hidden = model(input_ids = context_t, masked_lm_labels = masked_lm_labels)
    # unpert_all_hidden : 12+1 layers, [1,sent_len,768]
    # get the last layer, this will be used for prediction and calc loss
    unpert_hidden_ll = unpert_all_hidden[-1]

    #finished unperturbed outputing
    # check if we are abowe grad max length
    # if i >= grad_length:
    #     current_stepsize = stepsize * 0
    # else:
    #     current_stepsize = stepsize
    current_stepsize = stepsize
    # modify the past if necessary
    if not perturb or num_iterations == 0:
        pert_all_hidden = unpert_all_hidden

    else:
        #TODO fix this
        if all_hidden is not None:
            pert_all_hidden, _, grad_norms, loss_this_iter = perturb_hidden_bert(
                all_hidden,
                model,
                masked= None,
                context = context_t,
                masked_lm_labels= masked_lm_labels,
                unpert_all_hidden=unpert_all_hidden,
                unpert_logits=unpert_pred_logits,
                grad_norms=grad_norms,
                stepsize=current_stepsize,
                one_hot_bows_vectors=one_hot_bows_vectors,
                classifier=classifier,
                class_label=class_label,
                loss_type=loss_type,
                num_iterations=num_iterations,
                horizon_length=horizon_length,
                decay=decay,
                gamma=gamma,
                kl_scale=kl_scale,
                device=device,
                verbosity_level=verbosity_level
            )
            loss_in_time.append(loss_this_iter)
        else:
            pert_past = past
    #feed the perturbed hidden representations to the model and get new hidden representations
    # Oh no, we need to feed the hidden representations instead of input ids to the model!!!! Argh!
    last_hidden = pert_all_hidden[-1]
    pert_logits = model.cls(last_hidden)
    pert_probs = F.softmax(pert_logits, dim=-1)
    # pert_mlm_loss, pert_pred_logits, pert_all_hidden = model(input_ids = context_t, masked_lm_labels = masked_lm_labels)
    ###here only take the last logit as the output
    pert_logits = pert_logits[0] / temperature  # + SMALL_CONST
    pert_probs = F.softmax(pert_logits, dim=-1)

    if classifier is not None:
        ce_loss = torch.nn.CrossEntropyLoss()
        prediction = classifier(torch.mean(unpert_hidden_ll, dim=1))
        label = torch.tensor([class_label], device=device,
                                dtype=torch.long)
        unpert_discrim_loss = ce_loss(prediction, label)
        if verbosity_level >= VERBOSE:
            print(
                "unperturbed discrim loss",
                unpert_discrim_loss.data.cpu().numpy()
            )
    else:
        unpert_discrim_loss = 0

    # Fuse the modified model and original model
    if perturb:

        unpert_probs = F.softmax(unpert_pred_logits[0], dim=-1)

        pert_probs = ((pert_probs ** gm_scale) * (
                unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
        pert_probs = top_k_filter(pert_probs, k=top_k,
                                    probs=True)  # + SMALL_CONST

        # rescale
        if torch.sum(pert_probs) <= 1:
            pert_probs = pert_probs / torch.sum(pert_probs)

    else:
        pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

    # sample or greedy
    # TODO Actually we might not need to sample the actual words, but just take the representations.
    if sample:
        prediction = torch.multinomial(pert_probs, num_samples=1)

    else:
        _, prediction = torch.topk(pert_probs, k=2, dim=-1)

    output_so_far = [context[i] if masked_lm_labels[0][i].item() == -100 else prediction[i].item()  for i in range(len(context))]
    if verbosity_level >= REGULAR:
        print(tokenizer.decode(output_so_far))

    return output_so_far, unpert_discrim_loss, loss_in_time

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

def get_score(token_ids, model, classifier, device):
    token_ids_tensor = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)
    _,all_hidden = model(input_ids = token_ids_tensor)
    accumulated_hidden = all_hidden[-1].sum(1)[0]
    sentence_length = len(all_hidden)
    prob = F.softmax(classifier(accumulated_hidden / sentence_length))
    score = [p.item() for p in prob]
    return score


def selective_mask(raw_text, mask_prob, model, tokenizer, classifier, class_id, device, strategy = 'pick_best'):
    # The goal is to mask our tokens to maximize the prob of the class_id
    tokens = tokenizer.tokenize(raw_text)
    #Notice that len(token_ids) = len(tokens) + 2
    token_ids = tokenizer.encode(raw_text) 
    sentence_length = len(token_ids)
    #at least mask out 1 token
    mask_num = max(int(mask_prob * (sentence_length - 2)), 1)
    scores = []
    curr_tokens = tokens 
    curr_token_ids = token_ids 
    curr_score = None
    masked_indices = []
    
    init_score = get_score(token_ids, model, classifier, device)
    if strategy == 'iterative':
        print (tokens)
        for i in range(mask_num):
            # print ('Masking the {}th token of all {} tokens to be masked.'.format(i, mask_num))
            curr_token_ids_tensor = torch.tensor(curr_token_ids, dtype=torch.long).unsqueeze(0).to(device)
            _,all_hidden = model(input_ids = curr_token_ids_tensor)
            if curr_score is None:
                curr_score = init_score
            all_new_probs = []
            for tok_no, tok in enumerate(curr_tokens):
                if tok == '[MASK]' or tok in [',', '.', ':',';', '?', '!']:
                    all_new_probs.append(-1000)
                    continue
                #iteratively mask out each tok and see how the score changes
                masked_token_ids = curr_token_ids.copy()
                #plus one because <bos> token
                masked_token_ids[tok_no + 1] = tokenizer.mask_token_id
                new_score = get_score(masked_token_ids, model, classifier, device)
                all_new_probs.append(new_score[class_id])
                diff = new_score[class_id] - curr_score[class_id]
                # print ('{} , newprob {}, diff = {}'.format(tok, new_score, diff))
            best_tok_id = np.argmax(all_new_probs)
            #TODO needs fixing
            masked_score = all_new_probs[best_tok_id]
            # print ('Masking out "{}" '.format(curr_tokens[best_tok_id]))
            curr_tokens[best_tok_id] = '[MASK]'
            curr_token_ids[best_tok_id + 1] = tokenizer.mask_token_id
            #added 1 here
            masked_indices.append(best_tok_id +1)
        # print ('After masking --- "{}" '.format(' '.join(curr_tokens)) )
        # print ('masked_indices: {} '.format(masked_indices) )
        
    elif strategy == 'pick_best':
        num_reinit = 10
        # re init several times, each time mask out same amount of words, 
        # pick the sentence with the highest score of the target class?
        # or lowest score for the initial class?
        all_score, all_masked_indices = [],[]
        for _ in range(num_reinit):
            curr_token_ids = token_ids.copy()
            masked_indices_cand = np.random.choice(range(len(tokens)), mask_num)
            all_masked_indices.append(masked_indices_cand)
            for m_id in masked_indices_cand:
                curr_token_ids[m_id + 1] = tokenizer.mask_token_id
            print (curr_token_ids)
            new_score = get_score(curr_token_ids, model, classifier, device)
            all_score.append(new_score)
        best_ind = np.argmax([s[class_id] for s in all_score ])
        masked_indices = all_masked_indices[best_ind]
        #add 1 to acount for the fact that there is the [cls] token
        masked_indices = [m + 1 for m in masked_indices]
        masked_score = all_score[best_ind]
    
    return masked_indices, init_score, masked_score

def run_pplm_example_bert(
        pretrained_model="bert-base-cased",
        mask_prob = 0.5,
        do_selective_mask = True,
        cond_text="",
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        strategy = 'pick_best',
        verbosity='regular',
        return_sent = False
):
    ##This is the main function for bert

    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    ###set discriminator? TODO need to figure this part out. where is this used
    #Modifies the global variables
    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)
    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    model = BertForMaskedLM.from_pretrained(
        pretrained_model,
        output_hidden_states=True
    )
    model.to(device)
    model.eval()

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    # Freeze Bert weights
    #!! this is interesting, i should also do this for my code.
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    # if uncond, use start of sentence as the prompt
    # we need to change this into a whole sentence
   
    raw_text = cond_text

    while not raw_text:
        print("Did you forget to add `--cond_text`? ")
        raw_text = input("Model prompt >>> ")
    ##Different: we are also adding eos token now (as opposed to only bos)
    tokenized_cond_text = tokenizer.encode(raw_text)

    print("= Original sentence =")
    print(tokenizer.decode(tokenized_cond_text))
    print()
    #randomly mask out a certain percentage of tokens or do_selective
    sent_len = len(tokenized_cond_text)-2
    # masked_indices = np.random.choice( range(1, len(tokenized_cond_text)-1), int(sent_len * mask_prob))

    #add a function to mask out indices that 
    if discrim is not None and do_selective_mask:
        classifier, class_id = get_classifier( discrim, class_label, device )
        masked_indices, init_score, masked_score = selective_mask(raw_text, mask_prob, model, tokenizer, classifier, class_id, device, strategy)
    orig_scores = [init_score, masked_score, ]

    # masked_indices = np.array([5,6,7])
    
        # get the mask labels
        # Fuck they changed the ignore_index!!!!
    masked_lm_labels = [[-100 for _ in range(len(tokenized_cond_text))]]
    for ind in masked_indices:
        masked_lm_labels[0][ind] = tokenized_cond_text[ind]
    masked_lm_labels = torch.tensor(masked_lm_labels, device=device, dtype=torch.long)
    for ind in masked_indices:
        tokenized_cond_text[ind] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    #PRINT the masked version of the input_text
    print ("After masking")
    masked_text = tokenizer.decode(tokenized_cond_text)
    print(masked_text)


    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    # bert-completed sentence without perterbing 
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation_bert(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        masked_indices = masked_indices,
        masked_lm_labels = masked_lm_labels,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level
    )

    # untokenize unperturbed text
    print ('UNPERT\n')
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text)

    if verbosity_level >= REGULAR:
        print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    print()

    generated_texts = []

    bow_word_ids = set()
    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices_bert(bag_of_words.split(";"),
                                               tokenizer)
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)
    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if colorama:
                import colorama

                pert_gen_text = ''
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += '{}{}{}'.format(
                            colorama.Fore.RED,
                            tokenizer.decode([word_id]),
                            colorama.Style.RESET_ALL
                        )
                    else:
                        pert_gen_text += tokenizer.decode([word_id])
            else:
                pert_gen_text = tokenizer.decode(pert_gen_tok_text)

            print("= Perturbed generated text {} =".format(i + 1))
            print(pert_gen_text)
            print()
        except:
            pass
        # keep the prefix, perturbed seq, original seq for each index
        # return should contain: masked sentence, pert_gen_text, unpert_gen_text
        # scores = [initial_score, score_after_masking, score_after_filling_in]
        new_score = get_score(pert_gen_tok_text, model, classifier, device)
        generated_texts.append(
            (pert_gen_text, unpert_gen_text, new_score)
        )
    if return_sent:
        return [masked_text, orig_scores, generated_texts]
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="bert-base-cased",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
        #what does this mean??
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    # parser.add_argument(
    #     "--window_length",
    #     type=int,
    #     default=0,
    #     help="Length of past which is being optimized; "
    #          "0 corresponds to infinite window length",
    # )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")

    parser.add_argument("--strategy", type=str, default='pick_best')
    

    args = parser.parse_args()
    run_pplm_example_bert(**vars(args))
