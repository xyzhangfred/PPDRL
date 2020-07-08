import argparse
import os
from datetime import datetime as datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from operator import add
from transformers import BertTokenizer, BertModel
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter
from sklearn.linear_model import LogisticRegression

from bert_util import tokenize, InputExample, convert_examples_to_features,get_dataloader
from data_util import load_yelp, load_imdb,load_moji, load_moji_split
from run_pplm import to_var, get_classifier_new
from loss_funcs import NTXentLoss



def perturb_hidden(model, all_hidden,masks,labels, classifier, args, only_last = True,num_iterations=1):
    #

    """
    function for shifting the hidden states of Bert
    :param model: bert model
    :param all_hidden: the hidden states before shifting, [batch_size, layer_num, hidden_dim]
    :classifier the pre-trained classifier 
    :return: the shifted hidden-states
    """
    last_hidden_old = all_hidden[-1].clone()
    device = args.device
    if only_last:
        grad_accumulator = (np.zeros(all_hidden[-1].shape).astype("float32"))
    else: 
        grad_accumulator = [ (np.zeros(h.shape).astype("float32")) for h in all_hidden ]
    # accumulate perturbations for num_iterations
    new_accumulated_hidden = None
    sentence_length = masks.sum(1)
    #masked token = 1, unmasked = 0

    #we never modify the accumulated_hidden of unmasked tokens!
    for i in range(num_iterations):
        #in each iteration, update something
        curr_perturbation = to_var(torch.from_numpy(grad_accumulator), requires_grad=True, device=device)
        unmasked_perturbation = curr_perturbation * masks.unsqueeze(-1)
        #zero out masked indices

        # Compute hidden using perturbed hidden
        # perturbed_hidden = list(map(add, all_hidden[-1], unmasked_perturbation))
        perturbed_hidden = all_hidden[-1] + unmasked_perturbation
       
        new_accumulated_hidden = (perturbed_hidden * masks.unsqueeze(-1)).sum(1)
        loss = 0.0
        
        ce_loss = torch.nn.CrossEntropyLoss()

        #TODO do we include the bos and eos tokens?
        predictions = classifier(new_accumulated_hidden /
                                sentence_length.unsqueeze(1))

        new_labels = torch.tensor(1-labels,
                                device=device,
                                dtype=torch.long)
        #had to have a dimension for batchsize
        discrim_loss = ce_loss(predictions, new_labels).sum()
        loss += discrim_loss

        reg_w = args.reg_weight
        reg = torch.norm(curr_perturbation)
        loss += reg_w * reg

        # compute gradients
        loss.backward(retain_graph=True)
        # calculate gradient norms
        grad_norms = torch.norm(curr_perturbation.grad + 1e-7)  
        # normalize gradients
        grad = -args.stepsize * (curr_perturbation.grad/ grad_norms ** args.gamma).data.cpu().numpy()

        # accumulate gradient
        # grad_accumulator = list(map(add, grad, grad_accumulator))
        grad_accumulator += grad
        # reset gradients, just to make sure
        curr_perturbation.grad.data.zero_()
    # apply the accumulated perturbations to the past
    ##
    grad_accumulator = to_var(torch.from_numpy(grad_accumulator), requires_grad=True, device=device)
    perturbed_last_hidden = all_hidden[-1]+ grad_accumulator
    last_hidden_new = all_hidden[-1].clone()

    return perturbed_last_hidden




class Projector(nn.Module):
    def __init__(self, input_dim, output_dim = 256, hidden_dim = None):
        super(Projector, self).__init__()
        if hidden_dim is None:
            hidden_dim = output_dim
        self.mlp = nn.Sequential(
          nn.Linear(input_dim,hidden_dim),
          nn.ReLU(),
          nn.Linear(hidden_dim,output_dim),
        )

    def forward(self, hidden, mask):
        sent_length = mask.sum(1).unsqueeze(1)

        sum_hidden = (hidden * mask.unsqueeze(-1)).sum(1)
        avg_hidden = sum_hidden/sent_length
        z = self.mlp(avg_hidden)
        return z


def eval_loss(model, classifier, projector, eval_dataloader, args, tb_writer, curr_step):
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    step_num = 0
    if args.loss_type == 'triplet':
        loss_fn =  torch.nn.TripletMarginLoss(reduction='sum')
    elif args.loss_type == 'clr':
        loss_fn =  NTXentLoss(device=args.device, batch_size = args.eval_batch_size, temperature=1, use_cosine_similarity=True)
    eval_loss = 0
    for step, batch in enumerate(epoch_iterator):
        step_num += 1
        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            if len(batch[0]) < args.eval_batch_size:
                continue
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]}
            last_hidden, pooler_output, all_hidden = model(**inputs)
        pert_last_hidden = perturb_hidden(model=model,all_hidden=all_hidden,masks=batch[1],labels=batch[3],\
                classifier=classifier,args = args,only_last= True,num_iterations=5)
        with torch.no_grad():
            z = projector(last_hidden,batch[1])
            pert_z = projector(pert_last_hidden,batch[1])
            if args.loss_type == 'triplet':
                neg_idx = torch.randperm(n=len(batch[0]))
                neg_z = z[neg_idx,:]
                eval_loss += loss_fn(z,pert_z,neg_z)/len(batch[0])
            elif args.loss_type == 'clr':
                eval_loss += loss_fn(z,pert_z)

    eval_loss = eval_loss / len(epoch_iterator)
    tb_writer.add_scalar('eval/loss', eval_loss.item(), curr_step)

    return eval_loss


def eval_probe(model, projector, eval_dataloader, args, tb_writer, labels,curr_step, label_name = 'protected'):
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    global_step = 0
    all_z = None
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]}
        with torch.no_grad():
            last_hidden, pooler_output, all_hidden = model(**inputs)
            z = projector(last_hidden,batch[1])
            if all_z is None:
                all_z = z
            else:
                all_z=torch.cat((all_z,z), dim=0)
    num_data = len(labels)
    
    train_idx = np.random.choice(range(num_data), int(num_data * 0.8), replace=False)
    eval_idx = [i for i in range(num_data) if i not in train_idx]
    train_z = all_z[train_idx,:].detach().cpu().numpy()
    eval_z = all_z[eval_idx,:].detach().cpu().numpy()
    train_labels = [int(labels[i]) for i in train_idx]
    eval_labels = [int(labels[i]) for i in eval_idx]
    clf = LogisticRegression(random_state=0).fit(train_z, train_labels)
    train_acc = clf.score(train_z, train_labels)
    eval_acc = clf.score(eval_z, eval_labels)
    predictions = clf.predict(eval_z)
    tb_writer.add_scalar('eval/{}_acc'.format(label_name), eval_acc, curr_step)


    return train_acc, eval_acc,predictions

def calc_tpr(preds, labels):
    num = len(preds)
    assert num == len(labels)
    tp, tn, fp, fn = [],[],[],[]
    for i, (p,l) in enumerate(zip(preds, labels)):
        if p == 1:
            if l == 1:
                tp.append(i)
            else:
                fp.append(i)
        else:
            if l == 1:
                fn.append(i)
            else:
                tn.append(i)
    tpr = len(tp)/(len(tp) + len(fn))
    tnr = len(tn)/(len(tn) + len(fp))
    acc = len(tp + tn)/num

    return tpr, tnr, acc

def get_embeddings(model, dataloader, projector, device):
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    all_z = None
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]}
        with torch.no_grad():
            last_hidden, pooler_output, all_hidden = model(**inputs)
            z = projector(last_hidden,batch[1])
            if all_z is None:
                all_z = z
            else:
                all_z=torch.cat((all_z,z), dim=0)
    return all_z

def eval_both_probe(model, projector, eval_dataloader, args, tb_writer, ev_protected_labels, ev_main_labels, curr_step,\
    train_dataloader = None,train_protected_labels=None, train_main_labels=None):
    #if no training set is provided, use eval set as the new training set
    if train_dataloader is None:
        all_z = get_embeddings(model, eval_dataloader, projector, args.device)
        num_data = len(ev_protected_labels)
        train_idx = np.random.choice(range(num_data), int(num_data * 0.8), replace=False)
        eval_idx = [i for i in range(num_data) if i not in train_idx]
        train_z = all_z[train_idx,:].detach().cpu().numpy()
        eval_z = all_z[eval_idx,:].detach().cpu().numpy()
        train_protected_labels = [int(ev_protected_labels[i]) for i in train_idx]
        eval_protected_labels = [int(ev_protected_labels[i]) for i in eval_idx]
        train_main_labels = [int(ev_main_labels[i]) for i in train_idx]
        eval_main_labels = np.array([int(ev_main_labels[i]) for i in eval_idx])
    else:
        assert train_main_labels is not None
        train_z = get_embeddings(model, train_dataloader, projector, args.device).detach().cpu().numpy()
        eval_z = get_embeddings(model, eval_dataloader, projector, args.device).detach().cpu().numpy()
        eval_protected_labels = np.array(ev_protected_labels)
        eval_main_labels = np.array(ev_main_labels)


    #else: calc reps on both training set and eval set, and retrain on training set and evaluate on eval set.
    

    clf_protected = LogisticRegression(random_state=0).fit(train_z, train_protected_labels)
    train_protected_acc = clf_protected.score(train_z, train_protected_labels)
    eval_protected_acc = clf_protected.score(eval_z, eval_protected_labels)

    clf_main = LogisticRegression(random_state=0).fit(train_z, train_main_labels)
    train_main_acc = clf_main.score(train_z, train_main_labels)
    eval_main_acc = clf_main.score(eval_z, eval_main_labels)
    main_predictions = clf_main.predict(eval_z)

    prot_groups = [ [idx for idx,eval_label in enumerate(eval_protected_labels) if eval_label == prot_label ]  for prot_label in range(2)]
    
    tpr_0, tnr_0, acc_0 = calc_tpr(main_predictions[prot_groups[0]], eval_main_labels[prot_groups[0]])
    tpr_1, tnr_1, acc_1 = calc_tpr(main_predictions[prot_groups[1]], eval_main_labels[prot_groups[1]])
    
    tpr_gap = tpr_0 - tpr_1
    tnr_gap = tnr_0 - tnr_1
    acc_gap = acc_0 - acc_1

    tb_writer.add_scalar('eval/protected_acc', eval_protected_acc, curr_step)
    tb_writer.add_scalar('eval/main_acc', eval_main_acc, curr_step)
    tb_writer.add_scalar('eval/TPR_0', tpr_0, curr_step)
    tb_writer.add_scalar('eval/TPR_1', tpr_1, curr_step)
    tb_writer.add_scalar('eval/TPR_gap', tpr_gap, curr_step)
    tb_writer.add_scalar('eval/TNR_0', tnr_0, curr_step)
    tb_writer.add_scalar('eval/TNR_1', tnr_1, curr_step)
    tb_writer.add_scalar('eval/TNR_gap', tnr_gap, curr_step)
    tb_writer.add_scalar('eval/acc_0', acc_0, curr_step)
    tb_writer.add_scalar('eval/acc_1', acc_1, curr_step)
    tb_writer.add_scalar('eval/acc_gap', acc_gap, curr_step)




def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx

def get_score_hidden(hidden,mask,label, model, classifier, device):
    accumulated_hidden = (hidden*mask.unsqueeze(-1)).sum(1)
    sentence_length = mask.sum(1)
    scores = F.softmax(classifier(accumulated_hidden / sentence_length.unsqueeze(-1)))
    pred = scores.argmax(1)
    acc = ((pred==label).sum().item())/len(pred)
    return scores,acc



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pretrained_model", default="bert-base-uncased", type=str,    
                        help="The pretrained transformer model")

    parser.add_argument("--dataset", default='yelp', type=str,    
                        help="pick from yelp, imdb, moji, moji_finetune")
    parser.add_argument("--dataset_fp", default='/home/xiongyi/dataxyz/data/demog-text-removal/sentiment_race/r0s0_0.5', type=str,    
                        help="data set file path")
    parser.add_argument("--g0_pos_ratio", "-r", default=-1, type=float,    
                        help="for imdb, pos ratio for genre 0")
    

    parser.add_argument("--learning_rate", default=1e-5, type=float,    
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3, type=int,    
                        help="epochs to train")
    parser.add_argument("--stepsize", default=0.1, type=float,    
                        help="stepsize for pplm")
    parser.add_argument("--pert_iter", default=3, type=int,    
                        help="num of iterations for pplm")
    parser.add_argument("--gamma", default=0.9, type=float,    
                        help="gamma for pplm")
    parser.add_argument("--train_batch_size", default=32, type=int,    
                        help="train_batch_size")
    parser.add_argument("--eval_batch_size", default=32, type=int,    
                        help="eval_batch_size")
    parser.add_argument("--eval_step", default=100, type=int,    
                        help="eval_step")
    parser.add_argument("--reg_weight", default=1e-4, type=float,    
                        help="regularizer weight")
    parser.add_argument("--num_data", default=50000, type=int,    
                        help="num of data to use")
    curr_time = datetime.now().strftime("%d%m_%H%M%S")
    parser.add_argument("--out_dir", default=curr_time, type=str,    
                        help="output dir")
    curr_time = datetime.now().strftime("%m%d%H%M%S")
    parser.add_argument("--log_dir", default=curr_time, type=str,    
                        help="tensorboard dir")
    parser.add_argument("--loss_type", default='clr', type=str,    
                        help="clr or triplet")
    args = parser.parse_args()
    args.device = torch.device("cuda")
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    #add a step for training classifiers, two types --- finetuned and average hidden rep

    model = BertModel.from_pretrained(args.pretrained_model,  output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    if args.dataset == 'yelp':
        labels,sents = load_yelp(file_dir = '/home/xiongyi/dataxyz/data/yelp_style_transfer', file_name = 'yelp.train', \
        line_num = args.num_data)
    elif args.dataset == 'imdb':
        file_name = 'train_Drama_pos_ratio_{}.tsv'.format(args.g0_pos_ratio)
        sents, sentiments, genres = load_imdb(file_dir = '/home/xiongyi/dataxyz/data/imdb/tsv', file_name = file_name, \
        line_num = args.num_data)
        labels = genres
    elif args.dataset == 'moji':
        races,sentiments, sents = load_moji(file_dir='/home/xiongyi/dataxyz/repos/NotMine/demog-text-removal/data/processed/sentiment_race')
        labels = races

    elif args.dataset == 'moji_finetune':
        train_races,train_sentiments, train_sents = load_moji_split(file_dir = args.dataset_fp, prefix = 'train')
        eval_races,eval_sentiments, eval_sents = load_moji_split(file_dir = args.dataset_fp, prefix = 'eval')
        train_labels = train_races
        eval_labels = eval_races
        train_main_labels = train_sentiments
        eval_main_labels = eval_sentiments

    if not args.dataset == 'moji_finetune':
        train_num = int(len(sents) * 0.9)
        print ('train num:', train_num)
        train_labels = labels[:train_num]
        train_sents = sents[:train_num]
        eval_labels = labels[train_num:]
        eval_sents = sents[train_num:]
    if args.dataset == 'imdb':
        max_seq_length = 512
        eval_main_labels = sentiments[train_num:]
        label_list = [0, 1]
        classifier_model_path ='imdb_bert_uncased/generic_classifier_head_epoch_3.pt'
        classifier_meta_path ='imdb_bert_uncased/generic_classifier_head_meta.json'
       
    elif args.dataset == 'yelp':
        max_seq_length = 128
        label_list = ['0', '1']
        classifier_model_path ='yelp_bert_uncased/generic_classifier_head_epoch_3.pt'
        classifier_meta_path ='yelp_bert_uncased/generic_classifier_head_meta.json'

    elif args.dataset == 'moji':
        max_seq_length = 512
        label_list = [0, 1]
        eval_main_labels = sentiments[train_num:]
        classifier_model_path ='moji_bert_uncased/generic_classifier_head_epoch_3.pt'
        classifier_meta_path ='moji_bert_uncased/generic_classifier_head_meta.json'
        
    elif args.dataset == 'moji_finetune':
        max_seq_length = 128
        label_list = [0, 1]
        classifier_model_path ='moji-10-finetuned/generic_classifier_head_epoch_10.pt'
        classifier_meta_path ='moji-10-finetuned/generic_classifier_head_meta.json'
        
    print ('label list: ', label_list)
    train_input_examples = [InputExample(guid=i, text_a=train_sents[i], label=train_labels[i])   for i in range(len(train_sents))]
    train_input_features = convert_examples_to_features(examples = train_input_examples, label_list=label_list, max_seq_length = max_seq_length, tokenizer=tokenizer)
    train_dataloader = get_dataloader(train_input_features, batch_size=args.train_batch_size)
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")


    eval_input_examples = [InputExample(guid=i, text_a=eval_sents[i], label=eval_labels[i])   for i in range(len(eval_sents))]
    eval_input_features = convert_examples_to_features(examples = eval_input_examples, label_list=label_list, max_seq_length = max_seq_length, tokenizer=tokenizer)
    eval_dataloader = get_dataloader(eval_input_features, batch_size=args.eval_batch_size)

    projector = Projector(input_dim=768)
    projector.train()
    projector.to(args.device)
    tb_writer = SummaryWriter(os.path.join(args.out_dir, args.log_dir))

    optimizer = Adam(projector.parameters(), lr=args.learning_rate)
    classifier = get_classifier_new(classifier_model_path,classifier_meta_path, device = args.device)
    if args.loss_type == 'triplet':
        loss_fn = torch.nn.TripletMarginLoss(reduction='sum')
    elif args.loss_type == 'clr':
        loss_fn = NTXentLoss(device=args.device, batch_size = args.train_batch_size, temperature=1, use_cosine_similarity=True)

    
    model.to(args.device)
    model.eval()
    global_step = 0    
    all_acc_before,all_acc_after = [],[]



    for iter in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            if (global_step % args.eval_step == 0 and global_step) or global_step == 1:
                eval_loss(model, classifier, projector, eval_dataloader, args, tb_writer, global_step)
                if args.dataset == 'yelp':
                    probe_res = eval_probe(model, projector, eval_dataloader, args, tb_writer, eval_labels, global_step)
                elif args.dataset == 'imdb':
                    ##use train set or eval set for probing?????????
                    eval_both_probe(model, projector, eval_dataloader, args, tb_writer, eval_labels, eval_main_labels, global_step,\
                         train_dataloader=train_dataloader, train_protected_labels=train_labels, train_main_labels=train_main_labels)
                elif args.dataset in ['moji', 'moji_finetune']:
                    eval_both_probe(model, projector, eval_dataloader, args, tb_writer, eval_labels, eval_main_labels, global_step,\
                        train_dataloader=train_dataloader, train_protected_labels=train_labels, train_main_labels=train_main_labels)
                else:
                    print ('dataset not supported!')
                    return 
            if (global_step % 300 == 0 and global_step) or global_step == 1:
                save_path ='{}/Step_{}'.format(args.out_dir, global_step)
                
                torch.save(projector.state_dict(), save_path)

            batch = tuple(t.to(args.device) for t in batch)
            if len(batch[0]) < args.train_batch_size:
                continue
            inputs = {'input_ids':      batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]}
            with torch.no_grad():
                last_hidden, pooler_output, all_hidden = model(**inputs)
            scores_before, acc_before = get_score_hidden(hidden= last_hidden, mask=batch[1],label=batch[3],model=model,classifier=classifier,device=args.device)
            pert_last_hidden = perturb_hidden(model=model,all_hidden=all_hidden,masks=batch[1],labels=batch[3],\
                classifier=classifier,args = args,only_last= True,num_iterations=args.pert_iter)
            scores_after, acc_after= get_score_hidden(hidden=pert_last_hidden, mask=batch[1],label=batch[3],model=model,classifier=classifier,device=args.device)
            if global_step < 50:
                print(" acc_before:{}, ".format(acc_before))
                print("acc_after:{}, ".format(acc_after))
                # print("diff:{} ".format(torch.norm(pert_last_hidden-last_hidden)))
            all_acc_before.append(acc_before)
            all_acc_after.append(acc_after)
            if global_step % 100 == 0:
                print("global_step:{}, average acc_before:{}, ".format(global_step, np.mean(all_acc_before)))
                print("global_step:{}, average acc_after:{}, ".format(global_step, np.mean(all_acc_after)))
                tb_writer.add_scalar('train_accs/acc_before',np.mean(all_acc_before), global_step )
                tb_writer.add_scalar('train_accs/acc_after',np.mean(all_acc_after), global_step )
            z = projector(last_hidden,batch[1])
            pert_z = projector(pert_last_hidden,batch[1])
            
            if args.loss_type == 'triplet':
                dist_mat = pdist_torch(z, pert_z)
                for j in range(len(batch[0])):
                    dist_mat[j,j] += 100
                
                #neg_idx = torch.randperm(n=len(batch[0]))
                neg_idx = torch.argmin(dist_mat, 1)
                neg_z = z[neg_idx,:]
                loss = loss_fn(z,pert_z,neg_z)
                loss = loss / len(batch[0])
            elif args.loss_type == 'clr':
                loss = loss_fn(z,pert_z)

            tb_writer.add_scalar('train_loss/loss', loss.item(), global_step)
            loss.backward()
            optimizer.step()
            model.zero_grad()
            global_step += 1

    #turn into bert feature vecs. Do the whole hugging face stuff.
    
    #goal: get reps of sents that do not have information of labels.

    ##for each batch of (sents, labels) 
    #do step 1 to 3




if __name__ == "__main__":
    main()