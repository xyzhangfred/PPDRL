import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from transformers import BertTokenizer, BertModel
from bert_util import tokenize, InputExample, convert_examples_to_features,get_dataloader
from data_util import load_yelp, load_imdb

def eval_probe(eval_dataloader,projection_mat, model,device, protected_labels,main_labels):
    epoch_iterator = tqdm(eval_dataloader, desc="Iteration")
    global_step = 0
    all_z = None
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(device) for t in batch)
        inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]}
        with torch.no_grad():
            last_hidden, pooler_output, all_hidden = model(**inputs)
            sent_length = batch[1].sum(1).unsqueeze(1)

        sum_hidden = (last_hidden * batch[1].unsqueeze(-1)).sum(1)
        avg_hidden = sum_hidden/sent_length.detach()
        avg_hidden = avg_hidden.cpu().numpy()
        z = np.matmul(avg_hidden,projection_mat)
        if all_z is None:
            all_z = z
        else:
            all_z=np.concatenate((all_z,z), 0)
    num_data = len(protected_labels)


    train_idx = np.random.choice(range(num_data), int(num_data * 0.8), replace=False)
    eval_idx = [i for i in range(num_data) if i not in train_idx]
    train_z = all_z[train_idx,:]
    eval_z = all_z[eval_idx,:]
    train_protected_labels = [int(protected_labels[i]) for i in train_idx]
    eval_protected_labels = [int(protected_labels[i]) for i in eval_idx]
    clf_pro = LogisticRegression(random_state=0).fit(train_z, train_protected_labels)
    train_acc_pro = clf_pro.score(train_z, train_protected_labels)
    eval_acc_pro = clf_pro.score(eval_z, eval_protected_labels)

    train_main_labels = [int(main_labels[i]) for i in train_idx]
    eval_main_labels = [int(main_labels[i]) for i in eval_idx]
    clf_main = LogisticRegression(random_state=0).fit(train_z, train_main_labels)
    train_acc_main = clf_main.score(train_z, train_main_labels)
    eval_acc_main = clf_main.score(eval_z, eval_main_labels)

    eval_idx_groups = [ [idx for idx in range(len(eval_idx)) if eval_protected_labels[idx] == prot_label ]  for prot_label in range(2)]
    tprs = []
    accs = []
    for main_label, idx_group in enumerate(eval_idx_groups):
        z_subset = [eval_z[idx] for idx in idx_group]
        main_label_subset = [eval_main_labels[idx] for idx in idx_group]
        accs.append(clf_main.score(z_subset, main_label_subset))
    
    # tpr_gap = np.abs(tprs[1] - tprs[0])
    acc_gap = np.abs(accs[1] - accs[0])

    # print (tpr_gap)
    print (acc_gap)
    return train_acc_pro, eval_acc_pro,train_acc_main,eval_acc_main,acc_gap





def main():
    file_name = 'full_train_Drama_pos_ratio_{}.tsv'.format(-1)    
    sents, sentiments, genres = load_imdb(file_dir = '/home/xiongyi/dataxyz/repos/Mine/PPDRL', file_name = file_name, \
    line_num = 50000)
    labels = genres
    outputs = np.load('/home/xiongyi/dataxyz/repos/Mine/nullspace_projection_plug/imdb/P_svm.num-clfs=300.npy',allow_pickle=True)
    P = outputs[0]
    model = BertModel.from_pretrained('bert-base-uncased',  output_hidden_states=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda:1')
    model.to(device)
    model.eval()
    train_num = int (0.8 * len(sents))
    train_labels = labels[:train_num]
    train_sents = sents[:train_num]
    eval_labels = labels[train_num:]
    eval_sents = sents[train_num:]
    eval_main_labels = sentiments[train_num:]
    label_list = [0, 1]


    print ('label list: ', label_list)
    eval_input_examples = [InputExample(guid=i, text_a=eval_sents[i], label=eval_labels[i])   for i in range(len(eval_sents))]
    eval_input_features = convert_examples_to_features(examples = eval_input_examples, label_list=label_list, max_seq_length = 128, tokenizer=tokenizer)
    eval_dataloader = get_dataloader(eval_input_features, batch_size=32)

    accs = eval_probe(eval_dataloader,P, model,device, protected_labels = eval_labels,main_labels=eval_main_labels)
    print (accs)

if __name__ == "__main__":
    main()