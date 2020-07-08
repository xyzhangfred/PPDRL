import logging
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

def tokenize(tokenizer, data, max_length = 128):
    """
    Iterate over the data and tokenize it. Sequences longer than max_length are trimmed.
    :param tokenizer: tokenizer to use for tokenization
    :param data: list of sentences
    :return: a list of the entire tokenized data
    """
    tokenized_data = []
    for sent in data:
        tokens = tokenizer.encode(sent, add_special_tokens=True)
        # keeping a maximum length of bert tokens: 512
        tokenized_data.append(tokens[:max_length])
    return tokenized_data


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=1, pad_token_segment_id=0,logger = None,
                                 mask_padding_with_zero=True, has_label = True):
    if logger is None:
        logger = logging.getLogger('root')
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    #TODO: Modify this so that we also precompute the (Bert/STS/NLI) embeddings if needed.
    if has_label:
        label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
       
        tokens_a = tokenizer.tokenize(example.text_a)
        # Account for [CLS] and [SEP] with "- 2"
        # if len(tokens_a) > max_seq_length - 2:
        #     print (example.text_a, len(tokens_a), ' Too Long')
        tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = tokens_a + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_length - len(input_ids)
    
        input_ids = input_ids + ([pad_token] * padding_length)
        input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if has_label:
            label_id = label_map[example.label]
        else:
            label_id = None
        
        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id)
                            )
    return features


def get_dataloader(features, batch_size = 16):
    all_input_ids = torch.tensor([f.input_ids for f in features])
    all_input_masks = torch.tensor([f.input_mask for f in features])
    all_segment_ids = torch.tensor([f.segment_ids for f in features])
    all_label_ids = torch.tensor([f.label_id for f in features])

    dataset = TensorDataset(all_input_ids,all_input_masks,all_segment_ids,all_label_ids)
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
