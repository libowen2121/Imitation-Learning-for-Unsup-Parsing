import argparse
import json

import nltk
import numpy
import torch
from torch.autograd import Variable
import sys

TRAIN = 0
VALID = 1
TEST = 2

# Test model
def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def get_brackets(tree, idx=0):
    brackets = set()
    if isinstance(tree, list) or isinstance(tree, nltk.Tree):
        for node in tree:
            node_brac, next_idx = get_brackets(node, idx)
            if next_idx - idx > 1:
                brackets.add((idx, next_idx))
                brackets.update(node_brac)
            idx = next_idx
        return brackets, idx
    else:
        return brackets, idx + 1


def mean(x):
    return sum(x) / len(x)


def batchify(data, bsz):
    nbatch = len(data) // bsz
    
    data = data[0: nbatch * bsz]

    # Evenly divide the data across the bsz batches.
    def list2batch(x_list):
        maxlen = max([len(x) for x in x_list])
        input = torch.LongTensor(maxlen, bsz).zero_()
        mask = torch.FloatTensor(maxlen, bsz).zero_()
        target = torch.LongTensor(maxlen, bsz).zero_()
        for idx, x in enumerate(x_list):
            input[:len(x), idx] = x
            mask[:len(x) - 1, idx] = 1
            target[:len(x) - 1, idx] = x[1:]
        cuda = True
        if cuda:
            input = input.cuda()
            mask = mask.cuda()
            target = target.cuda()
        return input, mask, target.view(-1)

    data_batched = []
    for i in range(nbatch):
        batch = data[i * bsz: (i + 1) * bsz]
        input, mask, target = list2batch(batch)
        data_batched.append((input, mask, target))

    return data_batched


def get_batch(source, i, evaluation=False):
    input, mask, target = source[i]
    input = Variable(input, volatile=evaluation)
    mask = Variable(mask, volatile=evaluation)
    target = Variable(target)
    return input, target, mask


def get_distance_f1_batch(model, corpus, fname_prefix, mode=VALID):
    model.eval()

    train_fname = fname_prefix + '_train.json'
    valid_fname = fname_prefix + '_valid.json'
    test_fname = fname_prefix + '_test.json'

    prec_list = []
    reca_list = []
    f1_list = []

    nsens = 0
    if mode == TRAIN:
        eval_batch_size = 64     # should be an even number
        sens, trees, labels, ids = corpus.train_sens, corpus.train_trees, corpus.train_labels, corpus.train_ids
        input_data = batchify(corpus.train, eval_batch_size)
        f = open(train_fname, 'w')
    elif mode == VALID:
        eval_batch_size = 32
        sens, trees, labels, ids = corpus.valid_sens, corpus.valid_trees, corpus.valid_labels, corpus.valid_ids
        input_data = batchify(corpus.valid, eval_batch_size)
        f = open(valid_fname, 'w')
    else:
        eval_batch_size = 32        
        sens, trees, labels = corpus.test_sens, corpus.test_trees, corpus.test_labels    
        input_data = batchify(corpus.test, eval_batch_size)
        f = open(test_fname, 'w')
    example = {}

    assert len(labels) == len(ids)

    num = 0 # number of sentences
    remove_num = 0
    for i in range(len(input_data)):
        data, _, mask = get_batch(input_data, i, evaluation=True)
        hidden = model.init_hidden(eval_batch_size)
        _, hidden = model(data, hidden)

        length = torch.sum(mask, 0).int().data.cpu().numpy()
        gates = model.gates.squeeze().data.cpu().numpy()

        skip = False

        for k in range(eval_batch_size):
            if skip:
                skip = False
                continue
            if length[k] <= 1:  # find invalid sample
                remove_num += 2
                if num % 2 == 0:    # the first sample
                    num += 2
                    skip = True # skip the next sample
                    continue
                else:               # the second sample
                    num += 1    # remove the previous sample
                    example = {}    # reset example
                    continue 

            depth = gates[k, 1:length[k]]      
            sen = sens[i * eval_batch_size + k][1:-1]

            sen_tree = trees[i * eval_batch_size + k]
            parse_tree = build_tree(depth, sen)

            if num % 2 == 0:
                example['sentence1'] = sen
                example['sentence1_binary_parse'] = sen_tree
                example['sentence1_prpn_binary_parse'] = parse_tree
                example['sentence1_prpn_gates'] = depth.tolist()
            else:
                example['sentence2'] = sen
                example['sentence2_binary_parse'] = sen_tree
                example['sentence2_prpn_binary_parse'] = parse_tree
                example['sentence2_prpn_gates'] = depth.tolist()
                example['gold_label'] = labels[(num-1) / 2]
                example['pairID'] = ids[(num-1) / 2]
                f.write(json.dumps(example))
                f.write('\n')
                example = {}
            num += 1
            if num % 1000 == 0:
                print 'processed (bsz-{:d}) {:d}'.format(eval_batch_size, num)
                
            model_out, model_max_l = get_brackets(parse_tree)
            std_out, std_out_l = get_brackets(sen_tree)
            assert model_max_l == std_out_l
            std_out.add((0, model_max_l))
            model_out.add((0, model_max_l))
            overlap = model_out.intersection(std_out)
            prec = float(len(overlap)) / (len(model_out) + 1e-8)
            reca = float(len(overlap)) / (len(std_out) + 1e-8)
            if len(std_out) == 0:
                reca = 1.
                if len(model_out) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            prec_list.append(prec)
            reca_list.append(reca)
            f1_list.append(f1)

    f.close()

    print 'removing invalid samples ', remove_num
    return mean(f1_list)

def get_distance_f1_batch_single(model, corpus, fname_prefix, mode=VALID):
    model.eval()

    train_fname = fname_prefix + '_train.json'
    valid_fname = fname_prefix + '_valid.json'
    test_fname = fname_prefix + '_test.json'

    prec_list = []
    reca_list = []
    f1_list = []

    nsens = 0
    if mode == TRAIN:
        eval_batch_size = 64     # should be an even number
        sens, trees, labels = corpus.train_sens, corpus.train_trees, corpus.train_labels
        input_data = batchify(corpus.train, eval_batch_size)
        f = open(train_fname, 'w')
    elif mode == VALID:
        eval_batch_size = 32
        sens, trees, labels = corpus.valid_sens, corpus.valid_trees, corpus.valid_labels
        input_data = batchify(corpus.valid, eval_batch_size)
        f = open(valid_fname, 'w')
    else:
        eval_batch_size = 32        
        sens, trees, labels = corpus.test_sens, corpus.test_trees, corpus.test_labels    
        input_data = batchify(corpus.test, eval_batch_size)
        f = open(test_fname, 'w')
    example = {}


    num = 0 # number of sentences
    for i in range(len(input_data)):
        data, _, mask = get_batch(input_data, i, evaluation=True)
        hidden = model.init_hidden(eval_batch_size)
        _, hidden = model(data, hidden)

        length = torch.sum(mask, 0).int().data.cpu().numpy()
        gates = model.gates.squeeze().data.cpu().numpy()

        for k in range(eval_batch_size):

            depth = gates[k, 1:length[k]]      
            sen = sens[i * eval_batch_size + k][1:-1]

            sen_tree = trees[i * eval_batch_size + k]
            parse_tree = build_tree(depth, sen)

            example['sentence'] = sen
            example['sentence_binary_parse'] = sen_tree
            example['sentence_prpn_binary_parse'] = parse_tree
            example['sentence_prpn_gates'] = depth.tolist()
            if labels[num] < 2:
                example['gold_label'] = 0
                f.write(json.dumps(example))
                f.write('\n')
            elif labels[num] > 2:
                example['gold_label'] = 1
                f.write(json.dumps(example))
                f.write('\n')
            else:                
                pass
            example = {}
            num += 1
            if num % 1000 == 0:
                print 'processed (bsz-{:d}) {:d}'.format(eval_batch_size, num)
                
            model_out, model_max_l = get_brackets(parse_tree)
            std_out, std_out_l = get_brackets(sen_tree)
            assert model_max_l == std_out_l
            std_out.add((0, model_max_l))   # outmost brackets
            model_out.add((0, model_max_l)) # outmost brackets
            overlap = model_out.intersection(std_out)
            prec = float(len(overlap)) / (len(model_out) + 1e-8)
            reca = float(len(overlap)) / (len(std_out) + 1e-8)
            if len(std_out) == 0:
                reca = 1.
                if len(model_out) == 0:
                    prec = 1.
            f1 = 2 * prec * reca / (prec + reca + 1e-8)
            prec_list.append(prec)
            reca_list.append(reca)
            f1_list.append(f1)

    f.close()

    return mean(f1_list)
    