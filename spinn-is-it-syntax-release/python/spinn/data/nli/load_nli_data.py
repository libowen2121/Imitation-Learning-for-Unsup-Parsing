#!/usr/bin/env python

import math
import json
import codecs
import os
import sys
from spinn.data.nli.convert_to_distance import compute_order, compute_order_by_denition
cwd = os.getcwd()

sys.path.append('../../../')

from spinn.util.data import PADDING_TOKEN

SENTENCE_PAIR_DATA = True
FIXED_VOCABULARY = None

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
    # Used in the unlabeled test set---needs to map to some arbitrary label.
    "hidden": 0,
}


def roundup2(N):
    """ Round up using factors of 2. """
    return int(2 ** math.ceil(math.log(N, 2)))


def full_tokens(tokens):
    """
    Pads a sequence of tokens from the left so the resulting
    length is a factor of two.
    """
    target_length = roundup2(len(tokens))
    padding_length = target_length - len(tokens)
    tokens = [PADDING_TOKEN] * padding_length + tokens
    return tokens


def full_transitions(N, left_full=False, right_full=False):
    """
    Recursively creates a full binary tree of with N
    leaves using shift reduce transitions.
    """

    if N == 1:
        return [0]

    if N == 2:
        return [0, 0, 1]

    assert not (left_full and right_full), "Please only choose one."

    if not left_full and not right_full:
        N = float(N)

        # Constrain to full binary trees.
        assert math.log(N, 2) % 1 == 0, \
            "Bad value. N={}".format(N)

        left_N = N / 2
        right_N = N - left_N

    if left_full:
        left_N = roundup2(N) / 2
        right_N = N - left_N

    if right_full:
        right_N = roundup2(N) / 2
        left_N = N - right_N

    return full_transitions(left_N, left_full=left_full, right_full=right_full) + \
           full_transitions(right_N, left_full=left_full, right_full=right_full) + \
           [1]


def balanced_transitions(N):
    """
    Recursively creates a balanced binary tree with N
    leaves using shift reduce transitions.
    """
    if N == 3:
        return [0, 0, 1, 0, 1]
    elif N == 2:
        return [0, 0, 1]
    elif N == 1:
        return [0]
    else:
        right_N = N // 2
        left_N = N - right_N
        return balanced_transitions(left_N) + balanced_transitions(right_N) + [1]


def convert_binary_bracketing_half_full(parse, lowercase=False):
    # Modified to provided a "half-full" binary tree without padding.
    tokens, transitions = convert_binary_bracketing(parse, lowercase)
    if len(tokens) > 1:
        transitions = full_transitions(len(tokens), left_full=True)
    return tokens, transitions


def convert_binary_bracketing_half_full_right(parse, lowercase=False):
    # Modified to provided a "half-full" binary tree without padding.
    # Difference between the other method is the right subtrees are
    # the half full ones.
    tokens, transitions = convert_binary_bracketing(parse, lowercase)
    if len(tokens) > 1:
        transitions = full_transitions(len(tokens), right_full=True)
    return tokens, transitions


def convert_binary_bracketing_full(parse, lowercase=False):
    tokens, transitions = convert_binary_bracketing(parse, lowercase)
    if len(tokens) > 1:
        tokens = full_tokens(tokens)
        transitions = full_transitions(len(tokens))
    return tokens, transitions


def convert_binary_bracketing_balanced(parse, lowercase=False):
    tokens, transitions = convert_binary_bracketing(parse, lowercase)
    if len(tokens) > 1:
        transitions = balanced_transitions(len(tokens))
    return tokens, transitions


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Downcase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)

    return tokens, transitions


def load_data(path, lowercase=False, choose=lambda x: True, mode='default', tree_joint=False, distance_type='definition'):
    if mode == 'default':
        convert = convert_binary_bracketing
    elif mode == 'full':
        convert = convert_binary_bracketing_full
    elif mode == 'half_full':
        convert = convert_binary_bracketing_half_full
    elif mode == 'half_full_right':
        convert = convert_binary_bracketing_half_full_right
    elif mode == 'balanced':
        convert = convert_binary_bracketing_balanced
    else:
        raise NotImplementedError("The mode={} is not implemented.".format(mode))
    #print "Loading", path
    examples = []
    failed_parse = 0

    with codecs.open(path, encoding='utf-8') as f:
        for line in f:
            try:
                line = line.encode('UTF-8')
            except UnicodeError as e:
                print "ENCODING ERROR:", line, e
                line = "{}"
            loaded_example = json.loads(line)
            if loaded_example["gold_label"] not in LABEL_MAP:
                    # 158 here
                continue

            if not choose(loaded_example):
                continue


            example = {}
            example["label"] = loaded_example["gold_label"]
            example["premise"] = loaded_example["sentence1"]
            example["hypothesis"] = loaded_example["sentence2"]
            example["example_id"] = loaded_example.get('pairID', 'NoID')
            example['premise_tree'] = loaded_example['sentence1_binary_parse']
            example['hypothesis_tree'] = loaded_example['sentence2_binary_parse']
            example['premise_tokens'] = loaded_example['sentence1']
            example['hypothesis_tokens'] = loaded_example['sentence2']

            if len(example['premise_tokens']) < 1 or len(example['hypothesis_tokens']) < 1:
                continue    # 30 empty sentences

            if tree_joint:
                # example['premise_dist'] = loaded_example['sentence1_prpn_gates']
                # example['hypothesis_dist'] = loaded_example['sentence2_prpn_gates']

                example['premise_prpn_tree'] = loaded_example['sentence1_prpn_binary_parse']
                example['hypothesis_prpn_tree'] = loaded_example['sentence2_prpn_binary_parse']

                if distance_type == 'definition':
                    example['premise_dist'] = compute_order_by_denition(example['premise_prpn_tree'])
                    example['hypothesis_dist'] = compute_order_by_denition(example['hypothesis_prpn_tree'])
                else:
                    example['premise_dist'] = compute_order(example['premise_prpn_tree'])
                    example['hypothesis_dist'] = compute_order(example['hypothesis_prpn_tree'])           

            # else:
                # if loaded_example["sentence1_binary_parse"] and loaded_example["sentence2_binary_parse"]:
                #     (example["premise_tokens"], example["premise_transitions"]) = convert(
                #         loaded_example["sentence1_binary_parse"], lowercase=lowercase)
                #     (example["hypothesis_tokens"], example["hypothesis_transitions"]) = convert(
                #         loaded_example["sentence2_binary_parse"], lowercase=lowercase)
                #     examples.append(example)
                # else:
                #     failed_parse += 1
            examples.append(example)
    if failed_parse > 0:
        print(
            "Warning: Failed to convert binary parse for {} examples.".format(failed_parse))
    return examples


if __name__ == "__main__":
    # Demo:
    examples = load_data( '../../../../smallsnli/debug_train.json', tree_joint=True)
    print examples[0]

    for ex in examples[:10]:
        print ex['premise_tokens']
        print ex['hypothesis_tokens']