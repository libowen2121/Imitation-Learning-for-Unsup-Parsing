import os
import re
import cPickle
import copy
import sys
import codecs

import numpy
import torch
import nltk

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
currency_tags_words = ['#', '$', 'C$', 'A$']
ellipsis = ['*', '*?*', '0', '*T*', '*ICH*', '*U*', '*RNR*', '*EXP*', '*PPA*', '*NOT*']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']
punctuation_words = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``', '--', ';', '-', '?', '!', '...', '-LCB-', '-RCB-']

train_file = 'train.txt'
valid_file = 'dev.txt'
test_file = 'test.txt'

class Dictionary(object):
    def __init__(self):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']
        self.word2frq = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        if word not in self.word2frq:
            self.word2frq[word] = 1
        else:
            self.word2frq[word] += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

    def __getitem__(self, item):
        if self.word2idx.has_key(item):
            return self.word2idx[item]
        else:
            return self.word2idx['<unk>']

    def rebuild_by_freq(self, thd=3):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.iteritems():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print 'Number of words:', len(self.idx2word)
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.path = path
        dict_file_name = os.path.join(path, 'prpn_sst_dict.pkl')   # 
        if os.path.exists(dict_file_name):
            self.dictionary = cPickle.load(open(dict_file_name, 'rb'))
        else:
            self.dictionary = Dictionary()
            self.add_words(train_file)
            # self.add_words(valid_file)
            # self.add_words(test_file)
            self.dictionary.rebuild_by_freq()
            cPickle.dump(self.dictionary, open(dict_file_name, 'wb'))

        self.train, self.train_sens, self.train_trees = self.tokenize(train_file)
        self.valid, self.valid_sens, self.valid_trees = self.tokenize(valid_file)
        self.test, self.test_sens, self.test_trees = self.tokenize(test_file)

    def filter_words(self, tree):
        words = []
        for w, tag in tree.pos():
            # if tag in word_tags:
            w = w.lower()
            w = re.sub('[0-9]+', 'N', w)
            # if tag == 'CD':
            #     w = 'N'
            words.append(w)
        return words

    def add_words(self, file):
        # Add words to the dictionary
        with codecs.open(os.path.join(self.path, file), 'r') as fr:
            for line in fr:
                sen_tree = nltk.Tree.fromstring(line)
                words = self.filter_words(sen_tree)
                words = ['<s>'] + words + ['</s>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, file):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                # if tree.label() in word_tags:
                if isinstance(tree[0], str):
                    return tree.leaves()[0]
                else:
                    root = []
                    for child in tree:
                        c = tree2list(child)
                        if c != []:
                            root.append(c)
                    if len(root) > 1:
                        return root
                    elif len(root) == 1:
                        return root[0]
            return []

        sens_idx = []
        sens = []
        trees = []
        with codecs.open(os.path.join(self.path, file), 'r') as fr:
            for line in fr:
                sen_tree = nltk.Tree.fromstring(line)
                words = self.filter_words(sen_tree)
                words = ['<s>'] + words + ['</s>']
                # if len(words) > 50:
                #     continue
                sens.append(words)
                idx = []
                for word in words:
                    idx.append(self.dictionary[word])
                sens_idx.append(torch.LongTensor(idx))
                trees.append(tree2list(sen_tree))
                # if len(words) < 11:
                #     print sen_tree
                #     print trees[-1]
                #     sys.exit(0)

        return sens_idx, sens, trees

if __name__ == '__main__':
    path = '/disk/ostrom/s1636966/bowenli/data/sst/trees'
    dataset = Corpus(path)
else:
    pass