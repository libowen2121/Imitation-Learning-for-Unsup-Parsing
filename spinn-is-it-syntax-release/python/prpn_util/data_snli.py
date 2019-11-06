'''
data preprocessing for nli dataset snli
pay attention to the vocabulary pickle
'''
import os
import re
import cPickle
import json
import codecs
import torch
import nltk

train_file = 'snli_1.0_train.jsonl'
valid_file = 'snli_1.0_dev.jsonl'
test_file = 'snli_1.0_test.jsonl'

word_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT',
             'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ',
             'WDT', 'WP', 'WP$', 'WRB']
punctuation_tags = ['.', ',', ':', '-LRB-', '-RRB-', '\'\'', '``']


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

    def rebuild_by_freq(self, thd=2):
        self.word2idx = {'<unk>': 0}
        self.idx2word = ['<unk>']

        for k, v in self.word2frq.iteritems():
            if v >= thd and (not k in self.idx2word):
                self.idx2word.append(k)
                self.word2idx[k] = len(self.idx2word) - 1

        print 'Number of words:', len(self.idx2word)
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, punctuations=False):
        self.path = path
        if punctuations:
            self.keep_tags = word_tags + punctuation_tags
            dict_file_name = os.path.join(self.path, 'prpn_allnli_w-p_dict.pkl')    
                # Note: use model trained on all-nli and evaluate on snli
        else:
            self.keep_tags = word_tags
            dict_file_name = os.path.join(self.path, 'prpn_allnli_dict.pkl')
        if os.path.exists(dict_file_name):
            self.dictionary = cPickle.load(open(dict_file_name, 'rb'))
        else:
            self.dictionary = Dictionary()
            self.add_words(train_file)
            self.dictionary.rebuild_by_freq()
            cPickle.dump(self.dictionary, open(dict_file_name, 'wb'))

        print 'tokenizing...'
        # self.train, self.train_sens, self.train_trees = self.tokenize(train_file)
        self.valid, self.valid_sens, self.valid_trees = self.tokenize(valid_file)
        self.test, self.test_sens, self.test_trees = self.tokenize(test_file)

    def filter_words(self, tree):
            words = []
            for w, tag in tree.pos():
                if tag in self.keep_tags:
                    w = w.lower()
                    w = re.sub('[0-9]+', 'N', w)
                    # if tag == 'CD':
                    #     w = 'N'
                    words.append(w)
            return words

    def add_words(self, file):
        # Add words to the dictionary
        print 'loading', file
        with codecs.open(os.path.join(self.path, file), encoding='utf-8') as f:
            for line in f:
                try:
                    line = line.encode('UTF-8')
                except UnicodeError as e:
                    print 'ENCODING ERROR:', e
                    print line
                loaded_example = json.loads(line)
                if loaded_example['sentence1_parse'] is not None and \
                    loaded_example['sentence2_parse'] is not None:
                    tree1 = nltk.Tree.fromstring(loaded_example['sentence1_parse'])
                    tree2 = nltk.Tree.fromstring(loaded_example['sentence2_parse'])
                else:
                    continue
                words = self.filter_words(tree1)

                words = ['<s>'] + words + ['</s>']
                for word in words:
                    self.dictionary.add_word(word)

                words = self.filter_words(tree2)

                words = ['<s>'] + words + ['</s>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, file):

        def tree2list(tree):
            if isinstance(tree, nltk.Tree):
                if tree.label() in self.keep_tags:
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
        # i = 0
        with codecs.open(os.path.join(self.path, file), encoding='utf-8') as f:
            for line in f:
                # i += 1
                # if i > 2000:
                #     break
                try:
                    line = line.encode('UTF-8')
                except UnicodeError as e:
                    print 'ENCODING ERROR:', e
                    print line
                loaded_example = json.loads(line)
                if loaded_example['sentence1_parse'] is not None and \
                    loaded_example['sentence2_parse'] is not None:
                    tree1 = nltk.Tree.fromstring(loaded_example['sentence1_parse'])
                    tree2 = nltk.Tree.fromstring(loaded_example['sentence2_parse'])
                else:
                    continue
                words = self.filter_words(tree1)
                words = ['<s>'] + words + ['</s>']
                sens.append(words)
                idx = []
                for word in words:
                    idx.append(self.dictionary[word])
                sens_idx.append(torch.LongTensor(idx))
                trees.append(tree2list(tree1))

                words = self.filter_words(tree2)
                words = ['<s>'] + words + ['</s>']
                sens.append(words)
                idx = []
                for word in words:
                    idx.append(self.dictionary[word])
                sens_idx.append(torch.LongTensor(idx))
                trees.append(tree2list(tree2))

        # remove dup
        allsent = set()
        new_sens_idx=[]
        new_sens=[]
        new_trees=[]

        for i in range(len(sens)):
            if " ".join(sens[i]) in allsent:
                continue
            allsent.add(" ".join(sens[i]))
            new_sens_idx.append( sens_idx[i] )
            new_sens.append( sens[i] )
            new_trees.append( trees[i] )

        return new_sens_idx, new_sens, new_trees
