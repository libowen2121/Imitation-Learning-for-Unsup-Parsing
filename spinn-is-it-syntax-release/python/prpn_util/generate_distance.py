'''
read the trained model and output prpn trees
'''

import os
import sys
import torch
import data_snli_distance
import data_allnli_distance
import data_sst_distance
from model_PRPN import PRPN
from test_phrase_grammar_distance import TRAIN, VALID, TEST, get_distance_f1_batch, get_distance_f1_batch_single
import argparse

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='/disk/scratch/bowenli/data/snli/snli_1.0/snli_1.0',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=400,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.001,
                    help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-6,
                    help='weight decay')
parser.add_argument('--clip', type=float, default=1.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to output layers (0 = no dropout)')
parser.add_argument('--idropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--rdropout', type=float, default=0,
                    help='dropout applied to recurrent states (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--hard', action='store_true',
                    help='use hard sigmoid')
parser.add_argument('--res', type=int, default=0,
                    help='number of resnet block in predict network')
parser.add_argument('--seed', type=int, default=1000,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')
parser.add_argument('--valid_interval', type=int, default=1000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='/disk/scratch/bowenli/data/tree_distillation/prpn/prpn_up_snli_02.pt',
                    help='path to save the final model')
parser.add_argument('--save_data_prefix', type=str, default='/disk/scratch/bowenli/data/tree_distillation/prpn_distance/snli_distance/prpn_up_snli_02', 
                    help='path to save the data')
parser.add_argument('--load', type=str, default=None,
                    help='path to save the final model')
parser.add_argument('--nslots', type=int, default=15,
                    help='number of memory slots')
parser.add_argument('--nlookback', type=int, default=5,
                    help='number of look back steps when predict gate')
parser.add_argument('--resolution', type=float, default=0.1,
                    help='syntactic distance resolution')
parser.add_argument('--model', type=str, default='new_gate',
                    help='type of model to use')
parser.add_argument('--device', type=int, default=0,
                    help='select GPU')
parser.add_argument('--punctuations', action='store_true', default=False,
                    help='keep punctuations')
args = parser.parse_args()

args.cuda = True
args.tied = True
args.hard = True

torch.cuda.set_device(args.device)

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Load the best saved model.
with open(args.save, 'rb') as f:
    model = torch.load(f)

if 'snli' in args.data:
    corpus = data_snli_distance.Corpus(args.data, punctuations=args.punctuations)
elif 'all_nli' in args.data:
    corpus = data_allnli_distance.Corpus(args.data, punctuations=args.punctuations)
elif 'sst' in args.data:
    corpus = data_sst_distance.Corpus(args.data)    
else:
    pass
    # corpus = data_ptb.Corpus(args.data) # TODO: punctuations
ntokens = len(corpus.dictionary)

# corpus info
print('-' * 89)
print('vocab size: {:d}'.format(len(corpus.dictionary)))
# print('train size: {:d}'.format(len(corpus.train)))
print('valid size: {:d}'.format(len(corpus.valid)))
if hasattr(corpus, 'test'):
    print('test size: {:d}'.format(len(corpus.test)))
print('-' * 89)

# Run on test data.
if hasattr(corpus, 'test'):
    test_f1 = get_distance_f1_batch(model, corpus, args.save_data_prefix, TEST)
    print('=' * 89)
    print('| End of training | len-full test f1 {:5.4f}'.format(
        test_f1))
    print('=' * 89)

# Run on train data.
test_f1 = get_distance_f1_batch(model, corpus, args.save_data_prefix, TRAIN)
print('=' * 89)
print('| End of training | len-full train f1 {:5.4f}'.format(
    test_f1))
print('=' * 89)

# Run on dev data.
test_f1 = get_distance_f1_batch(model, corpus, args.save_data_prefix, VALID)
print('=' * 89)
print('| End of training | len-full dev f1 {:5.4f}'.format(
    test_f1))
print('=' * 89)
