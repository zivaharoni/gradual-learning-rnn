import argparse
import os
import time
import math
import csv
import ast
import pickle
import numpy as np
import torch
import torch.nn as nn

import data
import model

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, SRU)')
parser.add_argument('--emsize', type=int, default=280,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=960,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=960,
                    help='number of hidden units for the last rnn layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=str, default='[0.25,0.1,0.15,0.15]',
                    help='gradient clipping, for lwgc reffer to the structure [emb, L0, L1, ..., Ln]')
parser.add_argument('--epochs', type=int, default=1000,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=str, default='[0.225]',
                    help='dropout for rnn layers (0 = no dropout), given as list for each layer')
parser.add_argument('--dropouti', type=float, default=0.4,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=0.29,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--wdrop', type=float, default=0.5,
                    help='amount of weight dropout to apply to the RNN hidden to hidden matrix')
parser.add_argument('--tied', action='store_false',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=28,
                    help='random seed')
parser.add_argument('--nonmono', type=int, default=5,
                    help='random seed')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='SWEEP/EXP',
                    help='path to save the final model')
parser.add_argument('--load_prev', type=str,  default='DYN-20180227-120514',
                    help='path to pretrained layers on GL processes')
parser.add_argument('--dir', type=str,  default=None,
                    help='path to directory that contains the saved files')
parser.add_argument('--alpha', type=float, default=2,
                    help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
parser.add_argument('--beta', type=float, default=1,
                    help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
parser.add_argument('--wdecay', type=float, default=1.2e-6,
                    help='weight decay applied to all weights')
parser.add_argument('--continue_train', action='store_true',
                    help='continue train from a checkpoint')
parser.add_argument('--n_experts', type=int, default=15,
                    help='number of experts')
parser.add_argument('--small_batch_size', type=int, default=-1,
                    help='the batch size for computation. batch_size should be divisible by small_batch_size.\
                     In our implementation, we compute gradients with small_batch_size multiple times, and accumulate the gradients\
                     until batch_size is reached. An update step is then performed.')
parser.add_argument('--max_seq_len_delta', type=int, default=40,
                    help='max sequence length')
parser.add_argument('--single_gpu', default=True, action='store_true',
                    help='use single GPU')
parser.add_argument('--gpu', type=int, default=0,
                    help='set gpu device ID (-1 for cpu)')
parser.add_argument('--GL', default=False,
                    help='use gradual learning')
parser.add_argument('--lwgc', default=False, help='use layer-wise grad clipping')
parser.add_argument('--start_layer', type=int, default=0,
                    help='which layer to train in case of gradual learning training')
parser.add_argument('--new_layer', type=int, default=None,
                    help='the new layer location in a GL process')
parser.add_argument('--record_stats', default=False, help='record gradients statistics')
args = parser.parse_args()


if args.gpu>=0:
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    #to run on cpu, model must have been trained on cpu
    args.cuda=False

args.dropouth = ast.literal_eval(args.dropouth)

if (args.nlayers - 1) > len(args.dropouth):
    args.dropouth = args.nlayers * args.dropouth
if args.nhidlast < 0:
    args.nhidlast = args.emsize
if args.dropoutl < 0:
    args.dropoutl = args.dropouth[-1]
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

args.clip = ast.literal_eval(args.clip)
if not args.lwgc:
    args.clip = args.clip[0]


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.save, 'log.txt'), 'a+') as f_log:
            f_log.write(s + '\n')


def log_results(s, name='.npy', print_=False, log_=True):
    if not os.path.exists('/'.join(args.save.split('/')[:-1]) + '/graph/'):
        os.makedirs('/'.join(args.save.split('/')[:-1]) + '/graph/')
    if not args.continue_train:
        fname = '/'.join(args.save.split('/')[:-1]) + '/graph/' + args.save.split('/')[-1] + name
    else:
        # fname = './' + args.save.split('/')[-2] + '/graph/' + args.save.split('/')[-1] + name
        fname = '/'.join(args.save.split('/')[:-1]) + '/graph/' + args.save.split('/')[-1] + name
    if print_:
        print(s)
    if log_:
        with open(fname, 'wb') as f_log:
            pickle.dump(s, f_log)
        f_log.close()


def log_csv(perp):
    if args.dir is not None:
        fname = args.dir + '/' + args.dir + '-results.csv'
    else:
        fname = 'results.csv'

    if not os.path.isfile(fname):
        with open(fname, 'a', newline='') as fd:
            fdwriter = csv.writer(fd, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            dicta = vars(args)
            keyslist = sorted(dicta, key=str.lower)
            line = ["save"]
            for k in keyslist:
                line += [k]
            line += ["perp"]
            fdwriter.writerow(line)
            fd.close()

    with open(fname, 'a', newline='') as fd:
        fdwriter = csv.writer(fd, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        dicta = vars(args)
        line = [dicta["save"]]
        del (dicta["save"])

        keyslist = sorted(dicta, key=str.lower)
        for k in keyslist:
            if isinstance(dicta[k], list):
                dicta[k] = '-'.join(str(dicta[k]).split(','))
                dicta[k] = dicta[k].replace(" ", "")
                dicta[k] = dicta[k].replace("[", "")
                dicta[k] = dicta[k].replace("]", "")
            line += [str(dicta[k])]
        line += [perp]
        fdwriter.writerow(line)
        fd.close()

avglist,normrec,avgrec,varrec = [],[],[],[]

def record_stats(glist, epoch, N):
    N = (epoch-1)*663 + N
    Nm = N - 1
    for i,g in enumerate(glist):
        rawg = g.data.cpu().numpy()
        if N == 1:
            if len(avglist) == i:
                avglist.append(rawg)
            else:
                avglist[i] = avglist[i] * Nm / N + rawg / N
            normrec.append([np.linalg.norm(rawg)])
            avgrec.append([np.mean(rawg)])
            varrec.append([np.var(rawg)])
        else:
            avglist[i] = avglist[i]*Nm/N + rawg/N
            normrec[i].append(normrec[i][-1]*Nm/N + np.linalg.norm(rawg)/N)
            avgrec[i].append(np.mean(rawg))
            varrec[i].append(np.var(rawg))


def save_stats(path):
    if not os.path.exists(path):
        os.makedirs(path)

    with open(path + 'total-grads-mean.npy', 'wb') as fag:
        pickle.dump(avglist, fag)
    with open(path + 'grads-norms.npy', 'wb') as fn:
        pickle.dump(normrec, fn)
    with open(path + 'grads-mean.npy', 'wb') as fm:
        pickle.dump(avgrec, fm)
    with open(path + 'grads-var.npy', 'wb') as fv:
        pickle.dump(varrec, fv)


def load_layers(model,path):
    # get a list of the restored parameters - exclude the new layer
    new_plist = []
    for p in model.parameters():
        new_plist.append(p)

    new_layer = (args.nlayers - 1) if args.new_layer is None else args.new_layer
    idx = 1 + 4 * new_layer
    new_plist[idx:idx + 4] = []

    # load old model and restore parameters:
    temp_model = torch.load(path)
    for i, p in enumerate(temp_model.parameters()):
        new_plist[i].data = p.data
    return model

# Set the random seed manually for reproducibility.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# if torch.cuda.is_available():
#     if not args.cuda:
#         print("WARNING: You have a CUDA device, so you should probably run with --cuda")
#     else:
#         torch.cuda.manual_seed_all(args.seed)

###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.data)

eval_batch_size = 10
test_batch_size = 1
train_data = batchify(corpus.train, args.batch_size, args)
val_data = batchify(corpus.valid, eval_batch_size, args)
test_data = batchify(corpus.test, test_batch_size, args)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = torch.load(os.path.join(args.save, 'model.pt'))

if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model

total_params = sum(x.data.nelement() for x in model.parameters())
logging('Args: {}'.format(args))
logging('Model total parameters: {}'.format(total_params))

criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        targets = targets.view(-1)

        log_prob, hidden = parallel_model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets).data

        total_loss += loss * len(data)

        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


# def train(epoch):
#     assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'
#
#     gradlist = None
#     # Turn on training mode which enables dropout.
#     total_loss = 0
#     start_time = time.time()
#     ntokens = len(corpus.dictionary)
#     hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
#     batch, i, N = 0, 0, 1
#     while i < train_data.size(0) - 1 - 1:
#         bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
#         # Prevent excessively small or negative sequence lengths
#         seq_len = max(5, int(np.random.normal(bptt, 5)))
#         # There's a very small chance that it could select a very long sequence length resulting in OOM
#         seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)
#
#         lr2 = optimizer.param_groups[0]['lr']
#         optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
#         model.train()
#         data, targets = get_batch(train_data, i, args, seq_len=seq_len)
#
#         optimizer.zero_grad()
#
#         start, end, s_id = 0, args.small_batch_size, 0
#         while start < args.batch_size:
#             cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)
#
#             # Starting each batch, we detach the hidden state from how it was previously produced.
#             # If we didn't, the model would try backpropagating all the way to start of the dataset.
#             hidden[s_id] = repackage_hidden(hidden[s_id])
#
#             log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
#             raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)
#
#             loss = raw_loss
#             # Activiation Regularization
#             loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
#             # Temporal Activation Regularization (slowness)
#             loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
#             loss *= args.small_batch_size / args.batch_size
#             total_loss += raw_loss.data * args.small_batch_size / args.batch_size
#             loss.backward()
#             if args.record_stats:
#                 if gradlist is None:
#                     gradlist = []
#                     for p in model.parameters():
#                         gradlist.append(p.grad)
#                 record_stats(gradlist, epoch, N)
#             N += 1
#             s_id += 1
#             start = end
#             end = start + args.small_batch_size
#
#             gc.collect()
#
#         # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#         if args.lwgc:
#             plist = []
#             for param in model.parameters():
#                 plist.append(param)
#             # embeddings clip:
#             emb_params = [plist[0], plist[-1]]
#             torch.nn.utils.clip_grad_norm(emb_params, args.clip[0])
#             # layers clip:
#             for idx in range(1,4*args.nlayers,4):
#                 l_params = plist[idx:idx+4]
#                 torch.nn.utils.clip_grad_norm(l_params, args.clip[1+idx//4])
#             # MoS clip:
#             mos_params = plist[-4:-1]
#             torch.nn.utils.clip_grad_norm(mos_params, args.clip[-1])
#         else:
#             torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
#         optimizer.step()
#
#         # total_loss += raw_loss.data
#         optimizer.param_groups[0]['lr'] = lr2
#         if batch % args.log_interval == 0 and batch > 0:
#             cur_loss = total_loss[0] / args.log_interval
#             elapsed = time.time() - start_time
#             logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
#                     'loss {:5.2f} | ppl {:8.2f}'.format(
#                 epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
#                 elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
#             total_loss = 0
#             start_time = time.time()
#         ###
#         batch += 1
#         i += seq_len


# Loop over epochs.
lr = args.lr
best_val_loss = []
stored_loss = 100000000
val_perp_list = []
# At any point you can hit Ctrl + C to break out of training early.
try:
    # if args.continue_train:
    #     optimizer_state = torch.load(os.path.join(args.save, 'optimizer.pt'))
    #     if 't0' in optimizer_state['param_groups'][0]:
    #         optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    #     else:
    #         optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    #     optimizer.load_state_dict(optimizer_state)
    #     fname = '/'.join(args.save.split('/')[:-1]) + '/graph/' + args.save.split('/')[-1] + '.npy'
    #     with open(fname, 'rb') as fv:
    #         val_perp_list = pickle.load(fv)
    #     fv.close()
    # else:
    #     optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)

    epoch_start_time = time.time()
    val_loss2 = evaluate(val_data)
    logging('-' * 89)
    logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(0, (time.time() - epoch_start_time),
                                       val_loss2, math.exp(val_loss2)))

    test_loss = evaluate(test_data, test_batch_size)
    logging('=' * 89)
    logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    logging('=' * 89)
    logging('recording to CSV file')
    logging('=' * 89)

except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# log_csv(math.exp(test_loss))
