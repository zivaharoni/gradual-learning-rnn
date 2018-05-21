import argparse
import ast
import time
import math
import numpy as np
np.random.seed(331)
import torch
import torch.nn as nn

import data
import model
import os

from utils import batchify, get_batch, repackage_hidden, save_checkpoint

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank/WikiText2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=280,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=960,
                    help='number of hidden units per layer')
parser.add_argument('--nhidlast', type=int, default=960,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=3,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=25,
                    help='initial learning rate')
parser.add_argument('--clip', type=str, default='[0.25,0.1,0.15,0.15]',
                    help='maximum gradient norm, given as a list with value for each layer. '
                         'for lwgc reffer to the structure: [emb, L0, L1, ..., Ln]')
parser.add_argument('--epochs', type=int, default=250,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=12, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.4,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--dropouth', type=str, default='[0.225]',
                    help='dropout for rnn layers (0 = no dropout), given as a list with value for each layer')
parser.add_argument('--dropouti', type=float, default=0.65,
                    help='dropout for input embedding layers (0 = no dropout)')
parser.add_argument('--dropoute', type=float, default=0.1,
                    help='dropout to remove words from embedding layer (0 = no dropout)')
parser.add_argument('--dropoutl', type=float, default=-0.2,
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
parser.add_argument('--save', type=str,  required=False, default='./GL/L0/RUN-0/',
                    help='path to the directory that save the final model')
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
parser.add_argument('--lwgc', default=True, help='use layer-wise gradient clipping')
parser.add_argument('--single_gpu', action='store_true', help='use single GPU')
parser.add_argument('--gpu', type=int, default=0,
                    help='set gpu device ID (-1 for cpu)')
args = parser.parse_args()

if args.gpu != '0':
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
else:
    #to run on cpu, model must have been trained on cpu
    args.cuda=False

args.dropouth = ast.literal_eval(args.dropouth)

print('finetune load path: {}/model.pt. '.format(args.save))
print('log save path: {}/finetune_log.txt'.format(args.save))
print('model save path: {}/finetune_model.pt'.format(args.save))

log_file = os.path.join(args.save, 'finetune_log.txt')

args.clip = ast.literal_eval(args.clip)
if not args.lwgc:
    args.clip = args.clip[0]

if not args.continue_train:
    if os.path.exists(log_file):
        os.remove(log_file)


def logging(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_file, 'a+') as f_log:
            f_log.write(s + '\n')


if args.dropoutl < 0:
    args.dropoutl = args.dropouth
if args.small_batch_size < 0:
    args.small_batch_size = args.batch_size

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed_all(args.seed)

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
if args.continue_train:
    model = torch.load(os.path.join(args.save, 'finetune_model.pt'))
else:
    model = torch.load(os.path.join(args.save, 'model.pt'))
if args.cuda:
    if args.single_gpu:
        parallel_model = model.cuda()
    else:
        parallel_model = nn.DataParallel(model, dim=1).cuda()
else:
    parallel_model = model
total_params = sum(x.size()[0] * x.size()[1] if len(x.size()) > 1 else x.size()[0] for x in model.parameters())
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

        total_loss += len(data) * loss
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)

def train():
    assert args.batch_size % args.small_batch_size == 0, 'batch_size must be divisible by small_batch_size'

    # Turn on training mode which enables dropout.
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = [model.init_hidden(args.small_batch_size) for _ in range(args.batch_size // args.small_batch_size)]
    batch, i = 0, 0
    while i < train_data.size(0) - 1 - 1:
        bptt = args.bptt if np.random.random() < 0.95 else args.bptt / 2.
        # Prevent excessively small or negative sequence lengths
        seq_len = max(5, int(np.random.normal(bptt, 5)))
        # There's a very small chance that it could select a very long sequence length resulting in OOM
        seq_len = min(seq_len, args.bptt + args.max_seq_len_delta)

        lr2 = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = lr2 * seq_len / args.bptt
        model.train()
        data, targets = get_batch(train_data, i, args, seq_len=seq_len)

        optimizer.zero_grad()

        start, end, s_id = 0, args.small_batch_size, 0
        while start < args.batch_size:
            cur_data, cur_targets = data[:, start: end], targets[:, start: end].contiguous().view(-1)

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden[s_id] = repackage_hidden(hidden[s_id])

            log_prob, hidden[s_id], rnn_hs, dropped_rnn_hs = parallel_model(cur_data, hidden[s_id], return_h=True)
            raw_loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), cur_targets)

            loss = raw_loss
            # Activiation Regularization - AR
            loss = loss + sum(args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h in dropped_rnn_hs[-1:])
            # Temporal Activation Regularization (slowness) - TAR
            loss = loss + sum(args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for rnn_h in rnn_hs[-1:])
            loss *= args.small_batch_size / args.batch_size
            total_loss += raw_loss.data * args.small_batch_size / args.batch_size
            loss.backward()

            s_id += 1
            start = end
            end = start + args.small_batch_size

        # gradient norm clipping or LWGC -
        # helps prevent the exploding gradient problem in RNNs and reduce covariate shift.
        if args.lwgc:
            plist = []
            for param in model.parameters():
                plist.append(param)

            # embeddings clip
            emb_params = [plist[0], plist[-1]]
            torch.nn.utils.clip_grad_norm(emb_params, args.clip[0])
            # layers clip
            for idx in range(1,4*args.nlayers,4):
                l_params = plist[idx:idx+4]
                torch.nn.utils.clip_grad_norm(l_params, args.clip[1+idx//4])
            # MoS clip
            mos_params = plist[-4:-1]
            torch.nn.utils.clip_grad_norm(mos_params, args.clip[-1])
        else:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # total_loss += raw_loss.data
        optimizer.param_groups[0]['lr'] = lr2
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            logging('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, optimizer.param_groups[0]['lr'],
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        ###
        batch += 1
        i += seq_len

lr = args.lr
stored_loss = evaluate(val_data)
best_val_loss = []
# At any point you can hit Ctrl + C to break out of training early.
try:
    #optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, weight_decay=args.wdecay)
    optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
    if args.continue_train:
        optimizer_state = torch.load(os.path.join(args.save, 'finetune_optimizer.pt'))
        optimizer.load_state_dict(optimizer_state)

    # Loop over epochs.
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()
        if 't0' in optimizer.param_groups[0]:
            tmp = {}
            for prm in model.parameters():
                tmp[prm] = prm.data.clone()
                prm.data = optimizer.state[prm]['ax'].clone()

            val_loss2 = evaluate(val_data)
            logging('-' * 89)
            logging('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss2, math.exp(val_loss2)))
            logging('-' * 89)

            if val_loss2 < stored_loss:
                save_checkpoint(model, optimizer, args.save, finetune=True)
                logging('Saving Averaged!')
                stored_loss = val_loss2

            for prm in model.parameters():
                prm.data = tmp[prm].clone()

        if (len(best_val_loss)>args.nonmono and val_loss2 > min(best_val_loss[:-args.nonmono])):
            logging('Done!')
            break
            optimizer = torch.optim.ASGD(model.parameters(), lr=args.lr, t0=0, lambd=0., weight_decay=args.wdecay)
            #optimizer.param_groups[0]['lr'] /= 2.
        best_val_loss.append(val_loss2)
        # setting loop breaking at convergence saturation
except KeyboardInterrupt:
    logging('-' * 89)
    logging('Exiting from training early')

# Load the best saved model.
model = torch.load(os.path.join(args.save, 'finetune_model.pt'))
parallel_model = nn.DataParallel(model, dim=1).cuda()
    
# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
logging('=' * 89)
logging('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
logging('=' * 89)
