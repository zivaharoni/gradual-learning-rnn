import argparse
import time
import math
import numpy as np
import os
import csv
import torch
import pickle
import torch.nn as nn
from torch.autograd import Variable
import data

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')

parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--dir', type=str,  default='./GL/L2/Best/',
                    help='path to the final model directory')
parser.add_argument('--model', type=str, default='finetune_model.pt',
                    help='name of model to eval')
parser.add_argument('--gpu', type=int, default=0,
                    help='set gpu device ID (-1 for cpu)')
parser.add_argument('--val', action='store_true',
                    help='set for validation error, test by default')
parser.add_argument('--lamb', type=float, default=0.075,
                    help='decay parameter lambda')
parser.add_argument('--epsilon', type=float, default=0.001,
                    help='stabilization parameter epsilon')
parser.add_argument('--epsilonu', type=float, default=0.001,
                    help='stabilization parameter epsilon for adadelta dividend')
parser.add_argument('--lr', type=float, default=0.002,
                    help='learning rate eta')
parser.add_argument('--ms', action='store_true', default=False,
                    help='uses mean squared gradients instead of sum squared')
parser.add_argument('--batch_size', type=int, default=70,
                    help='batch size for gradient statistics')
parser.add_argument('--bptt', type=int, default=5,
                    help='sequence/truncation length')
parser.add_argument('--max_batches', type=int, default=-1,
                    help='maximum number of training batches for gradient statistics')
parser.add_argument('--msg_calc', type=bool, default=True,
                    help='calculate msg before dynamic evaluation')
parser.add_argument('--msg_path', type=str, default='./msg/',
                    help='path to load msg from')
parser.add_argument('--mode', type=str, default='msg',
                    help='dynamic evaluation mode, options are: sgd, msg, adadelta')
parser.add_argument('--gamma', type=float, default=0.95,
                    help='moving average decay parameter')

# parser.add_argument('--n_experts', type=int, default=10, help='number of experts')


args = parser.parse_args()

if args.gpu>=0:
    args.cuda = True
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
else:
    #to run on cpu, model must have been trained on cpu
    args.cuda=False


def log_results(s, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(os.path.join(args.dir, 'dynamic_eval_results.txt'), 'a+') as f_res:
            f_res.write(s + '\n')


def log_csv(perp):
    fname = args.dir + 'results.csv'
    with open(fname, 'a', newline='') as fd:
        fdwriter = csv.writer(fd, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        dicta = vars(args)
        line = []
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


model_name=os.path.join(args.dir + args.model)

start_time = time.time()
print('loading')

corpus = data.Corpus(args.data)
eval_batch_size = 1
test_batch_size = 1

lr = args.lr
lamb = args.lamb
epsilon = args.epsilon
epsilonu = args.epsilonu

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data
#######################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

def get_msg_file():
    msg_file_name = 'MSG_bptt' + str(args.bptt)
    if not args.ms:
        msg_file_name += '_batch.npy'
    else:
        msg_file_name += '.npy'
    return os.path.join(args.msg_path,msg_file_name)

def gradstat():

    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0

    for param in model.parameters():
        param.MS = 0*param.data

    while i < train_data.size(0) - 1 - 1:
        seq_len = args.bptt
        model.eval()

        data, targets = get_batch(train_data, i)
        hidden = repackage_hidden(hidden)
        model.zero_grad()

        #assumes model has atleast 2 returns, and first is output and second is hidden
        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

        loss.backward()

        for param in model.parameters():
            param.MS = param.MS + param.grad.data*param.grad.data

        total_loss += loss.data

        batch += 1


        i += seq_len
        if args.max_batches>0:
            if batch>= args.max_batches:
                break
    gsum = 0

    msg_list = []
    for param in model.parameters():
        if args.ms:
            param.MS = torch.sqrt(param.MS/batch)
        else:
            param.MS = torch.sqrt(param.MS)
        msg_list.append(param.MS.cpu().numpy())
        gsum+=torch.mean(param.MS)

    decrate_list = []
    for param in model.parameters():
        param.decrate = param.MS/gsum
        decrate_list.append(param.decrate.cpu().numpy())

    print(40*'-')
    # msg_file_name = get_msg_file()
    # print('saving MSG statistics to: ' + msg_file_name)
    # fp = open(msg_file_name, 'wb')
    # pickle.dump([msg_list, decrate_list], fp)
    # fp.close()


def gradstatload():
    msg_file_name = get_msg_file()
    fp = open(msg_file_name, 'rb')
    list = pickle.load(fp)
    msg_list = list[0]
    decrate_list = list[1]
    fp.close()

    idx = 0
    for param in model.parameters():
        if args.cuda:
            param.MS = torch.from_numpy(msg_list[idx]).type(torch.cuda.FloatTensor)
            param.decrate = torch.from_numpy(decrate_list[idx]).type(torch.cuda.FloatTensor)
        else:
            param.MS = torch.from_numpy(msg_list[idx]).type(torch.FloatTensor)
            param.decrate = torch.from_numpy(decrate_list[idx]).type(torch.FloatTensor)
        idx += 1


def evaluate_msg():

    #clips decay rates at 1/lamb
    #otherwise scaled decay rates can be greater than 1
    #would cause decay updates to overshoot
    for param in model.parameters():
        if args.cuda:
            decratenp = param.decrate.cpu().numpy()
            ind = np.nonzero(decratenp>(1/lamb))
            decratenp[ind] = (1/lamb)
            param.decrate = torch.from_numpy(decratenp).type(torch.cuda.FloatTensor)
            param.data0 = 1*param.data
        else:
            decratenp = param.decrate.numpy()
            ind = np.nonzero(decratenp>(1/lamb))
            decratenp[ind] = (1/lamb)
            param.decrate = torch.from_numpy(decratenp).type(torch.FloatTensor)
            param.data0 = 1*param.data

    total_loss = 0

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    last = False
    seq_len= args.bptt
    seq_len0 = seq_len
    #loops through data
    while i < eval_data.size(0) - 1 - 1:

        model.eval()
        #gets last chunk of seqlence if seqlen doesn't divide full sequence cleanly
        if (i+seq_len)>=eval_data.size(0):
            if last:
                break
            seq_len = eval_data.size(0)-i-1
            last = True

        data, targets = get_batch(eval_data,i)

        hidden = repackage_hidden(hidden)

        model.zero_grad()

        #assumes model has atleast 2 returns, and first is output and second is hidden
        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

        #compute gradient on sequence segment loss
        loss.backward()

        #update rule
        for param in model.parameters():
            dW = lamb*param.decrate*(param.data0-param.data)-lr*param.grad.data/(param.MS+epsilon)
            param.data+=dW

        #seq_len/seq_len0 will be 1 except for last sequence
        #for last sequence, we downweight if sequence is shorter
        total_loss += (seq_len/seq_len0)*loss.data
        batch += (seq_len/seq_len0)

        i += seq_len

    #since entropy of first token was never measured
    #can conservatively measure with uniform distribution
    #makes very little difference, usually < 0.01 perplexity point
    #total_loss += (1/seq_len0)*torch.log(torch.from_numpy(np.array([ntokens])).type(torch.cuda.FloatTensor))
    #batch+=(1/seq_len0)

    perp = torch.exp(total_loss/batch)
    if args.cuda:
        return perp.cpu().numpy()
    else:
        return perp.numpy()


def evaluate_adadelta():

    # clips decay rates at 1/lamb
    # otherwise scaled decay rates can be greater than 1
    # would cause decay updates to overshoot
    for param in model.parameters():
        if args.cuda:
            decratenp = param.decrate.cpu().numpy()
            ind = np.nonzero(decratenp > (1 / lamb))
            decratenp[ind] = (1 / lamb)
            param.decrate = torch.from_numpy(decratenp).type(torch.cuda.FloatTensor)
            param.data0 = 1 * param.data
        else:
            decratenp = param.decrate.numpy()
            ind = np.nonzero(decratenp > (1 / lamb))
            decratenp[ind] = (1 / lamb)
            param.decrate = torch.from_numpy(decratenp).type(torch.FloatTensor)
            param.data0 = 1 * param.data

    total_loss = 0

    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    batch, i = 0, 0
    last = False
    seq_len = args.bptt
    seq_len0 = seq_len

    # create
    for param in model.parameters():
        param.MSdelta = torch.ones_like(param.MS)

    # loops through data
    while i < eval_data.size(0) - 1 - 1:

        model.eval()
        # gets last chunk of seqlence if seqlen doesn't divide full sequence cleanly
        if (i + seq_len) >= eval_data.size(0):
            if last:
                break
            seq_len = eval_data.size(0) - i - 1
            last = True

        data, targets = get_batch(eval_data, i)

        hidden = repackage_hidden(hidden)

        model.zero_grad()

        # assumes model has atleast 2 returns, and first is output and second is hidden
        log_prob, hidden = model(data, hidden)
        loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), targets)

        # compute gradient on sequence segment loss
        loss.backward()

        # update rule
        for param in model.parameters():
            # recalculate MSg for each parameter
            # calculate current dW
            # update MSdelta for next steps
            # update parameters
            param.MS = args.gamma * param.MS + (1 - args.gamma) * (param.grad.data * param.grad.data)
            MSg = torch.sqrt(param.MS / batch) if args.ms else torch.sqrt(param.MS)
            MSdelta = torch.sqrt(param.MSdelta / batch) if args.ms else torch.sqrt(param.MSdelta)

            dW = lamb * param.decrate * (param.data0 - param.data) - lr * param.grad.data * (MSdelta + epsilonu) / (MSg + epsilon)
            param.MSdelta = args.gamma * param.MSdelta + (1 - args.gamma) * (dW * dW)

            param.data += dW

        # seq_len/seq_len0 will be 1 except for last sequence
        # for last sequence, we downweight if sequence is shorter
        total_loss += (seq_len / seq_len0) * loss.data
        batch += (seq_len / seq_len0)

        i += seq_len

    # since entropy of first token was never measured
    # can conservatively measure with uniform distribution
    # makes very little difference, usually < 0.01 perplexity point
    # total_loss += (1/seq_len0)*torch.log(torch.from_numpy(np.array([ntokens])).type(torch.cuda.FloatTensor))
    # batch+=(1/seq_len0)

    perp = torch.exp(total_loss / batch)
    if args.cuda:
        return perp.cpu().numpy()
    else:
        return perp.numpy()


#load model
with open(model_name, 'rb') as f:
    model = torch.load(f)

if not isinstance(model.dropouth, list):
    model.dropouth = model.nlayers * [model.dropouth]

ntokens = len(corpus.dictionary)
criterion = nn.CrossEntropyLoss()

val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, test_batch_size)

if args.val== True:
    eval_data= val_data
else:
    eval_data=test_data
train_data = batchify(corpus.train, args.batch_size)

print('collecting gradient statistics')
#collect gradient statistics on training data
if args.msg_calc:
    gradstat()
else:
    gradstatload()

#change batch size to 1 for dynamic eval
args.batch_size=1
print(40*'-')
#apply dynamic evaluation
evaluate = {'msg': evaluate_msg, 'adadelta': evaluate_adadelta}
evaluate_op = evaluate.get(args.mode, 'msg')

print('running dynamic evaluation ' + args.mode)
loss = evaluate_op()
print('perplexity loss: ' + str(loss[0]))
print('dynamic evaluation time: %d' % (time.time() - start_time))
log_results('Args: {}'.format(args))
log_results(args.mode + ' perplexity loss: ' + str(loss[0]))
log_csv(str(loss[0]))
