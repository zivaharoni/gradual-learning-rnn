###############################################################################
# GL-LWGC-LSTM
###############################################################################
# PennTreebank/WikiText-2 word-level prediction state-of-the-art training code
# code is based on open source implementation of MoS article:
# https://arxiv.org/pdf/1711.03953.pdf by Zhilin Yang et. al.
# found at https://github.com/zihangdai/mos
# for further info please see LICENSE and README
# Gal Rattner
# May 2018

# script log files are saved by default under running dirsectory in ./GL/Lx/TEST/
# the logging path can be changed by setting 'dirs' under general serrings below

import os

###############################################################################
# General Settings
###############################################################################
gpu = '0,1,2'
dirs = 'WIKITEXT-2'
batch_size = 12
nhid = 1050
nhidlast = 1050
emsize = 300
n_experts = 15

###############################################################################
# Training Layer-0
###############################################################################
droph = '[0.2]'
clip = '[0.05,0.15,0.15]'
nlayers = 1
start_layer = 0
lr = 15.0
epochs = 600
drope = 0.1
dropi = 0.55
dropl = 0.29
dropo = 0.35
seed = 28
save = 'EXP0001'

# command line
line = 'python main.py'  + \
       ' --data='        + '../data/wikitext-2' + \
       ' --batch_size='  + str(batch_size) + \
       ' --clip='        + str(clip) + \
       ' --dropout='     + str(dropo) + \
       ' --dropoute='    + str(drope) + \
       ' --dropouti='    + str(dropi) + \
       ' --dropoutl='    + str(dropl) + \
       ' --dropouth='    + str(droph) + \
       ' --emsize='      + str(emsize) + \
       ' --epochs='      + str(epochs) + \
       ' --lr='          + str(lr) + \
       ' --lwgc='        + 'True' + \
       ' --n_experts='   + str(n_experts) + \
       ' --nhid='        + str(nhid) + \
       ' --nhidlast='    + str(nhidlast) + \
       ' --nlayers='     + str(nlayers) + \
       ' --gpu='         + gpu + \
       ' --save='        + save + \
       ' --seed='        + str(seed) + \
       ' --GL='          + 'True' + \
       ' --dirs='        + dirs + \
       ' --start_layer=' + str(start_layer)
os.system(line)

###############################################################################
# Fine-tune Layer-0
###############################################################################
# command line
droph = '[0.2]'
clip = '[0.05,0.15,0.15]'
nlayers = 1
start_layer = 0
epochs = 600
lr = 20.
drope = 0.1
dropi = 0.55
dropl = 0.29
dropo = 0.35
seed = 28
save = dirs + '/L0/EXP0001/'

line = 'python finetune.py' + \
       ' --data='        + '../data/wikitext-2' + \
       ' --batch_size='  + str(batch_size) + \
       ' --clip='        + str(clip) + \
       ' --dropout='     + str(dropo) + \
       ' --dropoute='    + str(drope) + \
       ' --dropouti='    + str(dropi) + \
       ' --dropoutl='    + str(dropl) + \
       ' --dropouth='    + str(droph) + \
       ' --emsize='      + str(emsize) + \
       ' --epochs='      + str(epochs) + \
       ' --lr='          + str(lr) + \
       ' --lwgc='        + 'True' + \
       ' --n_experts='   + str(n_experts) + \
       ' --nhid='        + str(nhid) + \
       ' --nhidlast='    + str(nhidlast) + \
       ' --nlayers='     + str(nlayers) + \
       ' --gpu='         + gpu + \
       ' --save='        + save + \
       ' --seed='        + str(seed) + \
       ' --GL='          + 'True' + \
       ' --dirs='        + dirs + \
       ' --start_layer=' + str(start_layer)
os.system(line)


###############################################################################
# Training Layer-1
###############################################################################
droph = '[0.24, 0.21]'
clip = '[0.035, 0.12, 0.16, 0.16]'
nlayers = 2
start_layer = 1
epochs = 600
lr = 15.
drope = 0.1
dropi = 0.55
dropl = 0.29
dropo = 0.35
seed = 28
save = 'EXP0001'


# command line
line = 'python main.py' + \
       ' --data='        + '../data/wikitext-2' + \
       ' --batch_size='  + str(batch_size) + \
       ' --clip='        + str(clip) + \
       ' --dropout='     + str(dropo) + \
       ' --dropoute='    + str(drope) + \
       ' --dropouti='    + str(dropi) + \
       ' --dropoutl='    + str(dropl) + \
       ' --dropouth='    + str(droph) + \
       ' --emsize='      + str(emsize) + \
       ' --epochs='      + str(epochs) + \
       ' --lr='          + str(lr) + \
       ' --lwgc='        + 'True' + \
       ' --n_experts='   + str(n_experts) + \
       ' --nhid='        + str(nhid) + \
       ' --nhidlast='    + str(nhidlast) + \
       ' --nlayers='     + str(nlayers) + \
       ' --gpu='         + gpu + \
       ' --save='        + save + \
       ' --seed='        + str(seed) + \
       ' --GL='          + 'True' + \
       ' --dirs='        + dirs + \
       ' --start_layer=' + str(start_layer)
os.system(line)

###############################################################################
# Fine-tune Layer-1
###############################################################################
# command line
droph = '[0.24, 0.21]'
clip = '[0.035, 0.12, 0.16, 0.16]'
nlayers = 2
start_layer = 1
epochs = 600
lr = 15.
drope = 0.1
dropi = 0.55
dropl = 0.29
dropo = 0.35
seed = 28
save = dirs + '/L1/EXP0001/'

line = 'python finetune.py' + \
       ' --data='        + '../data/wikitext-2' + \
       ' --batch_size='  + str(batch_size) + \
       ' --clip='        + str(clip) + \
       ' --dropout='     + str(dropo) + \
       ' --dropoute='    + str(drope) + \
       ' --dropouti='    + str(dropi) + \
       ' --dropoutl='    + str(dropl) + \
       ' --dropouth='    + str(droph) + \
       ' --emsize='      + str(emsize) + \
       ' --epochs='      + str(epochs) + \
       ' --lr='          + str(lr) + \
       ' --lwgc='        + 'True' + \
       ' --n_experts='   + str(n_experts) + \
       ' --nhid='        + str(nhid) + \
       ' --nhidlast='    + str(nhidlast) + \
       ' --nlayers='     + str(nlayers) + \
       ' --gpu='         + gpu + \
       ' --save='        + save + \
       ' --seed='        + str(seed) + \
       ' --GL='          + 'True' + \
       ' --dirs='        + dirs + \
       ' --start_layer=' + str(start_layer)
os.system(line)


###############################################################################
# Training Layer-2
###############################################################################
droph = '[0.25, 0.23, 0.22]'
clip = '[0.035, 0.13, 0.15, 0.16, 0.17]'
nlayers = 3
start_layer = 2
epochs = 600
lr = 15.
drope = 0.1
dropi = 0.45
dropl = 0.35
dropo = 0.45
seed = 28
save = 'EXP0001'

# command line
line = 'python main.py' + \
       ' --data='        + '../data/wikitext-2' + \
       ' --batch_size='  + str(batch_size) + \
       ' --clip='        + str(clip) + \
       ' --dropout='     + str(dropo) + \
       ' --dropoute='    + str(drope) + \
       ' --dropouti='    + str(dropi) + \
       ' --dropoutl='    + str(dropl) + \
       ' --dropouth='    + str(droph) + \
       ' --emsize='      + str(emsize) + \
       ' --epochs='      + str(epochs) + \
       ' --lr='          + str(lr) + \
       ' --lwgc='        + 'True' + \
       ' --n_experts='   + str(n_experts) + \
       ' --nhid='        + str(nhid) + \
       ' --nhidlast='    + str(nhidlast) + \
       ' --nlayers='     + str(nlayers) + \
       ' --gpu='         + gpu + \
       ' --save='        + save + \
       ' --seed='        + str(seed) + \
       ' --GL='          + 'True' + \
       ' --dirs='        + dirs + \
       ' --start_layer=' + str(start_layer)
os.system(line)

###############################################################################
# Fine-tune Layer-2
###############################################################################
# command line
droph = '[0.25, 0.23, 0.22]'
clip = '[0.035, 0.13, 0.15, 0.16, 0.17]'
nlayers = 3
start_layer = 2
epochs = 600
lr = 15.
drope = 0.1
dropi = 0.45
dropl = 0.35
dropo = 0.45
seed = 28
save = dirs + '/L2/EXP0001/'

line = 'python finetune.py' + \
       ' --data='        + '../data/wikitext-2' + \
       ' --batch_size='  + str(batch_size) + \
       ' --clip='        + str(clip) + \
       ' --dropout='     + str(dropo) + \
       ' --dropoute='    + str(drope) + \
       ' --dropouti='    + str(dropi) + \
       ' --dropoutl='    + str(dropl) + \
       ' --dropouth='    + str(droph) + \
       ' --emsize='      + str(emsize) + \
       ' --epochs='      + str(epochs) + \
       ' --lr='          + str(lr) + \
       ' --lwgc='        + 'True' + \
       ' --n_experts='   + str(n_experts) + \
       ' --nhid='        + str(nhid) + \
       ' --nhidlast='    + str(nhidlast) + \
       ' --nlayers='     + str(nlayers) + \
       ' --gpu='         + gpu + \
       ' --save='        + save + \
       ' --seed='        + str(seed) + \
       ' --GL='          + 'True' + \
       ' --dirs='        + dirs + \
       ' --start_layer=' + str(start_layer)
os.system(line)

###############################################################################
# END
###############################################################################
