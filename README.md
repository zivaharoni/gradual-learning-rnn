#Gradual Learning of Recurrent Neural Networks

<br/>

<br/>

## Intro
---

We present an implementation of the algorithms that are presented in our paper:
> [Gradual Learning of Recurrent Neural Networks](https://arxiv.org/abs/1708.08863),
> by Ziv Aharoni, Gal Rattner and Haim Permutter

The results from the paper were obtained by the pytorch implementation as given in
> zivaharoni/gradual-learning-rnn/pytorch_impl.

We also present also a tensorflow implementation even though that it is significantly slower in
> zivaharoni/gradual-learning-rnn/tensorflow_impl.

The pytorch implementation achieves language modeling state-of-the-art results on the Penn Treebank and WikiText-2 corpuses.

<br/>

## Requirements
---
* python 3.6
* tensorflow 1.3.0


<br/>

## Result reproduction
---

<br/>

### Penn Treebank
* pytorch
    * Train the model:
        *  `python ./pytorch_impl/run-ptb.py`
    * If you wish to run on multiple GPUs use:
        * `python  ./pytorch_impl/run-ptb.py --gpu=0,1,2 `
* tensorflow
    * Train the model:
        *  `python ./tensorflow_impl/main.py --data ptb --verbose`

### Wikitext 2

* pytorch
    * Train the model:
        *  `python ./pytorch_impl/run-wikitext2.py`
    * If you wish to run on multiple GPUs use:
        * `python  ./pytorch_impl/run-wikitext2.py --gpu=0,1,2 `
* tensorflow
    * Train the model:
        *  `python ./tensorflow_impl/main.py --data wiki2 --verbose`


(***) See run-XXXX.py script for hyper-parameters and configurations.
Script log files are saved by default under running directory in './GL/Lx/EXPxxxx/', the logging path can be changed by setting 'dirs' under general serrings in the script

<br/>

## Acknowledgements
---
Our code is based on the implementation of Zhilin Yang, Zihang Dai et. al., found in the repo:
https://github.com/zihangdai/mos


