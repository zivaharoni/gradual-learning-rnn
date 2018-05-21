# Gradual Learning of Recurrent Neural Networks

## Intro

We present an implementation of the algorithms that are presented in our paper:
> [Gradual Learning of Recurrent Neural Networks](https://arxiv.org/abs/1708.08863),
> by Ziv Aharoni, Gal Rattner and Haim Permutter

The results from the paper were obtained by the pytorch implementation as given in 
> zivaharoni/gradual-learning-rnn/pytorch_impl.

We also present also a tensorflow implementation even though that it it is significantly slower.

The pytorch implementation achieves language modeling state-of-the-art results on the Penn Treebank and WikiText-2 corpuses.


## Requirements

Python 3.6, PyTorch 0.2.0


## Results reproduction

### Penn Treebank

Train the model:

```python run-ptb.py```

If you wish to run on multiple GPUs use:

```python run-ptb.py --gpu=0,1,2 ```

### Wikitext 2

Train the model:

```python run-wikitext2.py```

If you wish to run on multiple GPUs use:

```python run-wikitext2.py --gpu=0,1,2 ```


*See run-XXXX.py script for hyper-parameters and configurations.
Script log files are saved by default under running directory in './GL/Lx/EXPxxxx/', the logging path can be changed by setting 'dirs' under general serrings in the script

## Acknowledgements

Our code is based on the implementation of Zhilin Yang, Zihang Dai et. al., found in the repo:
https://github.com/zihangdai/mos


